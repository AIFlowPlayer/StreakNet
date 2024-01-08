#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os
import cv2
import yaml
import argparse
from loguru import logger
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch 
from torch.utils.data import DataLoader

from streaknet.exp import get_exp
from streaknet.data import StreakImageDataset
from streaknet.utils import standard, get_model_info, setup_logger
from streaknet.data import cal_valid_results, cal_image_valid_results


def make_parse():
    parser = argparse.ArgumentParser("Band Pass Filter Algorithm.")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=2)
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="please input your experiment description file")
    parser.add_argument("-l", "--lower", type=float, default=450, help="Lower bound of Band Pass Filter, MHz")
    parser.add_argument("-u", "--upper", type=float, default=550, help="Upper bound of Band Pass Filter, MHz")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="Select device.")
    parser.add_argument("--save", default=False, action="store_true")
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def process_weight(weight):
    weight = torch.abs(weight)
    weight_sum = torch.mean(weight, dim=0)
    weight_sum_real = weight_sum[:weight_sum.shape[0]//2]
    weight_sum_imag = weight_sum[weight_sum.shape[0]//2:]
    filter = torch.complex(weight_sum_real, weight_sum_imag)
    return filter


def get_filter(args, max_len=4000):
    if args.exp_file is None:
        # 满屏扫描时间30ns，CCD分辨率2048，采样频率68.27GHz
        # 使用长度65536计算FFT，频率分辨率68.27GHz/65536=1.04MHz
        freq_resolutio = (2048 / 30) * 1000 / 65536
        lower = round(args.lower / freq_resolutio)
        upper = round(args.upper / freq_resolutio)
        filter = torch.ones((1, 1, max_len), dtype=torch.float32)
        filter[:, :, :lower] = 0
        filter[:, :, upper:] = 0
        file_name = os.path.join("StreakNet_outputs", "traditional", "bandpass_{}_{}".format(args.lower, args.upper))
        return filter, file_name
    else:
        exp = get_exp(args.exp_file)
        if not args.experiment_name:
            args.experiment_name = exp.exp_name
            
        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model)))
        logger.info("Model Structure:\n{}".format(str(model)))
        
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = args.device
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")
        
        signal_radar_weight = model.embedding.signal_embedding_block.dense.state_dict()['weight']
        signal_radar_filter = process_weight(signal_radar_weight)
        signal_radar_filter = standard(torch.absolute(signal_radar_filter))
        return signal_radar_filter, file_name


def band_pass_filter_algorithm(dataset, batch_size, filter, device=torch.device('cpu'), num_workers=4):
    data_loader = DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
    
    gray_img = torch.zeros((2048, dataset.nums), dtype=torch.float32).to(device)
    deep_img = torch.zeros((2048, dataset.nums), dtype=torch.float32).to(device)
    truth_img = dataset.gd
    
    template = torch.tensor(dataset.template, dtype=torch.float32).to(device)[None, None, :].repeat(batch_size, 1, 1)
    template_freq = torch.fft.rfft(template, 65536, dim=2)

    for img, _, _, idx in tqdm(data_loader):
        img = img.to(device)
        
        img_freq = torch.fft.rfft(img, 65536, dim=2) # 只取正频率部分
        tem_freq = template_freq[:img_freq.shape[0]]
        
        # 带通滤波 & 匹配滤波
        img_freq[:, :, 4000:] = 0
        img_freq[:, :, :4000] *= filter
        match_filt = torch.absolute(torch.fft.irfft(tem_freq * img_freq, dim=2)[:, :, :2048])
        
        # 确定最大响应及其时间
        max_resp, max_resp_index = torch.max(match_filt, dim=2)
        gray_img[:, idx[:, 0]] = max_resp.T
        
        # 计算景深
        distance = (max_resp_index * 30 * 1e-9 / 2 /2048) * 3e8 
        deep_img[:, idx[:, 0]] = distance.T
    
    # 转换为灰度值
    gray_img = standard(gray_img) * 255
    gray_img[gray_img > 255] = 255
    gray_img = gray_img.cpu().numpy().astype(np.uint8)
    deep_img = deep_img.cpu().numpy()
    
    # 抑制背景噪声
    _, mask = cv2.threshold(gray_img, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    gray_img *= mask
    deep_img *= mask

    return gray_img, deep_img, mask, truth_img


def valid(gray_img, deep_img, mask, truth_mask):
    acc, precision, recall, f1, cls_psnr = cal_valid_results(mask, truth_mask)
    img_psnr, img_cnr = cal_image_valid_results(gray_img, truth_mask)
    dis_var = np.var(deep_img[truth_mask == 1])
    ret = dict({
        "mask": {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "F1-Score": f1,
            "PSNR": cls_psnr
        },
        "image":{
            "PSNR": img_psnr,
            "CNR": img_cnr
        },
        "deep":{
            "variance": dis_var
        }
    })
    return ret


def merge_result(results):
    acc, precision, recall, f1, cls_psnr = [], [], [], [], []
    img_psnr, img_cnr = [], []
    dis_var = []
    for result in results:
        acc.append(result["mask"]["accuracy"])
        precision.append(result["mask"]["precision"])
        recall.append(result["mask"]["recall"])
        f1.append(result["mask"]["F1-Score"])
        cls_psnr.append(result["mask"]["PSNR"])
        img_psnr.append(result["image"]["PSNR"])
        img_cnr.append(result["image"]["CNR"])
        dis_var.append(result["deep"]["variance"])
    ret = dict({
        "mask": {
            "accuracy": np.mean(acc),
            "precision": np.mean(precision),
            "recall": np.mean(recall),
            "F1-Score": np.mean(f1),
            "PSNR": np.mean(cls_psnr)
        },
        "image":{
            "PSNR": np.mean(img_psnr),
            "CNR": np.mean(img_cnr)
        },
        "deep":{
            "variance": np.mean(dis_var)
        }
    })
    return ret


def main(args):
    device = torch.device(args.device)
    filter, file_name = get_filter(args, 4000)
    filter = filter.to(device)
    if args.save:
        os.makedirs(file_name, exist_ok=True)
        setup_logger(file_name, distributed_rank=0, filename="bandpass_log.txt", mode="override")
    
    with open(os.path.join("datasets", "valid_config.yaml"), "r") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    sub_datasets = data["sub_datasets"]
    config = data["config"]
    
    results = []
    
    for config_i in config:
        index = config_i["sub_idx"]
        row = config_i["row_slice"]
        col = config_i["col_slice"]
        logger.info("Processing {}...".format(sub_datasets[index]))
        dataset = StreakImageDataset(os.path.join("datasets", sub_datasets[index]), cache=args.cache, groundtruth=True)

        gray_img, deep_img, mask, truth_img = band_pass_filter_algorithm(
            dataset, args.batch_size, filter, 
            device=device, num_workers=args.num_workers)
        
        result = valid(
            gray_img[row[0]:row[1], col[0]:col[1]], 
            deep_img[row[0]:row[1], col[0]:col[1]], 
            mask[row[0]:row[1], col[0]:col[1]], 
            truth_img[row[0]:row[1], col[0]:col[1]])
        results.append(result)
        
        fig = plt.figure(figsize=(12, 6))
        x = np.arange(0, deep_img.shape[1], 1)
        y = np.arange(0, deep_img.shape[0], 1)
        x, y = np.meshgrid(x, y)
        x = x[mask == 1]
        y = y[mask == 1]
        deep_img = deep_img[mask == 1]
        
        ax1 = fig.add_subplot(121)
        ax1.imshow(gray_img)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(x, deep_img, y, c=deep_img, cmap='viridis', s=0.1)
        ax2.set_zlim([800, 1700])
        ax2.yaxis._set_scale("log")
        ax2.set_ylim([1, 6])

        if args.save:
            plt.savefig(os.path.join(file_name, "band_pass_{}.png".format(sub_datasets[index])))
        else:
            plt.show()
    
    result = merge_result(results)
    logger.info("[Mask] Accuracy:{:.3f}% Precision:{:.3f}% Recall:{:.3f}% F1-Score:{:.3f}% PSNR:{:.4f}".format(
        result["mask"]["accuracy"] * 100, result["mask"]["precision"] * 100, 
        result["mask"]["recall"] * 100, result["mask"]["F1-Score"] * 100, result["mask"]["PSNR"]
    ))
    logger.info("[Image] PSNR:{:.4f} CNR:{:.4f}".format(result["image"]["PSNR"], result["image"]["CNR"]))
    logger.info("[Deep] Variance:{:.4f}".format(result["deep"]["variance"]))


if __name__ == "__main__":
    args = make_parse().parse_args()
    main(args)
    