#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os
import cv2
import yaml
import pandas as pd
import argparse
from loguru import logger
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch 
from torch.utils.data import DataLoader

from streaknet.exp import get_exp
from streaknet.data import StreakImageDataset, RandomNoise
from streaknet.utils import standard, setup_logger, hilbert
from streaknet.utils import valid, merge_result, log_result, to_excel


def make_parse():
    parser = argparse.ArgumentParser("Reconstruction Algorithm.")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=2)
    parser.add_argument("-f", "--exp_file", default=None, type=str, required=True, help="please input your experiment description file")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="Select device.")
    parser.add_argument("--noise", default=0, type=float)
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


def get_filter(model):    
    signal_radar_weight = model.embedding.signal_embedding_block.dense.state_dict()['weight']
    signal_radar_filter = process_weight(signal_radar_weight)
    signal_radar_filter = standard(torch.absolute(signal_radar_filter))
    return signal_radar_filter


def reconstruction_algorithm(dataset, batch_size, filter, model, device=torch.device('cpu'), num_workers=4):
    from streaknet.data import StreakTransform
    from streaknet.data import DataPrefetcher
    transform = StreakTransform(True)
    
    data_loader = DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
    data_fetcher = DataPrefetcher(data_loader)
    
    gray_img = torch.zeros((2048, dataset.nums), dtype=torch.float32).to(device)
    deep_img = torch.zeros((2048, dataset.nums), dtype=torch.float32).to(device)
    mask_img = torch.zeros((2048, dataset.nums), dtype=torch.uint8).to(device)
    truth_img = dataset.gd
    
    template = torch.tensor(dataset.template, dtype=torch.float32).to(device)[None, None, :]
    template_std = transform(template)
    
    template_freq = torch.fft.rfft(template, 65536, dim=2).repeat(batch_size, 1, 1)
    template_freq_std = torch.fft.rfft(template_std, 65536, dim=2)[:, :, :4000].repeat(batch_size * 2048, 1, 1)

    for i in tqdm(range(dataset.nums // args.batch_size)):
        img, _, _, idx = data_fetcher.next()
        bsize = img.shape[0]
        
        img_std = transform(img)
        
        img_freq_std = torch.fft.rfft(img_std, 65536, dim=2)[:, :, :4000]
        tem_freq_std = template_freq_std[:bsize * 2048]
        
        img_inp = img_freq_std
        img_inp = img_inp.view((bsize * 2048, 1, -1))
        img_inp_real = torch.real(img_inp)
        img_inp_imag = torch.imag(img_inp)
        img_inp = torch.concat([img_inp_real, img_inp_imag], dim=1)
        img_inp_real, img_inp_imag = None, None
        
        tem_inp = tem_freq_std
        tem_inp = tem_inp.view((bsize * 2048, 1, -1))
        tem_inp_real = torch.real(tem_inp)
        tem_inp_imag = torch.imag(tem_inp)
        tem_inp = torch.concat([tem_inp_real, tem_inp_imag], dim=1)
        tem_inp_real, tem_inp_imag = None, None
        
        output = model(img_inp, tem_inp)
        output = output.view((batch_size, 2048))
        mask_img[:, i*batch_size:(i+1)*batch_size] = output.T
        
        # 释放内存
        img_std, img_inp, tem_inp, output = None, None, None, None
        
        # 带通滤波 & 匹配滤波
        img_freq = torch.fft.rfft(img, 65536, dim=2) 
        img_freq[:, :, 4000:] = 0
        img_freq[:, :, :4000] *= filter
        tem_freq = template_freq[:bsize]
        match_filt = torch.absolute(torch.fft.irfft(tem_freq * img_freq, dim=2)[:, :, :2048])
        match_filt = hilbert(match_filt)
        
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
    mask = mask_img.cpu().numpy() * 255
    kernel = np.ones((5, 5), np.float32) / 25
    mask = cv2.filter2D(mask, -1, kernel)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask[mask > 127] = 1
        
    gray_img *= mask
    deep_img *= mask

    return gray_img, deep_img, mask, truth_img
    

def main(args):
    device = torch.device(args.device)
    
    exp = get_exp(args.exp_file)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    
    model = exp.get_model(export=True)
    logger.info("Model Structure:\n{}".format(str(model)))
    
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    logger.info("loading checkpoint from {}".format(ckpt_file))
    loc = args.device
    ckpt = torch.load(ckpt_file, map_location=loc)
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    
    filter = get_filter(model).to(device)
    model = model.to(device).eval()
    
    if args.save:
        os.makedirs(file_name, exist_ok=True)
        if os.path.exists(os.path.join(file_name, "reconstruction_log.txt")):
            os.remove(os.path.join(file_name, "reconstruction_log.txt"))
        setup_logger(file_name, distributed_rank=0, filename="reconstruction_log.txt", mode="override")
    
    with open(os.path.join("datasets", "valid_config.yaml"), "r") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    sub_datasets = data["sub_datasets"]
    config = data["config"]
    
    results = []
    excel_results = np.zeros((1, 8*(len(sub_datasets)+1)), dtype=np.float32)
    transform = RandomNoise(args.noise)
    
    for i, config_i in enumerate(config):
        index = config_i["sub_idx"]
        row = config_i["row_slice"]
        col = config_i["col_slice"]
        logger.info("Processing {}...".format(sub_datasets[index]))
        dataset = StreakImageDataset(os.path.join("datasets", sub_datasets[index]), cache=args.cache, groundtruth=True, transform=transform)

        gray_img, deep_img, mask, truth_img = reconstruction_algorithm(
            dataset, args.batch_size, filter, model,
            device=device, num_workers=args.num_workers)
        
        result = valid(
            gray_img[row[0]:row[1], col[0]:col[1]], 
            deep_img[row[0]:row[1], col[0]:col[1]], 
            mask[row[0]:row[1], col[0]:col[1]], 
            truth_img[row[0]:row[1], col[0]:col[1]])
        log_result(result)
        excel_results = to_excel(excel_results, result, i + 1)
        results.append(result)
        
        fig = plt.figure(figsize=(18, 8))
        x = np.arange(0, deep_img.shape[1], 1)
        z = np.arange(0, deep_img.shape[0], 1)
        x, z = np.meshgrid(x, z)
        x = x[mask == 1]
        z = z[mask == 1]
        deep_img_src = deep_img.copy()
        deep_img_src[mask == 0] = 0
        deep_img = deep_img[mask == 1]
        
        ax1 = fig.add_subplot(2, 4, (1, 5))
        ax1.imshow(gray_img)
        
        ax2 = fig.add_subplot(2, 4, (2, 6))
        ax2.imshow(mask * 255, cmap='gray')
        
        ax3 = fig.add_subplot(2, 4, 3, projection='3d')
        ax3.scatter(x, deep_img, z, c=deep_img, cmap='viridis', s=0.1)
        ax3.set_zlim([800, 1700])
        ax3.yaxis._set_scale("log")
        ax3.set_ylim([1, 6])
        
        ax4 = fig.add_subplot(2, 4, 4)
        ax4.scatter(deep_img, z, c=deep_img, cmap='viridis', s=0.1)
        # ax4.set_xscale("log")
        # ax4.set_xlim([1, 6])
        ax4.set_ylim([800, 1700])
        
        ax5 = fig.add_subplot(2, 4, 7)
        ax5.scatter(x, deep_img, c=deep_img, cmap='viridis', s=0.1)
        # ax5.set_yscale("log")
        # ax5.set_ylim([1, 6])
        
        ax6 = fig.add_subplot(2, 4, 8)
        ax6.scatter(x, z, c=deep_img, cmap='viridis', s=0.1)
        ax6.set_ylim([800, 1700])
        
        plt.subplots_adjust(wspace=0.3)

        if args.save:
            os.makedirs(os.path.join(file_name, "npy"), exist_ok=True)
            if args.noise > 0:
                np.save(os.path.join(file_name, "npy", "noise_{:.2f}_reconstruction_gray_{}.png".format(args.noise, sub_datasets[index])), gray_img)
                np.save(os.path.join(file_name, "npy", "noise_{:.2f}_reconstruction_mask_{}.png".format(args.noise, sub_datasets[index])), mask)
                np.save(os.path.join(file_name, "npy", "noise_{:.2f}_reconstruction_deep_{}.png".format(args.noise, sub_datasets[index])), deep_img_src)
                plt.savefig(os.path.join(file_name, "noise_{:.2f}_reconstruction_{}.png".format(args.noise, sub_datasets[index])))
            else:
                np.save(os.path.join(file_name, "npy", "reconstruction_gray_{}.png".format(sub_datasets[index])), gray_img)
                np.save(os.path.join(file_name, "npy", "reconstruction_mask_{}.png".format(sub_datasets[index])), mask)
                np.save(os.path.join(file_name, "npy", "reconstruction_deep_{}.png".format(sub_datasets[index])), deep_img_src)
                plt.savefig(os.path.join(file_name, "reconstruction_{}.png".format(sub_datasets[index])))
        else:
            plt.show()
    
    logger.info("--> Macro-Average:")
    result = merge_result(results)
    log_result(result)
    excel_results = to_excel(excel_results, result, 0)
    
    if args.save:
        df = pd.DataFrame(excel_results)
        if args.noise > 0:
            df.to_excel(os.path.join(file_name, "noise_{:.2f}_reconstruction_log.xlsx".format(args.noise)), index=False, engine='openpyxl')
        else:
            df.to_excel(os.path.join(file_name, "reconstruction_log.xlsx"), index=False, engine='openpyxl')
    

if __name__ == "__main__":
    args = make_parse().parse_args()
    main(args)
    