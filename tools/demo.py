#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from loguru import logger
from matplotlib import pyplot as plt

import cv2

import torch 
from torch.utils.data import DataLoader

from streaknet.exp import get_exp
from streaknet.utils import get_model_info, hilbert, standard
from streaknet.data import StreakImageDataset, RandomNoise

from valid_streaknet import get_filter


def make_parse():
    parser = argparse.ArgumentParser("StreakNet Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=2)
    parser.add_argument("-f", "--exp_file", default=None, type=str, required=True, help="please input your experiment description file")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="Select device.")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for demo")
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-t", "--template", type=str, default=None)
    parser.add_argument("--noise", default=0, type=float)
    parser.add_argument("--save", default=False, action="store_true")
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def demo(dataset, batch_size, filter, model, device=torch.device('cpu'), num_workers=4):
    from streaknet.data import StreakTransform
    from streaknet.data import DataPrefetcher
    transform = StreakTransform(True)
    
    data_loader = DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    gray_img = torch.zeros((2048, dataset.nums), dtype=torch.float32).to(device)
    deep_img = torch.zeros((2048, dataset.nums), dtype=torch.float32).to(device)
    mask_img = torch.zeros((2048, dataset.nums), dtype=torch.uint8).to(device)
    
    template = torch.tensor(dataset.template, dtype=torch.float32).to(device)[None, None, :]
    template_std = transform(template)
    
    template_freq = torch.fft.rfft(template, 65536, dim=2).repeat(batch_size, 1, 1)
    template_freq_std = torch.fft.rfft(template_std, 65536, dim=2)[:, :, :4000].repeat(batch_size * 2048, 1, 1)

    for i, (img, _, idx) in enumerate(tqdm(data_loader)):
        bsize = img.shape[0]
        img = img.to(device)
        
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

    return gray_img, deep_img, mask


def main(exp, args):
    device = torch.device(args.device)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    model = exp.get_model(export=True)
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    
    filter = get_filter(model).to(device)
    model = model.to(device).eval()
    transform = RandomNoise(args.noise)
    
    dataset = StreakImageDataset(args.path, args.template, transform=transform, cache=args.cache)
    gray_img, deep_img, mask = demo(
        dataset, args.batch_size, filter, model,
        device=device, num_workers=args.num_workers)
    
    fig = plt.figure(figsize=(18, 8))
    x = np.arange(0, deep_img.shape[1], 1)
    z = np.arange(0, deep_img.shape[0], 1)
    x, z = np.meshgrid(x, z)
    x = x[mask == 1]
    z = z[mask == 1]
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
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        )
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, "result.png"))
    else:
        plt.show()


if __name__ == "__main__":
    args = make_parse().parse_args()
    exp = get_exp(args.exp_file)
    main(exp, args)
    