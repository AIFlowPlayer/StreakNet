#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All righs reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch 
from torch.utils.data import Dataset, DataLoader

from streaknet.data import cal_valid_results, cal_rates


class StreakData(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 data_num: int, 
                 filename_format: str):
        self.data_dir = data_dir
        self.data_num = data_num
        self.filename_format = filename_format
        
    def __getitem__(self, index):
        filename = os.path.join(self.data_dir, self.filename_format.format(index + 1))
        assert os.path.exists(filename)
        img = cv2.imread(filename, -1).astype(np.float32)
        img_tensor = torch.tensor(img, dtype=torch.float32)[None, :, :]
        return img_tensor
    
    def __len__(self):
        return self.data_num


def get_kernel():
    h = np.array([[[
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 0, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1]
    ]]], dtype=np.float32)    # 模板矩阵
    h /= 24
    h = torch.tensor(h, dtype=torch.float32).cuda()
    return h 


def get_wave_template(template_path:str):
    match_wave = np.load(template_path).astype(np.float32)
    match_wave = torch.tensor(match_wave, dtype=torch.float32)[None, :].cuda()
    local_wave_spec = torch.fft.fftshift(torch.fft.fft(match_wave, N_fft))
    local_wave_spec[:1, zeroFreq_idx-6:zeroFreq_idx+6+1] = 0    # 频谱置0，高通滤波
    local_wave_spec[:1, 1-1:2001-1+1] = 0                       # 频谱置0
    local_wave_spec[:1, 2097-1:] = 0                            # 频谱置0，116.6-782.6MHz
    local_wave_spec = local_wave_spec[None, None, :, :]
    return local_wave_spec


def main(template_path:str, data_dir:str, data_num:int, rate: float, filename_format:str, batch_size:int):
    h = get_kernel()
    local_wave_spec = get_wave_template(template_path)
    gray_img = np.zeros((2044, data_num))
    dataset = StreakData(data_dir=data_dir, data_num=data_num, filename_format=filename_format)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    for i, data in enumerate(tqdm(dataloader)):
        data = data.cuda()
        
        demean = torch.mean(data, dim=(1,2))
        demean_img = data - demean[:, None, None]
        
        aver_neighb = torch.nn.functional.conv2d(demean_img, h)
        
        stripe_wave_spaec = torch.fft.fftshift(torch.fft.fft(aver_neighb, N_fft, dim=3))
        match_filt = torch.absolute(torch.fft.ifft(torch.fft.ifftshift(stripe_wave_spaec * local_wave_spec)))
        match_filt /= rate
        
        yz, _ = torch.max(match_filt, dim=3)
        yz, _ = torch.max(yz, dim=2)
        yz = yz[:, :, None] * 0.2
        
        m, _ = torch.max(match_filt, dim=3)
        
        m[m < yz] = 0
        m[m < 2] = 0
        m = m[:, 0, :]
        gray_img[:, i*batch_size:(i+1)*batch_size] = np.transpose(m.cpu().numpy(), (1, 0))
    
    gray_img[gray_img > 255] = 255
    gray_img = gray_img.astype(np.uint8)
    return gray_img


def valid(pred, label):
    pred = cv2.resize(pred, (label.shape[1], label.shape[0]))
    pred = (pred >= 128)
    
    tp, fn, fp, tn = cal_valid_results(pred, label, origin=True)
    acc, precision, recall, f1, psnr = cal_rates(tp, fn, fp, tn)
    print("Accuracy:{:.03f}% Precision:{:.03f}% Recall:{:.03f}% F1-Score:{:.03f}% PSNR:{:.03f}dB".format(
        acc * 100, precision * 100, recall * 100, f1 * 100, psnr
    ))
    
    return tp, fn, fp, tn


def make_parse():
    parser = argparse.ArgumentParser("GPU streak-image processor")
    parser.add_argument("-i", "--img-dir", type=str)
    parser.add_argument("-t", "--template-path", type=str)
    parser.add_argument("-g", "--groundtruth", type=str)
    parser.add_argument("-f", "--file-format", type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-r", "--rate", type=float, default=50)
    parser.add_argument("-l", "--length", type=int)
    parser.add_argument("-o", "--output", type=str)
    return parser
    

# 全局超参数
N_fft = 4096
zeroFreq_idx = 2049 - 1

img_dir = [
    "datasets/clean_water_10m/data",
    "datasets/clean_water_13m/data",
    "datasets/clean_water_15m/data",
    "datasets/clean_water_20m/data"
]
ground_truth = [
    "datasets/clean_water_10m/groundtruth.npy",
    "datasets/clean_water_13m/groundtruth.npy",
    "datasets/clean_water_15m/groundtruth.npy",
    "datasets/clean_water_20m/groundtruth.npy"
]
# tem_path = ["datasets/template.npy"]
tem_path = [
    "datasets/clean_water_10m/template.npy",
    "datasets/clean_water_13m/template.npy",
    "datasets/clean_water_15m/template.npy",
    "datasets/clean_water_13m/template.npy"
]
nums = [400, 349, 300, 267]
rates = [200, 1000, 500, 100]
outputs = [
    "StreakNet_outputs/traditional/0.png",
    "StreakNet_outputs/traditional/1.png",
    "StreakNet_outputs/traditional/2.png",
    "StreakNet_outputs/traditional/3.png"
]


if __name__ == "__main__":
    args = make_parse().parse_args()
    if args.img_dir is None:
        tp, fn, fp, tn = 0, 0, 0, 0
        for img_dir_i, gd_i, tem_i, rate_i, num_i, out_i in zip(img_dir, ground_truth, tem_path, rates, nums, outputs):
            print("processing {}...".format(img_dir_i))
            pred = main(template_path=tem_i, 
                        filename_format="{:03d}.tif",
                        data_dir=img_dir_i,
                        data_num=num_i,
                        rate=rate_i,
                        batch_size=args.batch_size)
            gd = np.load(gd_i)
            tp_i, fn_i, fp_i, tn_i = valid(pred, gd)
            tp, fn, fp, tn = tp + tp_i, fn + fn_i, fp + fp_i, tn + tn_i
            os.makedirs(os.path.dirname(out_i), exist_ok=True)
            cv2.imwrite(out_i, pred)
        acc, precision, recall, f1, psnr = cal_rates(tp, fn, fp, tn)
        print("[Total]Accuracy:{:.03f}% Precision:{:.03f}% Recall:{:.03f}% F1-Score:{:.03f}% PSNR:{:.03f}dB".format(
            acc * 100, precision * 100, recall * 100, f1 * 100, psnr
        ))
    else:
        pred = main(template_path=args.template_path, 
                    filename_format=args.file_format,
                    data_dir=args.img_dir,
                    data_num=args.length,
                    rate=args.rate,
                    batch_size=args.batch_size)
        gd = np.load(args.groundtruth)
        valid(pred, gd)
        if args.output is not None:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            cv2.imwrite(args.output, pred)
    