#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os
import time
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm

import torch 

from streaknet.utils import hilbert


def make_parse():
    parser = argparse.ArgumentParser("Band Pass Filter Algorithm.")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-l", "--lower", type=float, default=450, help="Lower bound of Band Pass Filter, MHz")
    parser.add_argument("-u", "--upper", type=float, default=550, help="Upper bound of Band Pass Filter, MHz")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="Select device.")
    parser.add_argument("-n", "--num", type=int, default=1030)
    parser.add_argument("-w", "--warmup", type=int, default=6)
    parser.add_argument("--save", default=False, action="store_true")
    return parser


def get_filter(args, max_len=4000):
    freq_resolutio = (2048 / 30) * 1000 / 65536
    lower = round(args.lower / freq_resolutio)
    upper = round(args.upper / freq_resolutio)
    filter = torch.ones((1, 1, max_len), dtype=torch.float32)
    filter[:, :, :lower] = 0
    filter[:, :, upper:] = 0
    file_name = os.path.join("StreakNet_outputs", "benchmark")
    return filter, file_name


def band_pass_filter_algorithm_benchmark(num, filter, device=torch.device('cpu')):
    gray_img = torch.zeros((2048, num), dtype=torch.float32).to(device)
    deep_img = torch.zeros((2048, num), dtype=torch.float32).to(device)
    
    template = np.random.rand(512)
    img_src = torch.rand((1, 2048, 2048))
    template = torch.tensor(template, dtype=torch.float32).to(device)[None, None, :].repeat(1, 1, 1)
    template_freq = torch.fft.rfft(template, 65536, dim=2)
    
    t_list = []
    t_list.append(time.time())

    for idx in tqdm(range(num)):
        img = img_src.to(device)
        
        img_freq = torch.fft.rfft(img, 65536, dim=2) # 只取正频率部分
        tem_freq = template_freq[:img_freq.shape[0]]
        
        # 带通滤波 & 匹配滤波
        img_freq[:, :, 4000:] = 0
        img_freq[:, :, :4000] *= filter
        match_filt = torch.absolute(torch.fft.irfft(tem_freq * img_freq, dim=2)[:, :, :2048])
        match_filt = hilbert(match_filt)
        
        # 确定最大响应及其时间
        max_resp, max_resp_index = torch.max(match_filt, dim=2)
        gray_img[:, idx:(idx+1)] = max_resp.T
        
        # 计算景深
        distance = (max_resp_index * 30 * 1e-9 / 2 /2048) * 3e8 
        deep_img[:, idx:(idx+1)] = distance.T
        
        t_list.append(time.time())
    
    return t_list
    

def main(args):
    device = torch.device(args.device)
    filter, file_name = get_filter(args, 4000)
    filter = filter.to(device)
    
    t_list = band_pass_filter_algorithm_benchmark(args.num, filter, device)
    t_list = t_list[args.warmup:]
    
    ait = []
    avg = []
    it = 0
    for i in range(1, len(t_list)):
        t = (t_list[i] - t_list[i-1]) * 1000
        avg.append(t)
        it += i * t 
        ait.append(it / i)
    
    print(np.mean(avg))
    
    if args.save:
        os.makedirs(file_name, exist_ok=True)
        df = pd.DataFrame({
            "ait": ait
        })
        df.to_excel(os.path.join(file_name, "benchmark_bandpass.xlsx"), engine='openpyxl')
    

if __name__ == "__main__":
    args = make_parse().parse_args()
    main(args)
    