#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os
import time
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
    parser.add_argument("-f", "--exp_file", default=None, type=str, required=True, help="please input your experiment description file")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="Select device.")
    parser.add_argument("-n", "--num", type=int, default=1030)
    parser.add_argument("-w", "--warmup", type=int, default=6)
    parser.add_argument("--save", default=False, action="store_true")
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


def reconstruction_algorithm_benchmark(num, filter, model, device=torch.device('cpu')):
    from streaknet.data import StreakTransform
    transform = StreakTransform(True)
    
    gray_img = torch.zeros((2048, num), dtype=torch.float32).to(device)
    deep_img = torch.zeros((2048, num), dtype=torch.float32).to(device)
    mask_img = torch.zeros((2048, num), dtype=torch.uint8).to(device)
    
    template = np.random.rand(512)
    img_src = torch.rand((1, 2048, 2048))
    template = torch.tensor(template, dtype=torch.float32).to(device)[None, None, :].repeat(1, 1, 1)
    template_std = transform(template)
    
    template_freq = torch.fft.rfft(template, 65536, dim=2).repeat(1, 1, 1)
    template_freq_std = torch.fft.rfft(template_std, 65536, dim=2)[:, :, :4000].repeat(1 * 2048, 1, 1)
    
    t_list = []
    t_list.append(time.time())

    for idx in tqdm(range(num)):
        img = img_src.to(device)
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
        output = output.view((1, 2048))
        mask_img[:, idx:(idx+1)] = output.T
        
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
        gray_img[:, idx:(idx+1)] = max_resp.T
        
        # 计算景深
        distance = (max_resp_index * 30 * 1e-9 / 2 /2048) * 3e8 
        deep_img[:, idx:(idx+1)] = distance.T
        
        t_list.append(time.time())
    
    return t_list
    

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
    
    t_list = reconstruction_algorithm_benchmark(args.num, filter, model, device)
    t_list = t_list[args.warmup:]
    
    ait = []
    it = 0
    for i in range(1, len(t_list)):
        t = (t_list[i] - t_list[i-1]) * 1000
        it += t
        ait.append(it / i)
    
    print(ait)
    
    if args.save:
        file_name = os.path.join(exp.output_dir, "benchmark")
        os.makedirs(file_name, exist_ok=True)
        df = pd.DataFrame({
            "ait": ait
        })
        df.to_excel(os.path.join(file_name, "benchmark_{}.xlsx".format(args.experiment_name)), engine='openpyxl')
    

if __name__ == "__main__":
    args = make_parse().parse_args()
    main(args)
    