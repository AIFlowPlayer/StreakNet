#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import argparse
import os
import numpy as np
from loguru import logger
from tqdm import tqdm

import cv2
from scipy.signal import convolve

import torch
from matplotlib import pyplot as plt

from streaknet.exp import get_exp
from streaknet.utils import get_model_info
from streaknet.data import DataPrefetcher
from streaknet.data import cal_valid_results, cal_rates

MODEL_EXT = [".pth"]


def make_parser():
    parser = argparse.ArgumentParser("Visualize Embedding Weights")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of streak images",
    )
    
    parser.add_argument(
        "-c",
        "--ckpt",
        default=None,
        type=str
    )
    
    parser.add_argument(
        "--path",
        type=str, 
        default="datasets/clean_water_10m"
    )
    
    parser.add_argument(
        "--row",
        type=int,
        default=0
    )
    parser.add_argument(
        "--col",
        type=int,
        default=0
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )

    return parser


def get_filter(weight):
    weight = torch.abs(weight)
    weight_sum = torch.sum(weight, dim=0)
    weight_sum_real = weight_sum[:weight_sum.shape[0]//2]
    weight_sum_imag = weight_sum[weight_sum.shape[0]//2:]
    filter = torch.complex(weight_sum_real, weight_sum_imag)
    filter = torch.fft.fftshift(filter)
    return filter


def standard(data):
    data_min = data.min()
    data_max = data.max()
    return (data - data_min) / (data_max - data_min)


def band_pass_filter(template_freq: torch.tensor):
    if template_freq.shape[0] == 4096:
        template_freq[2048-6:2048+6+1] = 0
        template_freq[:2000] = 0
        template_freq[2096:] = 0
    elif template_freq.shape[0] == 8192:
        template_freq[4096-12:4092+12+1] = 0
        template_freq[:4000] = 0
        template_freq[4192:] = 0
    return template_freq


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    assert os.path.isdir(file_name)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model)))
    
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt
    
    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    
    x_t = [i*30/2048 for i in range(2048)]
    x_t = np.array(x_t)
    
    x_f = [(i-2048)*50/3 for i in range(4096)]
    x_f2 = [(i-4096)*25/3 for i in range(8192)]
    x_f = np.array(x_f)
    x_f2 = np.array(x_f2)

    signal_radar_weight = model.embedding.signal_embedding_block.dense.state_dict()['weight']
    # signal_template_weight = model.embedding.template_embedding_block.dense.state_dict()['weight']
    signal_radar_filter = get_filter(signal_radar_weight)
    # signal_template_filter = get_filter(signal_template_weight)
    signal_radar_filter = standard(torch.absolute(signal_radar_filter)).cpu().numpy()
    # signal_template_filter = standard(torch.absolute(signal_template_filter)).cpu().numpy()
    max_attention = np.argmax(signal_radar_filter[2048:]) * 50 / 3
    
    traditional_filter = torch.ones((8192,), dtype=torch.float)
    traditional_filter = band_pass_filter(traditional_filter)
    
    img = cv2.imread(os.path.join(args.path, "data", "{:03d}.tif".format(args.col)), -1)
    target_signal = img[args.row, :]
    target_signal_freq = np.fft.fftshift(np.fft.fft(target_signal, 4096))
    
    h = [1] * 15
    h = np.array(h, dtype=np.float) / 15
    smooth_target_signal = convolve(target_signal, h, 'same')
    
    # template = np.load(args.template)
    # template_freq = np.fft.fftshift(np.fft.fft(template, 4096))
    # template_freq = np.absolute(template_freq)
    
    fig, _ = plt.subplots(figsize=(14,8))
    fig.suptitle(args.experiment_name)
    plt.subplots_adjust(hspace=0.25)
    ax11 = plt.subplot(2, 1, 1)
    ax11.set_title("Time Domain")
    ax11.plot(x_t, target_signal, color='gray', label="received signal")
    ax11.plot(x_t, smooth_target_signal, color='r', label="smoothed")
    ax11.set_xlabel("Time(ns)")
    ax11.set_xlim([3, 27])
    ax11.set_ylim([90, np.max(smooth_target_signal)])
    ax11.legend()
    
    ax31 = plt.subplot(2, 1, 2)
    ax31.set_title("Frequency Domain")
    ax32 = ax31.twinx()
    ax31.plot(x_f, target_signal_freq + np.abs(np.min(target_signal_freq)) + 1e-6, color='b', label="received signal")
    ax32.plot(x_f2, traditional_filter, color='r', label="traditional band filter")
    ax32.plot(x_f, signal_radar_filter, color='g', label="deep learning filter")
    ax32.axvline(x=max_attention, color='gray', linestyle="--")
    ax32.text(max_attention, -0.2,  "{:.2f}".format(max_attention), color='gray', ha='center')
    ax32.axvline(x=500, color='gray', linestyle="--")
    # ax32.text(500, -0.1,  "{:.2f}".format(500), color='gray', ha='center')
    ax31.set_xlabel("Frequency(MHz)")
    ax31.set_ylabel("Amplitude")
    ax31.set_yscale("log")
    ax32.set_ylabel("Frequency Selectivity")
    ax31.set_xlim([-20, 2000])
    # ax31.set_ylim([-15000, 250000])
    ax32.set_ylim([-0.05, 1.3])
    ax31.legend(loc="upper left")
    ax32.legend(loc="upper right")
    
    if args.save_result:
        path = os.path.join(vis_folder, "visualize_embedding.png")
        plt.savefig(path)
    else:
        plt.show()
        
        
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)

    main(exp, args)
    