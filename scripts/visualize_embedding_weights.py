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
from streaknet.utils import get_model_info, get_embedding_filter
from streaknet.data import DataPrefetcher
from streaknet.data import cal_valid_results, cal_rates

MODEL_EXT = [".pth"]


def make_parser():
    parser = argparse.ArgumentParser("Visualize Embedding Weights")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="whether to save the inference result of streak images",
    )
    
    parser.add_argument(
        "-c",
        "--ckpt",
        default=None,
        type=str
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


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    assert os.path.isdir(file_name)

    logger.info("Args: {}".format(args))
    model = exp.get_model()
    
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
    
    h = [1] * 15
    h = np.array(h, dtype=np.float) / 15
    
    # 满屏扫描时间30ns，CCD分辨率2048，采样频率68.27GHz
    # 使用长度65536计算FFT，频率分辨率68.27GHz/65536=1.04MHz
    freq_resolutio = (2048 / 30) * 1000 / 65536
    x_f = [(i * freq_resolutio) for i in range(4000)]
    x_f = np.array(x_f)
    embedding_filter = get_embedding_filter(model).cpu().numpy()
    smooth_embedding_filter = convolve(embedding_filter, h, 'same')
    max_attention = np.argmax(smooth_embedding_filter) * freq_resolutio
    
    plt.figure(figsize=(14, 8))
    plt.suptitle("{}".format(args.experiment_name))
    plt.subplot(2, 1, 1)
    plt.plot(x_f, embedding_filter, label="embedding-filter", c='gray')
    plt.plot(x_f, smooth_embedding_filter, label="smooth-embedding-filter", c='blue')
    plt.vlines(500, 0, 1, color='red', linestyle="--")
    plt.hlines(0.2, -0.2, 3999*freq_resolutio, color='red', linestyles='--')
    plt.text(500, -0.2,  "{:.2f}MHz".format(500), color='red', ha='center')
    plt.xlabel("Mhz")
    plt.ylabel("Attention")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(x_f, embedding_filter, label="embedding-filter", c='gray')
    plt.plot(x_f, smooth_embedding_filter, label="smooth-embedding-filter", c='blue')
    plt.xlim([0, 60])
    plt.vlines(4.04, 0, 1, color='red', linestyle="--")
    plt.vlines(max_attention, 0, 1, color='red', linestyle="--")
    plt.hlines(0.2, -0.2, 3999*freq_resolutio, color='red', linestyles='--')
    plt.text(4.04, -0.2,  "{:.2f}MHz".format(4.04), color='red', ha='center')
    plt.text(max_attention, -0.2,  "{:.2f}MHz".format(max_attention), color='red', ha='center')
    plt.legend()
    plt.xlabel("Mhz")
    plt.ylabel("Attention")
    
    if args.save:
        path = os.path.join(file_name, "visualize_embedding.png")
        plt.savefig(path)
        os.makedirs(os.path.join(file_name, "npy"), exist_ok=True)
        np.save(os.path.join(file_name, "npy", "embedding_filter.npy"), embedding_filter)
        np.save(os.path.join(file_name, "npy", "smooth_embedding_filter.npy"), smooth_embedding_filter)
    else:
        plt.show()
        
        
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    main(exp, args)
    