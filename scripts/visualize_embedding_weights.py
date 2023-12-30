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

import torch

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
        "-t",
        "--template",
        default="datasets/template.npy",
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
    
    import matplotlib
    from matplotlib import pyplot as plt 
    
    template = np.load(args.template)
    template_fft = np.fft.fft(template, 2304)
    template_fft = np.absolute(template_fft)
    
    plt.subplot(4, 1, 1)
    plt.plot(template_fft)
    plt.yscale("log")

    signal_weight = model.embedding.signal_embedding_block.dense.state_dict()['weight']
    template_weight = model.embedding.template_embedding_block.dense.state_dict()['weight']
    
    # template_weight = model.embedding.signal_embedding_block.dense.state_dict()['weight']
    signal_weight = torch.abs(signal_weight)
    template_weight = torch.abs(template_weight)
    signal_cnt = torch.sum(signal_weight, dim=0)
    template_cnt = torch.sum(template_weight, dim=0)
    signal_cnt_real = signal_cnt[:signal_cnt.shape[0]//2]
    signal_cnt_imag = signal_cnt[signal_cnt.shape[0]//2:]
    template_cnt_real = template_cnt[:template_cnt.shape[0]//2]
    template_cnt_imag = template_cnt[template_cnt.shape[0]//2:]
    signal_attention = torch.sqrt(signal_cnt_real ** 2 + signal_cnt_imag ** 2)
    template_attention = torch.sqrt(template_cnt_real ** 2 + template_cnt_imag ** 2)
    signal_attention = torch.nn.functional.softmax(signal_attention)
    template_attention = torch.nn.functional.softmax(template_attention)
    
    # h = torch.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]], dtype=torch.float32) / 20
    # attention = torch.nn.functional.conv1d(attention.unsqueeze(0), h, None)[0]
    
    plt.subplot(4, 1, 2)
    plt.plot(signal_attention.cpu().numpy())
    plt.yscale("log")
    
    plt.subplot(4, 1, 3)
    plt.plot(template_attention.cpu().numpy())
    plt.yscale("log")
    
    plt.subplot(4, 1, 4)
    plt.plot(template_attention.cpu().numpy() * template_fft)
    plt.yscale("log")
    
    plt.show()
        
        
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)

    main(exp, args)
    