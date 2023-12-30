#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os 
import yaml 
import argparse
import numpy as np
from scipy.signal import convolve
from matplotlib import pyplot as plt
from icecream import ic


def make_parser():
    parser = argparse.ArgumentParser("Visualize StreakData")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="datasets"
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=int,
        default=20
    )
    return parser


def read_config(path):
    with open(path, "r") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data 


def main(args):
    config = read_config(os.path.join(args.path, "test_config.yaml"))
    sub_datasets = config["sub_datasets"]
    num = len(sub_datasets)
    
    h = [1] * args.kernel
    h = np.array(h, dtype=np.float) / args.kernel
    
    for i, name in enumerate(sub_datasets):
        path = os.path.join(args.path, name, "template.npy")
        signal = np.load(path)
        processed_signal = convolve(signal, h, mode='valid')
        plt.subplot(2, num, i + 1)
        plt.title("signal-{}".format(name))
        plt.plot(signal)
        plt.subplot(2, num, i + 1 + num)
        plt.title("smooth-{}".format(name))
        plt.plot(processed_signal)
    
    plt.show()


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
    