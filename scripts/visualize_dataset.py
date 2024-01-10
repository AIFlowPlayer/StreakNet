#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os 
import cv2
import yaml 
import argparse
import numpy as np
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
    return parser


def read_config(path):
    with open(path, "r") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data 


def main(args):
    configs = []
    configs.append(["train", read_config(os.path.join(args.path, "train_config.yaml"))])
    configs.append(["valid", read_config(os.path.join(args.path, "valid_config.yaml"))])
    configs.append(["test", read_config(os.path.join(args.path, "test_config.yaml"))])
    lens = [len(config["sub_datasets"]) for _, config in configs]
    max_len = np.max(lens)
    plt.figure(figsize=(18, 8))
    plt.suptitle("Visualize Dataset")
    for j, (name, config) in enumerate(configs):
        sub_datasets = []
        area = [0 for _ in range(max_len)]
        for name_i in config["sub_datasets"]:
            gd = np.load(os.path.join(args.path, name_i, "groundtruth.npy")) * 255
            gd = cv2.cvtColor(gd, cv2.COLOR_GRAY2RGB)
            sub_datasets.append(gd)
        for i, info in enumerate(config["config"]):
            idx = info["sub_idx"]
            col = info["col_slice"]
            row = info["row_slice"]
            w = col[1] - col[0]
            h = row[1] - row[0]
            area[idx] += w * h
            sub_datasets[idx][row[0]:row[1], col[0]:col[1]] = 0.5 * sub_datasets[idx][row[0]:row[1], col[0]:col[1]] + 0.5 * np.array([[[255, 0, 0]]])
        for i, name_i in enumerate(config["sub_datasets"]):
            plt.subplot(max_len, 3, 3 * i + j + 1)
            plt.title("{}-{}".format(name_i, name))
            im = cv2.UMat(sub_datasets[i].transpose(1, 0, 2))
            im = cv2.putText(im, "area:{}".format(area[i]), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(255, 255, 255), thickness=3)
            plt.imshow(cv2.UMat.get(im), cmap="gray")
    plt.show()


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
    