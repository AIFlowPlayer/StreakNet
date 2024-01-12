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
    parser = argparse.ArgumentParser("Visualize Warmup Iters")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of streak images",
    )
    
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true"
    )
    
    parser.add_argument(
        "-c",
        "--checkpoint",
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
    parser.add_argument(
        "-b",
        "--batch",
        default=512,
        type=int
    )

    return parser


def get_model_list(path):
    model_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            name = os.path.basename(apath)
            name, ext = os.path.splitext(name)
            if ext in MODEL_EXT and "visualize_iter_" in name:
                model_names.append((apath, name))
    return model_names


def visualize_model(model_i, model_nums, model_name, model_path, model, test_loader, vis_folder):
    logger.info("[{}/{}]start visualizing {}...".format(model_i + 1, model_nums, model_name))
    max_iter = len(test_loader)
        
    logger.info("loading checkpoint {}...".format(model_path))
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    
    vis_results = []
    save_dirs = []
    results_num = len(test_loader.dataset.col_list)
    valid_results = np.zeros((results_num, 4), dtype=np.int32)
    for i, col_num in enumerate(test_loader.dataset.col_list):
        vis_results.append(np.zeros((2048, col_num), dtype=np.uint8))
        if args.save_result:
            save_folder = os.path.join(vis_folder, f"{i}")
            os.makedirs(save_folder, exist_ok=True)
            save_dirs.append(save_folder)
    
    prefetcher = DataPrefetcher(test_loader)
    for iter_i in tqdm(range(max_iter)):
        sig, tem, gd, info = prefetcher.next()
        if sig is None:
            break 
        gd = gd.reshape(-1)
        outputs = model(sig, tem)
        for i in range(results_num):
            cnt = np.sum((info[:, 0] == i).cpu().numpy())
            if cnt < 1: continue
            preds = outputs[info[:, 0] == i]
            labels = gd[info[:, 0] == i]
            tp, fn, fp, tn = cal_valid_results(preds, labels, origin=True)
            preds = preds.cpu().numpy()
            valid_results[i] += np.array([tp, fn, fp, tn], dtype=np.int32)
            vis_results[i][info[info[:, 0] == i][:, 1], info[info[:, 0] == i][:, 2]] = preds * 255
    
    for i in range(results_num):
        acc, precision, recall, f1, psnr = cal_rates(
            valid_results[i][0], valid_results[i][1], 
            valid_results[i][2], valid_results[i][3]
        )
        if args.save_result:
            cv2.imwrite(os.path.join(save_dirs[i], "{}.png".format(model_name)), vis_results[i])
            np.save(os.path.join(save_dirs[i], "{}.npy".format(model_name)), 
                    np.array([acc, precision, recall, f1, psnr], dtype=np.float32))
        else:
            from matplotlib import pyplot as plt 
            plt.imshow(vis_results[i])
            plt.show()


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

    model.cuda()
    model.eval()
    
    test_loader = exp.get_eval_loader(args.batch, False, args.cache, True)
    
    if args.checkpoint is None:
        model_names = get_model_list(file_name)
        model_nums = len(model_names)
        for model_i, (model_path, model_name) in enumerate(model_names):
            visualize_model(model_i, model_nums, model_name, model_path, model, test_loader, vis_folder)
    else:
        model_path = args.checkpoint
        assert os.path.isfile(model_path)
        assert os.path.exists(model_path)
        model_name = os.path.basename(args.checkpoint)
        model_name, _ = os.path.splitext(model_name)
        visualize_model(0, 1, model_name, model_path, model, test_loader, vis_folder)
        
        
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)

    main(exp, args)
    