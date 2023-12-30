#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import argparse
import os
import time
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

import cv2

import torch

from streaknet.exp import get_exp
from streaknet.utils import get_model_info
from streaknet.data import StreakTransform

IMAGE_EXT = [".tif"]


def make_parser():
    parser = argparse.ArgumentParser("StreakNet Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

    parser.add_argument(
        "--path", type=str, default="./datasets/clean_water_10m/data", help="path to streak images"
    )
    parser.add_argument(
        "--template", type=str, default="./datasets/template.npy", help="path to template signal"
    )
    
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of streak images",
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
        default=256,
        type=int
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        device="cpu",
    ):
        self.model = model
        self.device = device
        self.preproc = StreakTransform(batch=True)

    def inference(self, img, tem, cur, tot, batch_size):
        img = cv2.imread(img, -1)
        tem = np.load(tem)

        img = img.astype(np.float32)
        tem = tem.astype(np.float32)
        
        img = torch.from_numpy(img)
        tem = torch.from_numpy(tem)
        
        if self.device == "gpu":
            img = img.cuda()
            tem = tem.cuda()
        
        tem = tem.unsqueeze(0).repeat(batch_size, 1)
        tem = self.preproc(tem)

        outputs = np.zeros((2048,), dtype=np.uint8)
        with torch.no_grad():
            t0 = time.time()
            for i in range(2048 // batch_size):
                signal = img[i*batch_size:(i+1)*batch_size, :]
                signal = self.preproc(signal)
                output = self.model(signal, tem)
                outputs[i*batch_size:(i+1)*batch_size] = (output * 255).cpu().numpy()
            logger.info("[{}/{}]Infer time: {:.4f}s".format(cur, tot, time.time() - t0))
        
        return outputs


def demo(predictor, vis_folder, path, tem_path, current_time, save_result, batch_size):
    assert os.path.isdir(path)
    files = get_image_list(path)
    files.sort()
    
    img = np.zeros((2048, len(files)), dtype=np.uint8)
    if save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        save_file_name = os.path.join(save_folder, "result.png")
    
    for i, image_name in enumerate(files):
        outputs = predictor.inference(image_name, tem_path, i + 1, len(files), batch_size)
        img[:, i] = outputs
    
    if save_result:
        cv2.imwrite(save_file_name, img)
    else:
        plt.imshow(img, cmap='gray')
        plt.show()


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model)))

    if args.device == "gpu":
        model.cuda()
    else:
        model.cpu()
    model.eval()

    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    predictor = Predictor(model, args.device)
    current_time = time.localtime()
    demo(predictor, vis_folder, args.path, args.template, current_time, args.save_result, args.batch)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)

    main(exp, args)