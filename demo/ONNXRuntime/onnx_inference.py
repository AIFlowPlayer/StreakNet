#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All righs reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import onnxruntime

IMAGE_EXT = [".tif"]


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="../../streaknet_s.onnx",
        help="Input your onnx model.",
    )
    
    parser.add_argument(
        "--path", type=str, default="../../datasets/clean_water_10m/data", help="path to streak images"
    )
    parser.add_argument(
        "--template", type=str, default="../../datasets/template.npy", help="path to template signal"
    )
    parser.add_argument(
        "-w",
        "--width",
        type=float,
        default=0.125
    )
    
    parser.add_argument(
        "-b",
        "--batch",
        default=256,
        type=int
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


def preproc(inp):
    inp_max = np.max(inp, axis=1, keepdims=True)
    inp_min = np.min(inp, axis=1, keepdims=True)
    out = (inp - inp_min) / (inp_max - inp_min)
    return out


def FFT(inp, width):
    n_fft = 2048 + round(2048 * width) # Fixed calculation method
    freq = np.fft.fft(inp, n=n_fft)
    freq_real = np.real(freq).astype(np.float32)
    freq_imag = np.imag(freq).astype(np.float32)
    out = np.stack((freq_real, freq_imag), axis=1)
    return out


def inference(session, batch_size, width, img, tem):
    tem = np.expand_dims(tem, 0).repeat(batch_size, axis=0)
    tem = preproc(tem)
    tem = FFT(tem, width)   # ONNX does not support FFT operators
    
    outputs = np.zeros((2048,), dtype=np.uint8)
    for i in range(2048 // batch_size):
        signal = img[i*batch_size:(i+1)*batch_size, :]
        signal = preproc(signal)
        signal = FFT(signal, width) # ONNX does not support FFT operators
        
        ort_inputs = {
            "radar_fft": signal,
            "template_fft": tem
        }
        output = session.run(None, ort_inputs)[0]
        outputs[i*batch_size:(i+1)*batch_size] = output * 255
    
    return outputs
        

def main(args):
    session = onnxruntime.InferenceSession(args.model)
    
    file_list = get_image_list(args.path)
    file_list.sort()
    
    result_img = np.zeros((2048, len(file_list)), dtype=np.uint8)
    
    plt.ion()
    plt.figure(1)
    
    for i, image_path in enumerate(tqdm(file_list)):
        img = cv2.imread(image_path, -1)
        tem = np.load(args.template)
        
        img = img.astype(np.float32)
        tem = tem.astype(np.float32)
        
        outputs = inference(session, args.batch, args.width, img, tem)
        result_img[:, i] = outputs 
        
        plt.clf()
        plt.imshow(result_img, cmap='gray')
        plt.colorbar()
        plt.draw()
        plt.pause(0.1)
    
    plt.ioff()
    plt.clf()
    plt.imshow(result_img, cmap='gray')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args)
    