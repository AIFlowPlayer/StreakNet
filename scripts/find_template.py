import os 
import cv2
import yaml 
from scipy.signal import convolve
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
        default="datasets/clean_water_10m"
    )
    parser.add_argument(
        "-r",
        "--row",
        type=int,
        default=0
    )
    parser.add_argument(
        "-c",
        "--col",
        type=int,
        default=0
    )
    parser.add_argument(
        "-l",
        "--low",
        type=int,
        default=0
    )
    parser.add_argument(
        "-u",
        "--up",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=int,
        default=20
    )
    return parser


def main(args):
    img = cv2.imread(os.path.join(args.path, "data", "{:03d}.tif".format(args.col)), -1)
    signal = img[args.row, :]
    
    plt.subplot(2, 2, 1)
    plt.plot(signal)
    plt.title("signal")
    
    slice_signal = signal[args.low:args.up]
    plt.subplot(2, 2, 2)
    plt.plot(slice_signal)
    plt.title("slice")
    
    if args.save:
        path = os.path.join(args.path, "template.npy")
        np.save(path, slice_signal)
    
    h = [1] * args.kernel
    h = np.array(h, dtype=np.float) / args.kernel
    processed_signal = convolve(slice_signal, h, mode='valid')
    print(processed_signal.shape)
    plt.subplot(2, 2, 3)
    plt.plot(processed_signal)
    plt.title("smooth")
    
    signal_min = np.min(processed_signal)
    signal_max = np.max(processed_signal)
    processed_signal = (processed_signal - signal_min) / (signal_max - signal_min)
    
    freq = np.fft.fft(processed_signal, 4096)
    freq = np.fft.fftshift(freq)
    freq = np.absolute(freq)
    plt.subplot(2, 2, 4)
    plt.plot(freq)
    plt.yscale("log")
    plt.title("fft")
    
    plt.show()


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
    