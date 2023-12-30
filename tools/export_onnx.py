#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import argparse
import os
from loguru import logger

import torch
from torch import nn

from streaknet.exp import get_exp 
from streaknet.utils import replace_module
from streaknet.models.network_blocks import SiLU


def make_parser():
    parser = argparse.ArgumentParser("StreakNet onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="streaknet.onnx", help="output name of models"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model(export=True)
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
    model = replace_module(model, nn.SiLU, SiLU)

    logger.info("loading checkpoint done.")
    signal_input = torch.randn(args.batch_size, 2, 2048 + round(2048 * exp.width))
    template_input = torch.randn(args.batch_size, 2, 2048 + round(2048 * exp.width))

    torch.onnx.export(
        model,
        (signal_input, template_input),
        args.output_name,
        input_names=["radar_fft", "template_fft"],
        output_names=["output"],
        dynamic_axes={"radar_fft": {0: 'batch'},
                      "template_fft": {0: 'batch'},
                      "output": {0: 'batch'}},
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify

        # use onnx-simplifier to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    main()