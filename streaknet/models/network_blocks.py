#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All righs reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import torch
from torch import nn

from .streak_loss import StreakLoss, CrossLoss


def get_activation(name="silu", inplace=False):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


def get_loss(name="streakloss"):
    if name == "streakloss":
        module = StreakLoss()
    elif name == "crossloss":
        module = CrossLoss()
    else:
        raise AttributeError("Unsupported loss type: {}".format(name))
    return module


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    