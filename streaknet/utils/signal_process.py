#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import torch 
import numpy as np

from typing import Union


def band_pass_filter(
        signal: Union[torch.Tensor, np.array], 
        lower: int=0, upper: int=-1
    ):
    processed_signal = signal[:]
    if isinstance(signal, torch.Tensor):
        ndim = signal.dim()
    else:
        ndim = signal.ndim()
    if ndim == 1:
        processed_signal[:lower] = 0
        processed_signal[upper:] = 0
    elif ndim == 2:
        processed_signal[:, :lower] = 0
        processed_signal[:, upper:] = 0
    elif ndim == 3:
        processed_signal[:, :, :lower] = 0
        processed_signal[:, :, upper:] = 0
    else:
        raise TypeError("The ndim of torch.tensor or numpy.array must be 1, 2, or 3.")
    return processed_signal


def standard(signal):
    signal_min = signal.min()
    signal_max = signal.max()
    return (signal - signal_min) / (signal_max - signal_min)
