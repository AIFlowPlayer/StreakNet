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


def hilbert(x):
    N = x.shape[2]
    if N % 2 == 0:
        N += 1
        
    freq = torch.fft.fft(x, N)
    freq_real = torch.real(freq)
    freq_imag = torch.imag(freq)
    
    hilbert_freq_real = torch.zeros_like(freq_real)
    hilbert_freq_imag = torch.zeros_like(freq_imag)
    
    hilbert_freq_real[:, :, 1:N//2+1] = -freq_imag[:, :, 1:N//2+1]
    hilbert_freq_real[:, :, N//2+1:] = freq_imag[:, :, N//2+1:]
    hilbert_freq_real[:, :, 0] = 0
    
    hilbert_freq_imag[:, :, 1:N//2+1] = freq_real[:, :, 1:N//2+1]
    hilbert_freq_imag[:, :, N//2+1:] = -freq_real[:, :, N//2+1:]
    hilbert_freq_imag[:, :, 0] = 0
    
    hilbert_freq = torch.complex(hilbert_freq_real, hilbert_freq_imag)
    hilbert_sig = torch.fft.ifft(hilbert_freq)[:, :, :x.shape[2]]
    
    analysis = torch.abs(torch.sqrt(hilbert_sig * hilbert_sig + x * x))
    return analysis
