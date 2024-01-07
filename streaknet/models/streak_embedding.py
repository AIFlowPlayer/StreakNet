#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import torch 
from torch import nn

from .network_blocks import get_activation


class FrequencyDomainFilteringBlock(nn.Module):
    def __init__(self, width=1.0, act='silu', export=False):
        super(FrequencyDomainFilteringBlock, self).__init__()
        self.export = export
        self.embedding_size = round(512 * width)
        self.flatten = nn.Flatten(1)
        self.norm = nn.LayerNorm((4000 * 2,))
        self.dense = nn.Linear(4000 * 2, self.embedding_size)
        self.act = get_activation(act, inplace=False)
    
    def forward(self, x):
        """ export mode:
                x.shape:[batch, 2, 4000]
            else:
                x.shape:[batch, len]
        """
        if not self.export:
            # 满屏扫描时间30ns，CCD分辨率2048，采样频率68.27GHz
            # 使用长度65536计算FFT，频率分辨率68.27GHz/65536=1.04MHz
            signal_freq = torch.fft.rfft(x, 65536, dim=1)
            signal_freq = signal_freq[:, :4000] # 只要0~4GHz
            signal_real = torch.real(signal_freq)
            signal_imag = torch.imag(signal_freq)
            concat_signal = torch.concat([signal_real, signal_imag], dim=1)
        else:
            concat_signal = x
        concat_signal = self.flatten(concat_signal).unsqueeze(1)
        pred = self.dense(self.norm(concat_signal))
        pred = self.act(pred)
        return pred


class FrequencyDomainFilteringEmbedding(nn.Module):
    def __init__(self, width=1.0, act='silu', export=False):
        super(FrequencyDomainFilteringEmbedding, self).__init__()
        self.signal_embedding_block = FrequencyDomainFilteringBlock(width, act, export)
        self.template_embedding_block = FrequencyDomainFilteringBlock(width,act, export)
    
    def forward(self, signal, template):
        signal_embedding = self.signal_embedding_block(signal)
        template_embedding = self.template_embedding_block(template)
        ret = torch.concat([signal_embedding, template_embedding], dim=1)
        return ret


class DoubleBranchEmbeddingBlock(nn.Module):
    def __init__(self, width=1.0, act='silu', export=False):
        super(DoubleBranchEmbeddingBlock, self).__init__()
        self.export = export
        self.embedding_size = round(512 * width)
        self.norm = nn.LayerNorm((4000,))
        self.real_dense = nn.Linear(4000, self.embedding_size)
        self.imag_dense = nn.Linear(4000, self.embedding_size)
        self.act = get_activation(act, inplace=False)
    
    def forward(self, x):
        """ export mode:
                x.shape:[batch, 2, 4000]
            else:
                x.shape:[batch, len]
        """
        if not self.export:
            # 满屏扫描时间30ns，CCD分辨率2048，采样频率68.27GHz
            # 使用长度65536计算FFT，频率分辨率68.27GHz/65536=1.04MHz
            signal_freq = torch.fft.rfft(x, 65536, dim=1)
            signal_freq = signal_freq[:, :4000] # 只要0~4GHz
            signal_real = torch.real(signal_freq)
            signal_imag = torch.imag(signal_freq)
        else:
            signal_real = x[:, 0, :]
            signal_imag = x[:, 1, :]
        signal_real = signal_real.unsqueeze(1)
        signal_imag = signal_imag.unsqueeze(1)
        pred_real = self.real_dense(self.norm(signal_real))
        pred_imag = self.imag_dense(self.norm(signal_imag))
        pred = torch.concat([pred_real, pred_imag], dim=1)
        return pred


class DoubleBranchFrequencyDomainEmbedding(nn.Module):
    def __init__(self, width=1.0, act='silu', concat=False, export=False):
        super(DoubleBranchFrequencyDomainEmbedding, self).__init__()
        self.concat = concat
        self.signal_embedding_block = DoubleBranchEmbeddingBlock(width, act, export)
        self.template_embedding_block = DoubleBranchEmbeddingBlock(width,act, export)
    
    def forward(self, signal, template):
        signal_embedding = self.signal_embedding_block(signal)
        template_embedding = self.template_embedding_block(template)
        if self.concat:
            ret = torch.concat([signal_embedding, template_embedding], dim=1)
            return ret 
        else:
            return signal_embedding, template_embedding
        