#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import torch 
from torch import nn

from .network_blocks import get_activation


class FrequencyDomainFilteringBlock(nn.Module):
    def __init__(self, width=1.0, dropout=0.4, act='silu', export=False):
        super(FrequencyDomainFilteringBlock, self).__init__()
        self.export = export
        self.n_fft = 4096
        self.embedding_size = round(1024 * width)
        self.flatten = nn.Flatten(1)
        self.dense = nn.Linear(self.n_fft * 2, self.embedding_size)
        self.norm = nn.LayerNorm((self.embedding_size,))
        self.dropout = nn.Dropout(dropout)
        self.act = get_activation(act, inplace=False)
    
    def forward(self, x):
        """ export mode:
                x.shape:[batch, 2, self.n_fft]
            else:
                x.shape:[batch, len]
        """
        if not self.export:
            signal_freq = torch.fft.fft(x, self.n_fft, dim=1)
            signal_real = torch.real(signal_freq)
            signal_imag = torch.imag(signal_freq)
            concat_signal = torch.concat([signal_real, signal_imag], dim=1)
        else:
            concat_signal = x
        concat_signal = self.flatten(concat_signal).unsqueeze(1)
        pred = self.norm(self.dense(concat_signal))
        pred = self.dropout(pred)
        pred = self.act(pred)
        return pred


class FrequencyDomainFilteringEmbedding(nn.Module):
    def __init__(self, width=1.0, dropout=0.4, act='silu', export=False):
        super(FrequencyDomainFilteringEmbedding, self).__init__()
        self.signal_embedding_block = FrequencyDomainFilteringBlock(width, dropout, act, export)
        self.template_embedding_block = FrequencyDomainFilteringBlock(width, dropout, act, export)
    
    def forward(self, signal, template):
        signal_embedding = self.signal_embedding_block(signal)
        template_embedding = self.template_embedding_block(template)
        ret = torch.concat([signal_embedding, template_embedding], dim=1)
        return ret
        