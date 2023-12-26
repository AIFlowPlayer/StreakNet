#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All righs reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import copy
from torch import nn 

from .network_blocks import get_activation


class StreakTransformerEncoder(nn.Module):
    def __init__(self, width=1.0, depth=1.0, dropout=0.4, act='silu'):
        super(StreakTransformerEncoder, self).__init__()
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=round(1024 * width), 
            nhead=round(16 * width), 
            dim_feedforward=round(2 * 1024 * width),
            dropout=dropout,
            batch_first=True,
            activation=get_activation(act, inplace=False)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=round(8 * depth)
        )
    
    def forward(self, x):
        pred = self.transformer_encoder(x)
        return pred
         