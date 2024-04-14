#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import copy
import torch
from torch import nn 

from .network_blocks import get_activation, AddNorm


class SelfAttention(nn.Module):
    def __init__(self, width=1.0, depth=1.0, dropout=0.4, act='silu'):
        super(SelfAttention, self).__init__()
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=round(512 * width), 
            nhead=round(16 * width), 
            dim_feedforward=round(2 * 512 * width),
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


class DBCAttentionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: nn.Module):
        super(DBCAttentionLayer, self).__init__()

        self.signal_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.template_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.signal_addnorm1 = AddNorm(normalized_shape=(d_model,), dropout=dropout)
        self.signal_feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.signal_addnorm2 = AddNorm(normalized_shape=(d_model,), dropout=dropout)
        self.signal_act = activation
        
        self.template_addnorm1 = AddNorm(normalized_shape=(d_model,), dropout=dropout)
        self.template_feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.template_addnorm2 = AddNorm(normalized_shape=(d_model,), dropout=dropout)
        self.template_act = activation
    
    def forward(self, signal, template):
        signal_aten, _ = self.signal_attention(signal, template, template, need_weights=False)
        template_aten, _ = self.template_attention(template, signal, signal, need_weights=False)
        
        signal_aten_norm = self.signal_addnorm1(signal, signal_aten)
        signal_feedforward = self.signal_feedforward(signal_aten_norm)
        signal_feed_norm = self.signal_addnorm2(signal_feedforward, signal_aten_norm)
        signal_output = self.signal_act(signal_feed_norm)
        
        template_aten_norm = self.template_addnorm1(template, template_aten)
        template_feedforward = self.template_feedforward(template_aten_norm)
        template_feed_norm = self.template_addnorm2(template_feedforward, template_aten_norm)
        template_output = self.template_act(template_feed_norm)
        
        return signal_output, template_output


class DBCAttention(nn.Module):
    def __init__(self, width: float=1.00, depth: float=1.00, dropout: float=0.4, act: str='silu'):
        super(DBCAttention, self).__init__()
        
        attention_layer = DBCAttentionLayer(
            d_model=round(512 * width),
            nhead=round(16 * width),
            dim_feedforward=round(512 * 2 * width),
            dropout=dropout,
            activation=get_activation(act, False)
        )
        self.layers = _get_clones(attention_layer, N=round(8 * depth))
    
    def forward(self, x):
        out_sig = x[:, :1, :]
        out_tem = x[:, 1:, :]
        for layer in self.layers:
            out_sig, out_tem = layer(out_sig, out_tem)
        pred = torch.concat([out_sig, out_tem], dim=1)
        return pred
        

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)]) 
