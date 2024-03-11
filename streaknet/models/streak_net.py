#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import torch
from torch import nn 

from .streak_embedding import FDEmbedding
from .streak_backbone import SelfAttention, DBCAttention
from .streak_head import ImagingHead


class StreakNetArch(nn.Module):
    def __init__(self, embedding=None, backbone=None, head=None):
        super(StreakNetArch, self).__init__()
        if embedding is None:
            embedding = FDEmbedding()
        if backbone is None:
            backbone = SelfAttention()
        if head is None:
            backbone = ImagingHead()
        
        self.embedding = embedding
        self.backbone = backbone
        self.head = head
    
    def forward(self, signal, template, targets=None):
        embedding = self.embedding(signal, template)
        outs = self.backbone(embedding)
        
        if self.training:
            assert targets is not None
            outputs = self.head(outs, labels=targets)
        else:
            outputs = self.head(outs)
        
        return outputs
