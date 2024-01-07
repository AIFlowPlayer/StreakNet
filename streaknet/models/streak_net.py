#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import torch
from torch import nn 

from .streak_embedding import FrequencyDomainFilteringEmbedding, DoubleBranchFrequencyDomainEmbedding
from .streak_backbone import StreakTransformerEncoder, DoubleBranchCrossAttention
from .streak_head import ImagingHead


class StreakNet(nn.Module):
    def __init__(self, embedding=None, backbone=None, head=None):
        super(StreakNet, self).__init__()
        if embedding is None:
            embedding = FrequencyDomainFilteringEmbedding()
        if backbone is None:
            backbone = StreakTransformerEncoder()
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


class StreakNetV2(nn.Module):
    def __init__(self, embedding=None, backbone=None, head=None):
        super(StreakNetV2, self).__init__()
        if embedding is None:
            embedding = DoubleBranchFrequencyDomainEmbedding()
        if backbone is None:
            backbone = DoubleBranchCrossAttention()
        if head is None:
            backbone = ImagingHead()
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.embedding = embedding
        self.backbone = backbone
        self.head = head
    
    def forward(self, signal, template, targets=None):
        embedding = self.embedding(signal, template)
        if isinstance(embedding, tuple):
            outs = self.backbone(embedding[0], embedding[1])
            outs = torch.concat([outs[0], outs[1]], 1)
        else:
            outs = self.backbone(embedding)
            
        if self.training:
            assert targets is not None
            outputs = self.head(outs, labels=targets)
        else:
            outputs = self.head(outs)
        
        return outputs
