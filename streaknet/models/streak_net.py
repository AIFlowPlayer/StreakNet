#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

from torch import nn 

from .streak_embedding import FrequencyDomainFilteringEmbedding
from .streak_backbone import StreakTransformerEncoder
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
