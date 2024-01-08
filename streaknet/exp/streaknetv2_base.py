#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os 

import torch 
import torch.distributed as dist 
import torch.nn as nn

from .streaknet_base import Exp as Expv1

__all__ = ["Expv2"]


class Expv2(Expv1):
    def __init__(self):
        super().__init__()
    
    def get_model(self, export=False):
        from streaknet.models import FrequencyDomainFilteringEmbedding
        from streaknet.models import DoubleBranchCrossAttention
        from streaknet.models import ImagingHead, StreakNet

        if getattr(self, "model", None) is None:
            embedding = FrequencyDomainFilteringEmbedding(self.width, self.act, export=export)
            backbone = DoubleBranchCrossAttention(self.width, self.depth, self.dropout, self.act)
            head = ImagingHead(self.width, self.act, self.loss)
            self.model = StreakNet(embedding, backbone, head)

        self.model.train()
        return self.model
        