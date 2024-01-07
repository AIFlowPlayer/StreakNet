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
        from streaknet.models import DoubleBranchFrequencyDomainEmbedding
        from streaknet.models import DoubleBranchCrossAttention
        from streaknet.models import ImagingHead, StreakNetV2

        if getattr(self, "model", None) is None:
            embedding = DoubleBranchFrequencyDomainEmbedding(self.width, self.act, concat=False, export=export)
            backbone = DoubleBranchCrossAttention(self.width, self.depth, self.dropout, self.act)
            head = ImagingHead(self.width, self.act, self.loss, len=4)
            self.model = StreakNetV2(embedding, backbone, head)

        self.model.train()
        return self.model
        