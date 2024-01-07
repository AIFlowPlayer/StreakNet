#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os 

from streaknet.exp import Expv2 as MyExp 


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.25
        self.width = 0.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
    def get_model(self, export=False):
        from streaknet.models import DoubleBranchFrequencyDomainEmbedding
        from streaknet.models import StreakTransformerEncoder
        from streaknet.models import ImagingHead, StreakNetV2

        if getattr(self, "model", None) is None:
            embedding = DoubleBranchFrequencyDomainEmbedding(self.width, self.act, concat=False, export=export)
            backbone = StreakTransformerEncoder(self.width, self.depth, self.dropout, self.act)
            head = ImagingHead(self.width, self.act, self.loss, len=4)
            self.model = StreakNetV2(embedding, backbone, head)

        self.model.train()
        return self.model
        