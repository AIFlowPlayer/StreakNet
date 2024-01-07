#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import torch 
from torch import nn 


class CrossLoss(nn.Module):
    def __init__(self):
        super(CrossLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        
    def forward(self, preds, labels):
        labels = labels.reshape(-1)
        cls_loss = self.cls_loss(preds, labels)
        loss_dict = {
            "total_loss": cls_loss,
            "cls_loss": cls_loss
        }
        return loss_dict
