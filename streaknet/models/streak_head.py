#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All righs reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import torch 
from torch import nn 

from .network_blocks import get_activation, get_loss


class SingleBranchClsHead(nn.Module):
    def __init__(self, width=1.0, act='silu', loss='crossloss'):
        super(SingleBranchClsHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(round(1024 * width) * 2, 2)
        self.act = get_activation(act, inplace=False)
        self.losses = get_loss(loss)
    
    def forward(self, x, labels=None):
        flatten = self.flatten(x)
        pred = self.act(self.fc(flatten))
        if self.training:
            return self.get_losses(pred, labels)
        else:
            pred = torch.argmax(pred, 1)
            return pred

    def get_losses(self, x, labels):
        loss_dict = self.losses(x, labels)
        return loss_dict
    