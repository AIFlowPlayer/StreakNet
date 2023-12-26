#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All righs reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import torch 
from torch import nn 

from streaknet.data import cal_valid_results


class CMLoss(nn.Module):
    """
    Confusion Matrix Loss
    """
    def __init__(self, eps=1e-6):
        super(CMLoss, self).__init__()
        self.eps = eps 
    
    def forward(self, preds, labels):
        ans = torch.argmax(preds, 1)
        acc, precision, recall, f1, _ = cal_valid_results(ans, labels, self.eps)
        acc_loss = -torch.log(acc + self.eps)
        precision_loss = -torch.log(precision + self.eps)
        recall_loss = -torch.log(recall + self.eps)
        f1_loss = -torch.log(f1 + self.eps)
        return acc_loss, precision_loss, recall_loss, f1_loss
        

class StreakLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(StreakLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.cm_loss = CMLoss(eps)
    
    def forward(self, preds, labels):
        labels = labels.reshape(-1)
        cls_loss = self.cls_loss(preds, labels)
        acc_loss, precision_loss, recall_loss, f1_loss = self.cm_loss(preds, labels)
        total_loss = cls_loss + acc_loss + precision_loss + recall_loss + f1_loss
        loss_dict = {
            "total_loss": total_loss,
            "cls_loss":  cls_loss,
            "acc_loss": acc_loss,
            "recall_loss": recall_loss,
            "f1_loss": f1_loss
        }
        return loss_dict


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
        