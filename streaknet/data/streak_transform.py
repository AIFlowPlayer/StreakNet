#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import torch
import numpy as np


class StreakTransform(object):
    def __init__(self, batch=False):
        self.dim = 2 if batch else 0
        pass
    
    def __call__(self, signal, template=None, ground_truth=None, info=None): 
        signal_max = signal.max(self.dim, True)
        signal_min = signal.min(self.dim, True)
        if isinstance(signal_max, tuple): signal_max = signal_max[0]
        if isinstance(signal_min, tuple): signal_min = signal_min[0]
        signal = (signal - signal_min) / (signal_max - signal_min)
        
        if template is not None:
            template_max = template.max(self.dim, True)
            template_min = template.min(self.dim, True)
            if isinstance(template_max, tuple): template_max = template_max[0]
            if isinstance(template_min, tuple): template_min = template_min[0]
            template = (template - template_min) / (template_max - template_min)
        
        if ground_truth is not None:
            return signal, template, ground_truth, info
        elif template is not None:
            return signal, template
        else:
            return signal
        