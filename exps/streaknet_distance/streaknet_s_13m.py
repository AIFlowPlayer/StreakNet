#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os 

from streaknet.exp import Exp as MyExp 


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.125
        self.width = 0.125
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        self.train_config = dict({
            "sub_datasets": ["clean_water_13m"],
            "config": [
                {"sub_idx": 0, "row_slice": [0, 2048], "col_slice": [0, 165]}
            ]
        })
        
        self.valid_config = dict({
            "sub_datasets": ["clean_water_13m"],
            "config": [
                {"sub_idx": 0, "row_slice": [1047, 1667], "col_slice": [165, 262]}
            ]
        })
