#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All righs reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os
import cv2
import yaml
import numpy as np

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from loguru import logger
    

class StreakData(Dataset):
    def __init__(self, data_dir, config_file, transform=None, cache=False, max_len=1024):
        if transform is not None:
            self.transform = transform()
        else:
            self.transform = None
        
        # read config file
        with open(os.path.join(data_dir, config_file), "r") as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)
            sub_datasets = data["sub_datasets"]
            config = data["config"]
        
        # init vars
        self.data_dir = []
        self.ground_truth = []
        self.col_list = []
        self.template = []
        self.total_col = 0
        
        self.cache = cache
        self.data = None
        
        for name in sub_datasets:
            # ground-truth
            path = os.path.join(data_dir, name, "groundtruth.npy")
            logger.info("loading {} ...".format(path))
            self.ground_truth.append(np.load(path))
            
             # data-dir
            self.data_dir.append(os.path.join(data_dir, name, "data"))
            
            # col-list
            self.col_list.append(self.ground_truth[-1].shape[1])
            self.total_col += self.col_list[-1]
            
            # template
            path = os.path.join(data_dir, name, "template.npy")
            template = np.load(path).astype(np.float32)
            template_padding = np.full((max_len,), np.mean(template))
            template_padding[:template.shape[0]] = template
            self.template.append(template_padding)
        
        # init index
        idx = []
        for config_i in config:
            index = config_i["sub_idx"]
            row_slice = config_i["row_slice"]
            col_slice = config_i["col_slice"]
            logger.info("initializing subdataset {} ...".format(sub_datasets[index]))
            for row in range(row_slice[0], row_slice[1]):
                for col in range(col_slice[0], col_slice[1]):
                    idx.append([index, row, col])
        logger.info("Convert to numpy array ...")
        self.idx = np.array(idx, dtype=np.uint16)     
        del(idx)
        
        # cache
        if self.cache:
            self.data = []
            logger.info("Loading cache to RAM...")
            pbar = tqdm(total=self.total_col)
            for i, (col_num, tif_dir) in enumerate(zip(self.col_list, self.data_dir)):
                self.data.append(np.zeros((2048, col_num, 2048), dtype=np.uint16))
                for j in range(col_num):
                    self.data[-1][:, j, :] = cv2.imread(os.path.join(self.data_dir[i], "{:03d}.tif".format(j + 1)), -1)
                    pbar.update(1)
        else:
            logger.info("You haven't used RAM cache, the program will read data from disk every time.")
    
    def __del__(self):
        del self.template
        del self.ground_truth
        del self.data_dir 
        del self.col_list
        del self.total_col
        del self.idx 
        if self.cache:
            del self.data 
            del self.cache
        
    def __getitem__(self, index):
        idx = self.idx[index, 0]
        row = self.idx[index, 1]
        col = self.idx[index, 2]
        
        if self.cache:
            signal = self.data[idx][row, col]
        else:
            signal = cv2.imread(os.path.join(self.data_dir[idx], "{:03d}.tif".format(col + 1)), -1)[row, :]
        signal = signal.astype(np.float32)
        gd = self.ground_truth[idx][row, col]
        
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        gd_tensor = torch.tensor([gd], dtype=torch.int64)
        info_tensor = torch.tensor([idx, row, col], dtype=torch.int32)
        template_tensor = torch.tensor(self.template[idx], dtype=torch.float32)
        
        if self.transform:
            signal_tensor, template_tensor, gd_tensor, info_tensor = \
                self.transform(signal_tensor, template_tensor, gd_tensor, info_tensor)
        
        return signal_tensor, template_tensor, gd_tensor, info_tensor
    
    def __len__(self):
        return self.idx.shape[0]
    