#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import itertools
import time
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import numpy as np

import torch

from streaknet.utils import (
    gather,
    is_main_process,
    synchronize,
    time_synchronized,
)


def cal_rates(tp, fn, fp, tn, eps=1e-6):
    p = tp + fn 
    n = fp + tn 
    acc = (tp + tn) / (p + n)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    mse = (fp + fn) * 255 ** 2 / (p + n + eps)
    psnr = 10 * np.log10(255 ** 2 / mse + eps)
    
    return acc, precision, recall, f1, psnr


def cal_image_valid_results(preds, labels, eps=1e-6):
    labels = labels * 255
    mse = np.mean(np.power(preds - labels, 2))
    psnr = 10 * np.log10(255 ** 2 / mse + eps)
    
    u_roi = np.mean(preds[labels > 127])
    u_noise = np.mean(preds[labels < 127])
    sig_roi = np.var(preds[labels > 127])
    sig_noise = np.var(preds[labels < 127])
    cnr = np.abs(u_roi - u_noise) / (np.sqrt(sig_roi + sig_noise) + eps)
    
    snr = 10 * np.log10(u_roi / u_noise)
    
    return psnr, snr, cnr


def cal_valid_results(preds, labels, eps=1e-6, origin=False):
    tp = (preds[labels == 1] == 1).sum()
    fn = (preds[labels == 1] == 0).sum()
    fp = (preds[labels == 0] == 1).sum() 
    tn = (preds[labels == 0] == 0).sum()
    if origin:
        return tp.item(), fn.item(), fp.item(), tn.item() 
    else:
        return cal_rates(tp, fn, fp, tn, eps)
    

class StreakEvaluator:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def evaluate(self, model, distributed=False, eps=1e-6):
        self.eps = eps
        
        tensor_type = torch.cuda.FloatTensor
        model = model.eval()

        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, (signal, template, gd, _) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                signal = signal.type(tensor_type)
                template = template.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(signal, template)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            gd = gd.reshape(-1)
            tp, fn, fp, tn = cal_valid_results(outputs, gd, origin=True)
            data_list.append([tp, fn, fp, tn])

        statistics = torch.cuda.FloatTensor([inference_time, n_samples])
        if distributed:
            # different process/device might have different speed,
            # to make sure the process will not be stucked, sync func is used here.
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        
        return eval_results

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, 0, 0, ""

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        n_samples = statistics[1].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.4f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "inference"],
                    [a_infer_time, a_infer_time],
                )
            ]
        )

        info = time_info + "\n"

        if len(data_dict) > 0:
            data_numpy = np.array(data_dict)
            tp = data_numpy[:, 0].sum()
            fn = data_numpy[:, 1].sum() 
            fp = data_numpy[:, 2].sum() 
            tn = data_numpy[:, 3].sum()
            
            acc, precision, recall, f1, psnr = cal_rates(tp, fn, fp, tn, self.eps)
            
            info += "accuracy:{:.3f}%, ".format(acc * 100)
            info += "precision:{:.3f}%, ".format(precision * 100)
            info += "recall:{:.3f}%, ".format(recall * 100)
            info += "f1-score:{:.3f}%, ".format(f1 * 100)
            info += "PSNR:{:.3f}dB\n".format(psnr)
            
            return acc, precision, recall, f1, psnr, info
        else:
            return 0, 0, 0, 0, 0, info
