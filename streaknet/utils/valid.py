#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import numpy as np
from loguru import logger

from streaknet.data import cal_valid_results, cal_image_valid_results


def valid(gray_img, deep_img, mask, truth_mask):
    acc, precision, recall, f1, cls_psnr = cal_valid_results(mask, truth_mask)
    img_psnr, img_cnr = cal_image_valid_results(gray_img, truth_mask)
    dis_var = np.var(deep_img[mask == 1])
    ret = dict({
        "mask": {"accuracy": acc, "precision": precision, "recall": recall, "F1-Score": f1, "PSNR": cls_psnr},
        "image":{"PSNR": img_psnr, "CNR": img_cnr},
        "deep":{"variance": dis_var}
    })
    return ret


def merge_result(results):
    acc, precision, recall, f1, cls_psnr = [], [], [], [], []
    img_psnr, img_cnr = [], []
    dis_var = []
    for result in results:
        acc.append(result["mask"]["accuracy"])
        precision.append(result["mask"]["precision"])
        recall.append(result["mask"]["recall"])
        f1.append(result["mask"]["F1-Score"])
        cls_psnr.append(result["mask"]["PSNR"])
        img_psnr.append(result["image"]["PSNR"])
        img_cnr.append(result["image"]["CNR"])
        dis_var.append(result["deep"]["variance"])
    ret = dict({
        "mask": {"accuracy": np.mean(acc), "precision": np.mean(precision), "recall": np.mean(recall), "F1-Score": np.mean(f1), "PSNR": np.mean(cls_psnr)},
        "image":{"PSNR": np.mean(img_psnr), "CNR": np.mean(img_cnr)},
        "deep":{"variance": np.mean(dis_var)}
    })
    return ret


def log_result(result):
    logger.info("[Mask] Accuracy:{:.3f}% Precision:{:.3f}% Recall:{:.3f}% F1-Score:{:.3f}% PSNR:{:.4f}".format(
        result["mask"]["accuracy"] * 100, result["mask"]["precision"] * 100, 
        result["mask"]["recall"] * 100, result["mask"]["F1-Score"] * 100, result["mask"]["PSNR"]
    ))
    logger.info("[Image] PSNR:{:.4f} CNR:{:.4f}".format(result["image"]["PSNR"], result["image"]["CNR"]))
    logger.info("[Deep] Variance:{:.4f}".format(result["deep"]["variance"]))


def to_excel(excel_result, result, bias):
    excel_result[0, bias*7] = result["mask"]["accuracy"]
    excel_result[0, bias*7 + 1] = result["mask"]["precision"]
    excel_result[0, bias*7 + 2] = result["mask"]["recall"]
    excel_result[0, bias*7 + 3] = result["mask"]["F1-Score"]
    excel_result[0, bias*7 + 4] = result["image"]["PSNR"]
    excel_result[0, bias*7 + 5] = result["image"]["CNR"]
    excel_result[0, bias*7 + 6] = result["deep"]["variance"]
    return excel_result
