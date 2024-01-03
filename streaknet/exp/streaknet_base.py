#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All rights reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import os 

import torch 
import torch.distributed as dist 
import torch.nn as nn

from .base_exp import BaseExp

__all__ = ["Exp", "check_exp_value"]


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        
        # ---------------- model config ---------------- #
        # factor of model depth
        self.depth = 1.00
        # factor of model width
        self.width = 1.00
        # factor of model dropout
        self.dropout = 0.40
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"
        # loss function: ["streakloss", "crossloss"]
        self.loss = "crossloss"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        # dir of dataset
        self.data_dir = "datasets"
        # name of config file for training
        self.train_config = "train_config.yaml"
        # name of config file for evaluation
        self.valid_config = "valid_config.yaml"
        # name of config file for testing
        self.test_config = "test_config.yaml"

        # --------------- transform config ----------------- #
        # Nothing

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 1
        # max training epoch
        self.max_epoch = 40
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one signal. During training, lr will multiply batchsize.
        self.basic_lr_per_signal = 1e-4 * 0.02
        # name of LRScheduler
        self.scheduler = "streakwarmcos"
        # last #epoch to close augmention
        self.no_aug_epochs = 10
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 100
        # visualize period in iter of warmup epochs
        self.viz_interval = 50
        self.max_viz_iter = 1000
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 2
        # save history checkpoint or not.
        # If set to False, streaknet will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
    
    def get_model(self, export=False):
        from streaknet.models import FrequencyDomainFilteringEmbedding
        from streaknet.models import StreakTransformerEncoder
        from streaknet.models import SingleBranchClsHead, StreakNet

        if getattr(self, "model", None) is None:
            embedding = FrequencyDomainFilteringEmbedding(self.width, self.dropout, self.act, export=export)
            backbone = StreakTransformerEncoder(self.width, self.depth, self.dropout, self.act)
            head = SingleBranchClsHead(self.width, self.act, self.loss)
            self.model = StreakNet(embedding, backbone, head)

        self.model.train()
        return self.model

    def get_dataset(self, cache=False):
        from streaknet.data import StreakSignalDataset, StreakTransform
        
        return StreakSignalDataset(
            data_dir=self.data_dir,
            config_file=self.train_config,
            transform=StreakTransform,
            cache=cache
        )
    
    def get_data_loader(self, batch_size, is_distributed, cache=False):
        from streaknet.utils import wait_for_the_master
        from torch.utils.data import DataLoader
        
        with wait_for_the_master():
            self.dataset = self.get_dataset(cache)
        
        if is_distributed:
            batch_size = batch_size // dist.get_world_size() 
        
        train_loader = DataLoader(
            dataset=self.dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=self.data_num_workers, 
            pin_memory=True, 
            drop_last=False,
        )

        return train_loader
    
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1 = [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg1.append(v.bias)  # biases
                if hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg0.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg1, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg0, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            self.optimizer = optimizer

        return self.optimizer
    
    def get_lr_scheduler(self, lr, iters_per_epoch):
        from streaknet.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler
    
    def get_eval_dataset(self, cache=False):
        from streaknet.data import StreakSignalDataset, StreakTransform
        return StreakSignalDataset(
            data_dir=self.data_dir,
            config_file=self.valid_config,
            transform=StreakTransform,
            cache=cache
        )
    
    def get_test_dataset(self, cache=False):
        from streaknet.data import StreakSignalDataset, StreakTransform 
        return StreakSignalDataset(
            data_dir=self.data_dir, 
            config_file=self.test_config, 
            transform=StreakTransform,
            cache=cache
        )

    def get_eval_loader(self, batch_size, is_distributed, cache, test=False):
        if test:
            valdataset = self.get_test_dataset(cache)
        else:
            valdataset = self.get_eval_dataset(cache)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
    
    def get_evaluator(self, batch_size, is_distributed, cache=False, test=False):
        from streaknet.data import StreakEvaluator
        return StreakEvaluator(dataloader=self.get_eval_loader(batch_size, is_distributed, cache, test))
    
    def get_trainer(self, args):
        from streaknet.core import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer
    
    def eval(self, model, evaluator, is_distributed):
        return evaluator.evaluate(model, is_distributed)


def check_exp_value(exp: Exp):
    from loguru import logger 
    if exp.width not in [0.125, 0.25, 0.50, 1.00]:
        logger.warning("Your factor exp.width={:.3f} is not standard, it is suggested to set to 0.125, 0.25, 0.50 and 1.00.".format(exp.width))
    if exp.depth not in [0.125, 0.25, 0.50, 1.00]: 
        logger.warning("Your factor exp.depth={:.3f} is not standard, it is suggested to set to 0.125, 0.25, 0.50 and 1.00.".format(exp.depth))
    embedding_size = round(512 * exp.width)
    num_heads = round(16 * exp.width)
    assert embedding_size % num_heads == 0, "Embedding size must be multiples of num_heads."
        