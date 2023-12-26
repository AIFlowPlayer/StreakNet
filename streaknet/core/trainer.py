#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All righs reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from streaknet.exp import Exp
from streaknet.utils import (
    MeterBuffer,
    ModelEMA,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)
from streaknet.data import DataPrefetcher


class Trainer:
    def __init__(self, exp: Exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt
        self.cache = args.cache

        # data/dataloader related attr
        self.best_f1 = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            logger.info("init prefetcher, this might take one minute or less...")
            self.prefetcher = DataPrefetcher(self.train_loader)
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            ret = self.train_one_iter()
            if ret:
                break
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()
        signal, template, gd, _ = self.prefetcher.next()
        if signal is None:
            return True

        signal = signal.cuda()
        template = template.cuda()
        gd = gd.cuda()
        
        gd.requires_grad = False 
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(signal, template, gd)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )
        return False

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            cache=self.cache
        )
        
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_signal * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed, cache=self.cache
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            else:
                raise ValueError("logger must be 'tensorboard'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best F1 is {:.2f}".format(self.best_f1 * 100)
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            self.exp.eval_interval = 1

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.4f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", {}".format(eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "tensorboard":
                    self.tblogger.add_scalar(
                        "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                    for k, v in loss_meter.items():
                        self.tblogger.add_scalar(
                            f"train/{k}", v.latest, self.progress_in_iter)

            self.meter.clear_meters()
        
        if (self.epoch * self.max_iter) + self.iter <= self.exp.max_viz_iter and (self.iter + 1) % self.exp.viz_interval == 0:
            self.save_ckpt(f"visualize_iter_{(self.epoch * self.max_iter) + self.iter + 1}", f1=self.best_f1)

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_f1 = ckpt.pop("best_f1", 0)
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            acc, precision, recall, f1, psnr, summary = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed
            )

        update_best_ckpt = f1 > self.best_f1
        self.best_f1 = max(self.best_f1, f1)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/acc", acc, self.epoch + 1)
                self.tblogger.add_scalar("val/precision", precision, self.epoch + 1)
                self.tblogger.add_scalar("val/recall", recall, self.epoch + 1)
                self.tblogger.add_scalar("val/f1", f1, self.epoch + 1)
                self.tblogger.add_scalar("val/psnr", psnr, self.epoch + 1)
                
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, f1=f1)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", f1=f1)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, f1=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_f1": self.best_f1,
                "curr_f1": f1,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
