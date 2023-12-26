#!/usr/bin/env3 python3
# Copyright (c) School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. All righs reserved.
# Author: Hongjun An (Coder.AN)
# Email: an.hongjun@foxmail.com

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_signal 
        self.record_stream = DataPrefetcher._record_stream_for_signal 
        self.preload()

    def preload(self):
        try:
            self.next_signal, self.next_template, self.next_gd, self.next_info = next(self.loader)
        except StopIteration:
            self.next_signal = None 
            self.next_template = None 
            self.next_gd = None 
            self.next_info = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_gd = self.next_gd.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        signal = self.next_signal
        template = self.next_template 
        ground_truth = self.next_gd
        info = self.next_info
        if signal is not None:
            self.record_stream(signal)
        if template is not None:
            self.record_stream(template)
        if ground_truth is not None:
            ground_truth.record_stream(torch.cuda.current_stream())
        self.preload()
        return signal, template, ground_truth, info

    def _input_cuda_for_signal(self):
        self.next_signal = self.next_signal.cuda(non_blocking=True)
        self.next_template = self.next_template.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_signal(input):
        input.record_stream(torch.cuda.current_stream())