# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import time


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    cuda_rank:int,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    
    # prefetcher = data_prefetcher(data_loader)
    start_time = time.time()
    cpu_label = time.time()
    gpu_label = time.time()
    print('0epoch = ',time.time())
    for data_iter_step,data in enumerate(data_loader):



        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = data["image"]  #我自己的数据是data["image"]表示图片

        samples = samples.to(device, non_blocking=True)
        
        # samples = samples.cuda(cuda_rank)
        print('***',time.time()-gpu_label)
        cpu_label = time.time()
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # torch.cuda.synchronize()
        print('tuili=',time.time()-cpu_label)
        gpu_label = time.time()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# class data_prefetcher():
#     def __init__(self, loader):
#         #loader 1：real
#         #loader 2：fake
#         self.stream = torch.cuda.Stream()
#         self.loader = iter(loader)
#         self.preload()
 
#     def preload(self):
#         try:
#             self.next_input, self.next_target = next(self.loader)
#         except StopIteration:
#             self.next_input = None
#             self.next_target = None
#             return
#         with torch.cuda.stream(self.stream):
#             self.next_input = self.next_input.cuda(non_blocking=True).float()
#             self.next_target = self.next_target.cuda(non_blocking=True).long()
 
#     def next(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         input = self.next_input
#         target = self.next_target
#         self.preload()
#         return input, target