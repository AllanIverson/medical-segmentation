import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import time
import random


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    start_time = time.time()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    print('train loader 的 长度 是',len(data_loader))


    # for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step,data in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            # print('data_iter_step = ',data_iter_step)
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        samples = data["image"]  #自己的数据是data["image"]表示图片
        samples = samples.cuda(args.rank)
        for param in model.parameters():
            param.grad = None

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        #whether loss is NAN
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        #loss backward
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        
        #recode loss list
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    end_time = time.time()
    print('train 第'+str(epoch)+' need '+ str(end_time-start_time)+'s') 
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_N_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    start_time = time.time()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    print('train loader 的 长度 是',len(data_loader))

    #use grad accum_iter,for example accum_iter = 8
    for small_epoch in range(accum_iter):
        if (small_epoch+1) % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, small_epoch / len(data_loader) + epoch, args)

        for data_iter_step,data in enumerate(data_loader):

            if random.randint(0,1) == 0 :
                samples = data["image"]  #自己的数据是data["image"]表示图片
            else:
                samples = data['label']
            samples = samples.cuda(args.rank)

            with torch.cuda.amp.autocast():
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

            loss_value = loss.item()

            #whether loss is NAN
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
        for param in model.parameters():
            param.grad = None
        #loss backward
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(small_epoch + 1) % accum_iter == 0)

        if (small_epoch + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        
        #recode loss list
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (small_epoch + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((small_epoch / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    end_time = time.time()
    print('accum_iter=',accum_iter,'  train 第'+str(epoch)+' need '+ str(end_time-start_time)+'s') 
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def valid_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    epoch:int,
                    log_writer=None,
                    args=None):
    with torch.no_grad():
        model.eval()
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        start_time = time.time()
        if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))
        
        print('valid loader 的 长度 是',len(data_loader))

        for data_iter_step,data in enumerate(data_loader):
            # we use a per iteration (instead of per epoch) lr scheduler

            random.seed(0)
            if random.randint(0,1) == 0 :
                samples = data["image"]  #自己的数据是data["image"]表示图片
            else:
                samples = data['label']
            samples = samples.cuda(args.rank)
            for param in model.parameters():
                param.grad = None

            
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

            loss_value = loss.item()

            #whether loss is NAN
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            torch.cuda.synchronize()
            
            #recode loss list
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=-0.01)


            loss_value_reduce = misc.all_reduce_mean(loss_value)

            if log_writer is not None and (data_iter_step + 1) % 1 == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('valid_loss', loss_value_reduce, epoch_1000x)

        end_time = time.time()
        print('valid 第'+str(epoch)+' need '+ str(end_time-start_time)+'s') 
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}