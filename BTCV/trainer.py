# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch

def open_freeze_by_epoch(model,max_epoch,epoch,warm_epoch,args):
    # 计算哪层需要被open
    encoder_layers_num = args.count_layer # 这是encoder的总层数
    
    # 如果当前epoch小于warmup_epoch 则对encoder全部冻结
    if epoch < warm_epoch:
        print("冻结encoder全部参数,需要冻结的层数：",encoder_layers_num)
        idx = 0
        for name, param in model.named_parameters():
            if idx < encoder_layers_num:
                param.requires_grad = False
            else :
                param.requires_grad = True
            idx += 1
        return model
    else:
        open_layers =  encoder_layers_num *(1 - ( (epoch - warm_epoch) / (max_epoch - warm_epoch) ))
        print("当前epoch需要冻结的层数:",open_layers)
        idx = 0
        for name, param in model.named_parameters():
            if idx < open_layers:
                param.requires_grad = False
            else :
                param.requires_grad = True
            idx += 1
        if args.model_name == 'unetr':  # 对unetr进行特殊处理,永远不解冻patch_embed参数
            for name, param in model.named_parameters():
                if name == 'patch_embed.proj.weight':
                    param.requires_grad = False
                    print('freeze ', name)
                if name == 'patch_embed.proj.bias':
                    param.requires_grad = False
                    print('freeze ', name)
        return model
    
def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    if args.use_layer_finetune:
        model = open_freeze_by_epoch(model,args.max_epochs,epoch,args.warmup_epochs,args)
        if args.rank == 0:
            print("使用逐步放开微调策略下,本次epoch的参数量")
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Total parameters count", pytorch_total_params)
            
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        if args.out_channels == 36:
            target[target == 19] = 0
            target[target == 35] = 0
            # flat_tensor = target.view(-1).numpy()
            # missing_numbers = [num for num in range(37) if num not in flat_tensor]
            # print('没有出现的数字是:',missing_numbers)
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        print(data.shape,target.shape)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
            #
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()

    count = 0
    acc_all = 0
    print('count=',count,'acc_all=',acc_all)
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    print('使用sliding window 测试,args.amp = ',args.amp)
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            print('形状是',target.shape,logits.shape)
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            # ################################
            # 隐藏两个通道，不计算dice
            if args.out_channels == 36 and args.test_mode == True:
                val_output_convert[0] = torch.cat((val_output_convert[0][:18], val_output_convert[0][20:34]), dim=0)
                val_labels_convert[0] = torch.cat((val_labels_convert[0][:18], val_labels_convert[0][20:34]), dim=0)
                print(val_output_convert[0].size(),len(val_output_convert))
                print(val_labels_convert[0].size(),len(val_labels_convert))
            # ###############################
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            print('acc是什么',acc[0].item())

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
                acc_all += acc[0].item()
                count += 1
            start_time = time.time()
            if args.test_mode == True:
                print("平均dice:", acc_all /count)
    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    accuracy_list = []
    accuarcy_cnt = 0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)
            # accuracy_list.append(val_avg_acc)
            # accuarcy_cnt += 1
            # if accuarcy_cnt >=3 and (args.add + args.num_source) >= 100:
            #     if accuracy_list[accuarcy_cnt] < accuracy_list[accuarcy_cnt-1] and accuracy_list[accuarcy_cnt] < accuracy_list[accuarcy_cnt-2]:
            #         print('early stop')
            #         return val_acc_max

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max