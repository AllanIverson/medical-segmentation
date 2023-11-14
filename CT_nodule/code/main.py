import argparse
import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


from model import model_mae3d
from train.engine_pretrain3d import train_one_epoch,train_N_epoch,valid_one_epoch
import glob
from monai.data import decollate_batch
from util.mae_data_utils import get_loader
import torch.multiprocessing as mp
import torch.distributed as dist
from model.unet import weights_init


def get_args_parser():
    parser = argparse.ArgumentParser('MAE3D pre-training', add_help=False)
    parser.add_argument('--batch_size', default=5, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")

    # Model parameters
    parser.add_argument('--model', default='mae3d_base_16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=160, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.85, type=float,
                        help='Masking ratio 0(percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=4e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--num_segments', default=1, type=int)
    parser.add_argument('--frames_per_segment', default=160, type=int)
    parser.add_argument('--output_dir', default="/data2/zhanghao/runs/pretrain_model/mask85_wopix_goon/",
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default="/data2/zhanghao/runs/pretrain_model/mask85_wopix_goon/",
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default="/data2/zhanghao/runs/pretrain_model/mask85_wopix_goon/checkpoint-180.pth",
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=180, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument("--workers", default=2, type=int, help="number of workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    #data
    parser.add_argument("--use_normal_dataset", default=False,action="store_true", help="use monai Dataset class")
    parser.add_argument("--a_min", default=0.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=192, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=192, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=192, type=int, help="roi size in z direction")

    # distributed training parameters
    parser.add_argument("--distributed",default=True, action="store_true", help="start distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    # parser.add_argument('--dist_url', default='env://',
    #                     help='url used to set up distributed training')
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:23451", type=str, help="distributed url")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--norm-name", default='instance',action="store_true", help="use monai Dataset class")
    parser.add_argument("--Brats_json", default="/data2/zhanghao/trainset.json", help="Brats dictionary list")
    
    # data aug
    parser.add_argument("--RandScaleIntensityd_prob", default=0.2, type=float, help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.2, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument('--sw_batch_size',default=4)

    return parser

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES']='0,2'

    #whether use amp  default == false
    args.amp = not args.noamp

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu,args):
    print('Start distributed training')
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    train_loader = get_loader(args)[0]

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.output_dir

    # define the model
    model = model_mae3d.__dict__[args.model](
        num_frames=int(args.num_segments*args.frames_per_segment), norm_pix_loss=args.norm_pix_loss)
    weights_init(model)
    print("num_frame = ",args.num_segments*args.frames_per_segment)
    # model =torch.compile(model=model)
    model.cuda(args.gpu)
    model_without_ddp = model

    # define ddp model
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
        model_without_ddp = model.module

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    #freeze patch_embded param
    for name,param in model.named_parameters():
        if name=='patch_embed.proj.weight':
                param.requires_grad = False
                print('freeze ',name)
        if name=='patch_embed.proj.bias':
                param.requires_grad = False
                print('freeze ',name)

    # print("Model = %s" % str(model_without_ddp))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)


    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(
        model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()


    #load checkpoint
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs ,args.accum_iter):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            # valid_stats = valid_one_epoch(
            #     model, valid_loader,
            #     epoch,
            #     log_writer=log_writer,
            #     args = args
            # )
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
                

        train_log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        # vlaid_log_stats = {**{f'train_{k}': v for k, v in valid_stats.items()},
        #                 'epoch': epoch,}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(train_log_stats) + "\n")    

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
