# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
# from networks.unetr import UNETR
from monai.networks.nets import SwinUNETR
from utils.unetr import UNETR
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.unetr_data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from utils import models_vit3d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
# checkpoint path
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
# log path
parser.add_argument("--logdir", default="", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--log_dir", default="", type=str, help="directory to save the tensorboard logs")
parser.add_argument('--output_dir', default="",
                        help='path where to save, empty for no saving')
# use how many data in source domain and target domain
parser.add_argument('--add',default=0,type=int,help='number of add target domain')
parser.add_argument('--num_source',default=0,type=int,help='number of source domain')
parser.add_argument('--use_all_source',default=True)
parser.add_argument('--source',default='oasis')
# these may not be needed,you can load your pretrained model by your may.
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
# model set
parser.add_argument("--model_name", default="unetr", type=str, help="model name")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=48, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=5, type=int, help="number of output channels(background is 0)")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
# data process
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=192, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=192, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=192, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
# the training set
parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=100, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
parser.add_argument("--distributed", default=True,action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23457", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=2, type=int, help="number of workers")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--infer-size",default=160)
parser.add_argument('--start_epoch',type=int,default=0,help="when use checkpoint,need modify start epoch")
parser.add_argument('--use_pretrain',default=True,help="whether use pretrain model")
parser.add_argument('--use_base',action="store_true")
parser.add_argument("--use_layer_finetune",default=False)
parser.add_argument('--encoder_model', default='mae3d_160_base', type=str, metavar='MODEL',
                        help='Name of model to train')
parser.add_argument('--finetune',default='/data2/zhanghao/runs/pretrain_model/mask85_wopix_goon/checkpoint-150.pth')

# parser.add_argument('--finetune', default='',
#                         help='finetune from checkpoint')
# parser.add_argument('--finetune', default='',
#                         help='finetune from checkpoint')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
parser.add_argument('--finetune_size',default=160,help="input size")


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    # Whether to use mixed precision
    args.amp = not args.noamp
    # Set your own input size
    if args.model_name == 'swinunetr':
        args.finetune_size = 96
    else:
        args.finetune_size = 160

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):

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
    loader = get_loader(args)
    print("len train loader = ",len(loader[0]),"len valid loader = ",len(loader[1]))

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.infer_size,args.infer_size,args.infer_size]
    pretrained_dir = args.pretrained_dir

############################################  load  pretrained model  ############################
    if (args.model_name is None) or args.model_name == "unetr":
        encoder = models_vit3d.__dict__[args.encoder_model](
            drop_path_rate = args.drop_path,
        )
        # whether use pretrianing model
        if args.use_pretrain == True:
            print("use encoder")
            checkpoiont = torch.load(args.finetune,map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoiont['model']
            state_dict = encoder.state_dict()
            msg = encoder.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            total_parameters = count_parameters(encoder)

            print("The total number of parameters of the encoder is：", total_parameters)

        encoder.cuda(args.gpu)

        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(160,160,160),
            feature_size=16,
            hidden_size=args.hidden_size,
            mlp_ratio=4.,
            num_heads=args.num_heads,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.drop_path,
            encoder=encoder)
        
        print('args.use_base',args.use_base)
        # whether use model which is pretiraned and public data finetuned. we call it is "base model".
        if args.use_base == True: 
            print('use base model')
            model_dict = torch.load(args.checkpoint,map_location='cpu')
            print("Load base model checkpoint from: %s" % args.checkpoint)
            model.load_state_dict(model_dict['state_dict'])



        #freeze patch_embded param
        for name, param in encoder.named_parameters():
            if name == 'patch_embed.proj.weight':
                param.requires_grad = False
                print('freeze ', name)
            if name == 'patch_embed.proj.bias':
                param.requires_grad = False
                print('freeze ', name)

        # num_layers = len(list(encoder.children()))
        count_layer = 0
        for name, param in encoder.named_parameters():
            count_layer += 1
            # print(count_layer,name)
        args.count_layer = count_layer
        print("Number of layers in the encoder:", count_layer)
       
####################################################################################################
    elif (args.model_name is None) or args.model_name == "swinunetr":
        print("use swin unetr")
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
        )


        # Check that the parameter has been set to require no gradient
        total_parameters = count_parameters(model)
        print("Total number of parameters of the model：", total_parameters)

        # # All parameters of the model were set to require no gradient when 
        # for param in model.parameters():
        #     param.requires_grad = False

        if args.use_pretrain == True:
            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoiont = torch.load(args.finetune,map_location='cpu')
            state_dict = checkpoiont["state_dict"]
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            msg = model.load_state_dict(state_dict, strict=False)
            
            total_parameters = count_parameters(model)

            print(msg)
#####################################################################################################
    else:
        raise ValueError("Unsupported model " + str(args.model_name))
    
    if args.resume_ckpt:
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
        model.load_state_dict(model_dict)
        print("Use pretrained weights")
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)

    if args.resume_jit:
        if not args.noamp:
            print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
            args.amp = args.noamp
        model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))

    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
    )
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = args.start_epoch

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )

    return accuracy

if __name__ == "__main__":
    main()

# for example
# python /data2/zhanghao/BTCV/main.py --logdir /data2/zhanghao/segment_run/oasis4_baseN_addadniN_onadni_w/5 --log_dir /data2/zhanghao/segment_run/oasis4_baseN_addadniN_onadni_w/5 --output_dir /data2/zhanghao/segment_run/oasis4_baseN_addadniN_onadni_w/5 --add 5 --use_all_source True --source oasis 
# python /data2/zhanghao/BTCV/main.py --logdir /data2/zhanghao/segment_run/oasis4_baseN_addadniN_onadni_w/5 --log_dir /data2/zhanghao/segment_run/oasis4_baseN_addadniN_onadni_w/5 --output_dir /data2/zhanghao/segment_run/oasis4_baseN_addadniN_onadni_w/5 --add 5 --use_all_source True --source oasis --num_source 0 --use_base True --checkpoint /data2/zhanghao/segment_run/oasis4_base_onadni_w/all/model.pt
