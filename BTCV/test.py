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
from utils import models_vit3d
import numpy as np
import torch
from utils.unetr import UNETR
from utils.unetr_data_utils import get_loader
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from trainer import val_epoch
from functools import partial
from monai.transforms import Activations, AsDiscrete, Compose
import SimpleITK as sitk


parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--feature_size", default=48, type=int, help="feature size dimention")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=192, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=192, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=192, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--sw_batch_size",default=4)
parser.add_argument('--use_normal_dataset',default=True)
parser.add_argument('--encoder_model',default='mae3d_160_base')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
parser.add_argument('--amp',default=True)
parser.add_argument('--rank',default=0)
parser.add_argument('--max_epochs',default=0)
parser.add_argument('--batch_size',default=1)

parser.add_argument('--use_all_source',default=True)
parser.add_argument('--source',default=None)
parser.add_argument('--add',default=0)
parser.add_argument('--model_name',default='swinunetr')
parser.add_argument('--segment_model',default="/data2/zhanghao/segment_run_skull_swin/cc359_base_all/model.pt")
parser.add_argument('--visible_path',default='/data2/zhanghao/visible_expirementxxxxxxxxxxxxxxxxxxxxxx/segment_with_synseg/skull_swin/full/')
def main():
    # /data2/FuQiang/adni_with_seg/data/4217/bl/ADNI_068_S_4217_MR_MT1__GradWarp__N3m_Br_20110910142439801_S121337_I255437.nii
    args = parser.parse_args()

    dice = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)

    args.test_mode = True
    args.finetune_size = 0
    if args.model_name == 'unetr':
        args.finetune_size = 160
    else:
        args.finetune_size = 96
    
    val_loader = get_loader(args)
    # ###########################################
    # image_path = '/data2/zhanghao/test_adni/image/'
    # label_path = '/data2/zhanghao/test_adni/label/'
    # # template_image  = sitk.ReadImage('/data/qiuhui/data/oasis/oasis1/OASIS_OAS1_0453_MR1/norm.nii.gz')
    # for i,data in enumerate(val_loader):
    #     image = data['image'].squeeze().numpy()
    #     label = data['label'].squeeze().numpy()
    #     itk_image = sitk.GetImageFromArray(image)
    #     # itk_image.CopyInformation(template_image)
    #     itk_label = sitk.GetImageFromArray(label)
    #     # itk_label.CopyInformation(template_image)

    #     sitk.WriteImage(itk_image, image_path+str(i)+".nii.gz")
    #     sitk.WriteImage(itk_label, label_path+str(i)+".nii.gz")

    # input()
    # #############################################


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = args.segment_model
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        if args.model_name == 'unetr':
            encoder = models_vit3d.__dict__[args.encoder_model](
                drop_path_rate = args.drop_path,
            )
            model = UNETR(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(160,160,160),
                feature_size=16,
                hidden_size=768,
                mlp_ratio=4.,
                num_heads=12,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.drop_path,
                encoder=encoder)
            model_dict = torch.load(pretrained_pth,map_location='cpu')
            model.load_state_dict(model_dict['state_dict'])
        else:
            model = SwinUNETR(
                img_size=(96,96,96),
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                feature_size=args.feature_size,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=False,
            )
            model_dict = torch.load(pretrained_pth,map_location='cpu')
            model.load_state_dict(model_dict['state_dict'])
    model.eval()
    model.to(device)

    # with torch.no_grad():
    #     dice_list_case = []
    #     for i, batch in enumerate(val_loader):
    #         val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
    #         print("形状是：",val_inputs.shape,val_labels.shape)
    #         img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
    #         print("Inference on case {}".format(img_name))
    #         val_outputs = sliding_window_inference(val_inputs, (160, 160, 160), 4, model, overlap=args.infer_overlap)
    #         val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
    #         val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
    #         val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
    #         dice_list_sub = []
    #         for i in range(1, 5):
    #             organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
    #             dice_list_sub.append(organ_Dice)
    #         mean_dice = np.mean(dice_list_sub)
    #         print("Mean Organ Dice: {}".format(mean_dice))
    #         dice_list_case.append(mean_dice)
    #     print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

    model_inferer = partial(
        sliding_window_inference,
        roi_size=(args.finetune_size,args.finetune_size,args.finetune_size),
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)

    #################
    # 仅仅生成可视化图像  （要求图像和标签都有）
    image_path = '/data2/zhanghao/visible_expirementxxxxxxxxxxxxxxxxxxxxxx/segment_with_synseg/image/'
    pred_by_zeroshot = args.visible_path
    print(pred_by_zeroshot)

    
    with torch.no_grad():

        # 加载原始图像
        # original_image_path = "/data/qiuhui/data/oasis/oasis1/OASIS_OAS1_0453_MR1/aligned_orig.nii.gz"
        # original_image = sitk.ReadImage(original_image_path)

        # 获取原始图像的轴向、空间信息和原点
        # original_direction = original_image.GetDirection()
        # original_spacing = original_image.GetSpacing()
        # original_origin = original_image.GetOrigin()

        for i,data in enumerate(val_loader):
            image = data['image'].to(device)
            label = data['label'].to(device)
            pred = model_inferer(image).cpu().detach().numpy()  # 1,36,192,192,192
            image = image.squeeze().cpu()
            label = label.squeeze().cpu()
            print(pred.shape)
            # pred = torch.cat((pred[0][:18], pred[0][20:34]), dim=0)
            # print(pred.shape)
            # 在 C 维度上找到最大值的索引
            argmax_indices = np.argmax(pred, axis=1)

            # 将最大值的索引转换为结果图（添加一个新的维度）
            result_map = np.expand_dims(argmax_indices, axis=1).squeeze()
            # unique_values = set(result_map.tolist())
            result_map = np.array(result_map, dtype=float)
            print(result_map.shape,result_map.dtype)
            name = data['image_meta_dict']['filename_or_obj'][0].split('/')[6]

            image_itk = sitk.GetImageFromArray(image)
            # image_itk.SetDirection(original_direction)
            # image_itk.SetSpacing(original_spacing)
            # image_itk.SetOrigin(original_origin)
            sitk.WriteImage(image_itk, image_path+name+'.nii.gz')
            pred_itk = sitk.GetImageFromArray(result_map)
            # pred_itk.SetDirection(original_direction)
            # pred_itk.SetSpacing(original_spacing)
            # pred_itk.SetOrigin(original_origin)
            sitk.WriteImage(pred_itk, pred_by_zeroshot+name)
            print(pred_by_zeroshot)

            print("NIfTI文件已保存:")
            input()
    
    
    #################
    val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=0,
                acc_func=dice,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
    val_avg_acc = np.mean(val_avg_acc)
    print('dice=',val_avg_acc)

if __name__ == "__main__":
    main()
