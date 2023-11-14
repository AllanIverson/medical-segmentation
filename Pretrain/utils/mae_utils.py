import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist
import glob
import json
import random
import pandas as pd

#random sample from dataset for distributed ->  then be a new dataset(with random smple)


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        #set  num_replicas  ==  进程数
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()

        #set  rank == 当前进程序号
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(
            indices[self.rank: self.total_size: self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(
                        indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank: self.total_size: self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], reader='ITKReader'),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RSP"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear")
            ),
            transforms.ScaleIntensityRangePercentilesd(
                keys=["image", "label"], lower=5, upper=95, b_min=args.b_min, b_max=args.b_max, clip=True),
            # transforms.ScaleIntensityRanged(
            #     keys="image", a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            # transforms.ScaleIntensityRanged(
            #     keys="label", a_min=args.a_min2, a_max=args.a_max2, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),

            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(
                keys=["image", "label"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            ),
            transforms.RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                random_center=False,
                random_size=False,
                num_samples=1,
            ),

            transforms.Transposed(keys=["image", "label"], indices=(0, 3, 1, 2)),  # (channel,z,x,y)
            # transforms.RandCropByPosNegLabeld(
            #     keys="image",
            #     label_key="image",
            #     spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            # transforms.RandFlipd(keys="image", prob=args.RandFlipd_prob, spatial_axis=0),
            # transforms.RandFlipd(keys="image", prob=args.RandFlipd_prob, spatial_axis=1),
            # transforms.RandFlipd(keys="image", prob=args.RandFlipd_prob, spatial_axis=2),
            # transforms.RandRotate90d(keys="image", prob=args.RandRotate90d_prob, max_k=3),
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    pretrain_train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RSP"),
            transforms.ScaleIntensityRangePercentilesd(keys="image",lower=5,upper=95,b_min=args.b_min,b_max=args.b_max,clip=True),
            transforms.CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.CenterSpatialCropd(
                keys=["image"],
                roi_size = [args.roi_x, args.roi_y, args.roi_z],
            ),
            transforms.RandSpatialCropSamplesd(
                keys="image",
                roi_size = [96,96,96],
                random_size = False,
                num_samples  = args.sw_batch_size
            ),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    if True:

        '''
        ADNI的提取image和label路径的方式,其中label和image的路径一样
        '''
        # file_path = '/data/zhanghao/skull_project/CT_nodule/code/util/adni_list.txt'
        # list = []
        # with open(file_path) as file:
        #     content = file.readlines()
        # for i in range(len(content)):
        #     list.append(content[i].split('\n')[0])
        # dict_list = []
        # for i in list:
        #     dict = {"image": i, "label": i}
        #     dict_list.append(dict)
        # datalist = dict_list

        '''
        OASIS的提取image和label路径的方式,其中label和image的路径一样
        '''
        # data_path = pd.read_csv("/data/zhanghao/skull_project/CT_nodule/code/util/oasis_csv.csv")
        # list = []
        # for i in range(len(data_path)):
        #     list.append([data_path['image_paths'][i],data_path['mask4_paths'][i]])
        # for i in range(len(list)):
        #     list[i][0] = list[i][0].replace('\\','/')
        #     list[i][0] = list[i][0].replace('D:/oasis-seg4_35/','/data/qiuhui/code/graph/C2FViT/Data/OASIS/')
        #     list[i][1] = list[i][1].replace('\\', '/')
        #     list[i][1] = list[i][1].replace('D:/oasis-seg4_35/', '/data/qiuhui/code/graph/C2FViT/Data/OASIS/')
        # dict_list = []
        # for i in list:
        #     dict = {"image" : i[0] , "label" : i[1]}
        #     dict_list.append(dict)
        # train_list = dict_list[0:290]
        # valid_list = dict_list[290:331]
        # test_list = dict_list[331:]

        '''
        脑部MRI,混合数据集, ADNI OASIS (颅骨 有/无)
        '''
        adni_w = '/data2/zhanghao/Pretrain/utils/adni_w.txt'
        adni_wo = '/data2/zhanghao/Pretrain/utils/adni_wo.txt'
        oasis_w = '/data2/zhanghao/Pretrain/utils/oasis_w.txt'
        oasis_wo = '/data2/zhanghao/Pretrain/utils/oasis_wo.txt'
        files = [adni_w,adni_wo,oasis_w,oasis_wo]
        pretrian_list = []
        
        for file in files:
            with open(file) as f:
                content = f.readlines()
            list = []
            for i in range(len(content)):
                list.append(content[i].split('\n')[0])
            for i in list:
                dict = {"image": i, "label": i}
                pretrian_list.append(dict)
        random.shuffle(pretrian_list)
        train_list = pretrian_list[:-500]
        valid_list = pretrian_list[-500:]
        print('len pretrian :',len(train_list),len(valid_list))
    


        

        #if use cache dataset

        if args.use_normal_dataset:
            train_ds = data.Dataset(data=train_list, transform=pretrain_train_transforms)
            valid_ds = data.Dataset(data=valid_list, transform=pretrain_train_transforms)
            # test_ds = data.Dataset(data=test_list, transform=train_transform)
        else:
            # train_ds = data.SmartCacheDataset(data=train_list, transform=zirui_transfrom,
            #                                   replace_rate=1, cache_num=240, progress=True)   # as_contiguous =True
            train_ds = data.CacheDataset(data=train_list, transform=pretrain_train_transforms,
                    cache_num=30,progress=True,cache_rate=0.01,num_workers=args.workers)
            valid_ds = data.CacheDataset(data=valid_list, transform=pretrain_train_transforms,
                    cache_num=20,progress=True,cache_rate=0.01,num_workers=args.workers)

        train_sampler = Sampler(train_ds) if args.distributed else None
        # valid_sampler = Sampler(valid_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=False,   
            persistent_workers=True,
        )
        valid_loader = data.DataLoader(
            valid_ds,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers= args.workers,
            sampler = None,
            pin_memory = False,
            persistent_workers=True
        )
        loader = [train_loader, valid_loader]

    return loader
