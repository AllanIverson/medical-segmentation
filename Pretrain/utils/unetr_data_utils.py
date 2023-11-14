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
import csv
import random
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)


def random_sample_csv(csv_file, sample_size):
    list10 = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 读取标题行
        data = list(reader)    # 读取数据行

    # 随机抽样指定数量的条目
    sample = random.sample(data, sample_size)

    # 将抽样结果打印出来
    print(header)  # 打印标题行
    for row in sample:
        list10.append([row[2],row[3]])
    return list10

#random sample from dataset for distributed ->  then be a new dataset(with random smple)
class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        #set  num_replicas  ==  进程数 
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        
        #set  rank == 当前进程序号
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

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
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"],reader='ITKReader'), 
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RSP"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=["bilinear","nearest"]
            ),
            transforms.ScaleIntensityRangePercentilesd(keys="image",lower=5,upper=95,b_min=args.b_min,b_max=args.b_max,clip=True),
            # transforms.ScaleIntensityRanged(
            #     keys="image", a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(
                keys =["image", "label"],
                spatial_size = (args.roi_x, args.roi_y, args.roi_z),
            ),
            # transforms.RandSpatialCropSamplesd(  
            #     keys=["image", "label"],
            #     roi_size=(args.roi_x, args.roi_y, args.roi_z),
            #     random_center=False,
            #     random_size=False,
            #     num_samples = 1,
            # ),
            transforms.RandCropByPosNegLabeld(
                keys="image",
                label_key="image",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            # transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.Transposed(keys=["image","label"] , indices = (0,3,1,2)), # (channel,z,x,y)
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    pretrain_train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RSP"),
            transforms.ScaleIntensityRangePercentilesd(keys="image",lower=5,upper=95,b_min=args.b_min,b_max=args.b_max,clip=True),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            transforms.Transposed(keys="image" , indices = (0,3,1,2)), # (channel,z,x,y)
            ToTensord(keys=["image"]),
        ]
    )
    pretrain_val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RSP"),
            transforms.ScaleIntensityRangePercentilesd(keys="image",lower=5,upper=95,b_min=args.b_min,b_max=args.b_max,clip=True),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            transforms.Transposed(keys="image", indices = (0,3,1,2)), # (channel,z,x,y)
            ToTensord(keys=["image"]),
        ]
    )
    swin_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"],reader='ITKReader'),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RSP"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRangePercentilesd(keys="image",lower=5,upper=95,b_min=args.b_min,b_max=args.b_max,clip=True),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.Transposed(keys=["image","label"] , indices = (0,3,1,2)), # (channel,z,x,y)
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"],reader='ITKReader'), 
        transforms.AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RSP"),
        transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=["bilinear","nearest"]
            ),
        transforms.ScaleIntensityRangePercentilesd(keys="image",lower=5,upper=95,b_min=args.b_min,b_max=args.b_max,clip=True),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.SpatialPadd(
                keys =["image", "label"],
                spatial_size = (160,160,192), # 160,160,192
            ),
        transforms.CenterSpatialCropd(keys=["image","label"],roi_size=(160,160,192)),
        transforms.Transposed(keys=["image","label"] , indices = (0,3,1,2)), # (channel,z,x,y)
        transforms.ToTensord(keys=["image", "label"]),
    ]
    )

    trans_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"],reader='ITKReader'), 
        transforms.AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RSP"),
        transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=["bilinear","nearest"]
            ),
        # transforms.ScaleIntensityRangePercentilesd(keys="image",lower=5,upper=95,b_min=args.b_min,b_max=args.b_max,clip=True),
        # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.SpatialPadd(
                keys =["image", "label"],
                spatial_size = (args.roi_x, args.roi_y, args.roi_z), # 160,160,192
            ),
        transforms.CenterSpatialCropd(keys=["image","label"],roi_size=(args.roi_x, args.roi_y, args.roi_z)),
        # transforms.Transposed(keys=["image","label"] , indices = (0,3,1,2)), # (channel,z,x,y)
        transforms.ToTensord(keys=["image", "label"]),
    ]
    )



    if True:

        '''
        ADNI的提取image和label路径的方式,其中label和image的路径一样
        '''
        # file_path = '/data2/zhanghao/CT_nodule/code/util/adni_list.txt'
        file_path = '/data/zhanghao/oasis_adni_4_35/adni_new.txt'
        list = []
        with open(file_path) as file:
            content = file.readlines()
        for i in range(len(content)):
            list.append(content[i].split('\n')[0])
        dict_list = []
        for i in list:
            dict = {"image" : i , "label" : i}
            dict_list.append(dict)
        train_list = dict_list[:]
        test_list = dict_list[:]
        valid_list = dict_list[:]
        print(len(train_list))
        '''
        加载adni + oasis 的 pretrain 数据
        '''
        # import random
        # adni_path = '/data/zhanghao/oasis_adni_4_35/adni_pretrain.txt'
        # oasis_csv = '/data/zhanghao/oasis_adni_4_35/oasis4_35.csv'
        # # 加载 adni
        # list = []
        # with open(adni_path) as file:
        #     content = file.readlines()
        # for i in range(len(content)):
        #     list.append(content[i].split('\n')[0])
        # dict_list = []
        # for i in list:
        #     dict = {"image" : i , "label" : i}
        #     dict_list.append(dict)
        # # 加载oasis
        # oasis_pd = pd.read_csv(oasis_csv)
        # for i in range(len(oasis_pd)):
        #     dict = {"image":oasis_pd['image_paths'][i],"label":oasis_pd['mask4_paths'][i]}
        #     dict_list.append(dict)
        # train_list = random.sample(dict_list, len(dict_list))[:-10]
        # test_list = random.sample(dict_list, len(dict_list))
        # valid_list = random.sample(dict_list, len(dict_list))[-10:]
        # print('len train = ',len(train_list),'len valid =',len(valid_list))

        '''
        OASIS的提取image和label路径的方式,其中label和image的路径一样
        '''
        # data_path = pd.read_csv("/data/zhanghao/oasis_adni_4_35/oasis4_35.csv")
        # dict_list=[]
        # for i in range(len(data_path)):
        #     dict = {"image" : data_path['image_paths'][i] , "label" : data_path["mask4_paths"][i]}
        #     dict_list.append(dict)
        # train_list = dict_list
        # test_list = dict_list
        # valid_list = dict_list

        if args.test_mode:
            if args.use_normal_dataset:
                test_ds = data.Dataset(data=valid_list, transform=test_transform)

            else:
                test_ds = data.CacheDataset(data=valid_list,transform=test_transform,
                        cache_num=0,progress=True,cache_rate=1.0,num_workers=args.workers)
            test_loader = data.DataLoader(
                test_ds,
                batch_size=1,
                shuffle=False,
                num_workers=args.workers,
                sampler=None,
                pin_memory=False,
                persistent_workers=True,
            )
            return test_loader


        else:
            #if use cache dataset
            if args.use_normal_dataset:
                train_ds = data.Dataset(data=train_list, transform=pretrain_train_transforms)
                valid_ds = data.Dataset(data=valid_list, transform=pretrain_val_transforms)
                
            else:
                # train_ds = data.SmartCacheDataset(data=datalist,transform=train_transform,
                #     replace_rate=1,cache_num=100,progress=True)   # as_contiguous =True
                train_ds = data.CacheDataset(data=train_list,transform=pretrain_train_transforms,
                        cache_num=50,progress=True,cache_rate=0.1,num_workers=args.workers)
                valid_ds = data.CacheDataset(data=valid_list,transform=pretrain_val_transforms,
                        cache_num=10,progress=True,cache_rate=1.0,num_workers=args.workers)
            
            train_sampler = Sampler(train_ds) if args.distributed else None
            valid_sampler = Sampler(valid_ds) if args.distributed else None
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
                batch_size = 1,
                shuffle = False,
                num_workers= args.workers,
                sampler = valid_sampler,
                pin_memory = False,
                persistent_workers=True
            )
            loader = [train_loader,valid_loader]

            return loader
        