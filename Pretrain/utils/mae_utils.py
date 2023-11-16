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
        #set  num_replicas  ==  number of processes
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()

        #set  rank == current rank
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

        # build your dict list, like:[{image:image1_path,label:label1_path},{image:image2_path,label:label2_path}] ,
        # so make your train_list,valid_list,test_list.
        

        #if use cache dataset

        if args.use_normal_dataset:
            train_ds = data.Dataset(data=train_list, transform=pretrain_train_transforms)
            valid_ds = data.Dataset(data=valid_list, transform=pretrain_train_transforms)
            # test_ds = data.Dataset(data=test_list, transform=train_transform)
        else:
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
