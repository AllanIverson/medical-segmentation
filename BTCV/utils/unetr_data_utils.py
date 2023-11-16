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


#random sample from dataset for distributed ->  then be a new dataset(with random smple)
class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        #set  num_replicas 
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        
        #set  rank == current rank
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
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=2,
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


    fintune_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], reader='ITKReader'),
            transforms.AddChanneld(keys=["image","label"]),
            transforms.Orientationd(keys=["image","label"], axcodes="RSP"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=["bilinear","nearest"]
            ),
            transforms.ScaleIntensityRangePercentilesd(keys="image",lower=5,upper=95,b_min=args.b_min,b_max=args.b_max,clip=True),
            # transforms.RandRotated(keys=["image","label"],range_x=15,range_y=15,range_z=15,prob=0.3,mode=["bilinear","nearest"]),
            transforms.CropForegroundd(keys=["image","label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.SpatialPadd(keys=["image","label"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.CenterSpatialCropd(
                keys=["image","label"],
                roi_size = [args.roi_x, args.roi_y, args.roi_z],
            ),
            transforms.RandSpatialCropSamplesd(
                keys=["image","label"],
                roi_size = [args.finetune_size,args.finetune_size,args.finetune_size],
                random_size = False,
                num_samples  = args.sw_batch_size
            ),
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image","label"]),
        ]
    )
    

    if args.out_channels == 5:

        '''
        dataset,the ".txt" need build by your self.
        '''
        train_list = []
        valid_list = []
        test_list = []
        oasis_w_4_train = '/Pretrain/utils/train/oasis_wtrain.txt'
        adni_w_4_train = '/Pretrain/utils/train/adni_w4train.txt'
        oasis_w_4_val = '/Pretrain/utils/val/oasis_wval.txt'
        adni_w_4_val = '/Pretrain/utils/val/adni_w4val.txt'
        oasis_w_4_test = '/Pretrain/utils/test/oasis_wtest.txt'
        adni_w_4_test = '/Pretrain/utils/test/adni_w4test_unseen.txt'
        oasis_wo_4_train = '/Pretrain/utils/train/oasis_wotrain.txt'
        oasis_wo_4_val = '/Pretrain/utils/val/oasis_woval.txt'
        oasis_wo_4_test = '/Pretrain/utils/test/oasis_wotest.txt'
        adni_wo_4_train = '/Pretrain/utils/train/adni_wotrain.txt'
        adni_wo_4_val = '/Pretrain/utils/val/adni_woval.txt'
        adni_wo_4_test = '/Pretrain/utils/test/adni_wo_test.txt'

        # if use all source domain data
        if args.use_all_source:
            # if souce domain is oasis
            if args.source == 'oasis':
                print('source = oasis')
                with open(oasis_w_4_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt >= args.num_source:
                            break
                        else:
                            train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                        cnt += 1
                with open(adni_w_4_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt >= args.add:
                            break
                        else:
                            train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                        cnt += 1 
                    print('cnt ==',cnt ,'args.add',args.add)
                with open(adni_w_4_val) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        valid_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                with open(adni_w_4_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        test_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                        
            elif args.source == 'adni_wo':
                print('4 class wo adni')
                with open(adni_wo_4_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        train_list.append({'image':content[i].split('\n')[0],'label':content[i].split('\n')[0]})
                with open(oasis_wo_4_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt == args.add:
                            break
                        train_list.append({'image':content[i].split('\n')[0],'label':content[i].split('\n')[0]})
                        cnt += 1 
                    print('cnt == ',cnt)
                with open(adni_wo_4_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        valid_list.append({'image':content[i].split('\n')[0],'label':content[i].split('\n')[0]})
                with open(adni_wo_4_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        test_list.append({'image':content[i].split('\n')[0],'label':content[i].split('\n')[0]})

            elif args.source == 'oasis_wo':
                print('4 class  wo oasis')
                with open(oasis_wo_4_train) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                with open(oasis_wo_4_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt == args.add:
                            break
                        train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                        cnt += 1 
                    print('cnt == ',cnt)
                with open(oasis_wo_4_val) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        valid_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                with open(oasis_wo_4_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        test_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
            elif args.source == None:
                # train adni base model
                print('train adni base model,4class')
                with open(adni_w_4_train) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                with open(adni_w_4_val) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        valid_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                with open(adni_w_4_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        test_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
        
    elif args.out_channels == 36:
        train_list = []
        valid_list = []
        test_list = []
        oasis_w_35_train = '/Pretrain/utils/train/oasis_wtrain.txt'
        adni_w_35_train = '/Pretrain/utils/train/adni_w35train.txt'
        oasis_w_35_val = '/Pretrain/utils/val/oasis_wval.txt'
        adni_w_35_val = '/Pretrain/utils/val/adni_w35val.txt'
        oasis_w_35_test = '/Pretrain/utils/test/oasis_wtest.txt'
        adni_w_35_test = '/Pretrain/utils/test/adni_w35test_unseen.txt'
        oasis_wo_35_train = '/Pretrain/utils/train/oasis_wotrain.txt'
        oasis_wo_35_val = '/Pretrain/utils/val/oasis_woval.txt'
        oasis_wo_35_test = '/Pretrain/utils/test/oasis_wotest.txt'
        if args.use_all_source:
            # if source domain is oasis
            if args.source == 'oasis':
                print('source = oasis')
                with open(oasis_w_35_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt >= args.num_source:
                            break
                        else:
                            train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                        cnt += 1
                with open(adni_w_35_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt >= args.add:
                            break
                        else:
                            train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                        cnt += 1 
                    print('cnt ==',cnt ,'args.add',args.add)
                with open(adni_w_35_val) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        valid_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                with open(adni_w_35_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        test_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})

            elif args.source == 'oasis_wo':
                with open(oasis_wo_35_train) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                with open(oasis_wo_35_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt == args.add:
                            break
                        train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                        cnt += 1 
                    print('cnt == ',cnt)
                with open(oasis_wo_35_val) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        valid_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                with open(oasis_wo_35_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        test_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
            
            elif args.source == 'adni_wo':
                with open(oasis_wo_35_train) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                with open(oasis_wo_35_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt == args.add:
                            break
                        train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                        cnt += 1 
                    print('cnt == ',cnt)
                with open(oasis_wo_35_val) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        valid_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                with open(oasis_wo_35_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        test_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})

            elif args.source == 'oasis_test':
                print('仅仅在oasis上测试')
                with open(oasis_w_35_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                valid_list = test_list = train_list
                valid_list = train_list = test_list[:10]

            else:
                # train adni base model
                print('train adni base model')
                with open(adni_w_35_train) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                with open(adni_w_35_val) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        valid_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
                with open(adni_w_35_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        test_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[2]})
    
    elif args.out_channels == 2:
        train_list = []
        valid_list = []
        test_list = []
        oasis_skull_train = '/data2/zhanghao/Pretrain/utils/train/oasis_skull_new.txt'
        oasis_skull_val ='/data2/zhanghao/Pretrain/utils/val/oasis_skull_new.txt'
        oasis_skull_test = '/data2/zhanghao/Pretrain/utils/test/oasis_skull_new.txt'
        cc359_skull_train='/data2/zhanghao/Pretrain/utils/train/cc359train.txt'
        cc359_skull_val='/data2/zhanghao/Pretrain/utils/val/cc359val.txt'
        cc359_skull_test='/data2/zhanghao/Pretrain/utils/test/cc359test.txt'

        if args.use_all_source:
            # if souce is oasis
            if args.source == 'oasis':
                print('skull stripping source = oasis')
                with open(oasis_skull_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt >= args.num_source:
                            break
                        else:
                            train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                        cnt += 1
                with open(cc359_skull_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt >= args.add:
                            break
                        else:
                            train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                        cnt += 1 
                    print('cnt ==',cnt ,'args.add',args.add)
                with open(cc359_skull_val) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        valid_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                with open(cc359_skull_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        test_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})

            elif args.source == 'cc359':
                print('skull stripping source = cc359')
                with open(cc359_skull_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt >= args.num_source:
                            break
                        else:
                            train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                        cnt += 1
                with open(oasis_skull_train) as f:
                    content = f.readlines()
                    cnt = 0
                    for i in range(len(content)):
                        if cnt == args.add:
                            break
                        train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                        cnt += 1 
                    print('cnt == ',cnt)
                with open(oasis_skull_val) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        valid_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                with open(oasis_skull_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        test_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
            else:
                # train oasis base model
                print('train oasis base model')
                with open(oasis_skull_train) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        train_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                with open(oasis_skull_val) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        valid_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})
                with open(oasis_skull_test) as f:
                    content = f.readlines()
                    for i in range(len(content)):
                        test_list.append({'image':content[i].split('\n')[0].split(' ')[0],'label':content[i].split('\n')[0].split(' ')[1]})


    # shuffle train set
    random.shuffle(train_list)
    print('len train',len(train_list),'len val',len(valid_list),'len test',len(test_list))
    if args.test_mode:
        if args.use_normal_dataset:
            test_ds = data.Dataset(data=test_list, transform=test_transforms)
        else:
            test_ds = data.CacheDataset(data=test_list,transform=test_transforms,
                    cache_num=10,progress=True,cache_rate=1.0,num_workers=args.workers)
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
            train_ds = data.Dataset(data=train_list, transform=fintune_transforms)
            valid_ds = data.Dataset(data=valid_list, transform=fintune_transforms)
            
        else:
            # train_ds = data.SmartCacheDataset(data=datalist,transform=train_transform,
            #     replace_rate=1,cache_num=100,progress=True)   # as_contiguous =True
            train_ds = data.CacheDataset(data=train_list,transform=fintune_transforms,
                    cache_num=30,progress=True,cache_rate=0.1,num_workers=args.workers)
            valid_ds = data.CacheDataset(data=valid_list,transform=fintune_transforms,
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
    
