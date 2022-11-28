from random import shuffle
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import os
class GlobalConfig():
    cc359_root_dir = '/data/zhanghao/skull_project/cc359_data_224/'
    cc359_train_root_dir = '/data/zhanghao/skull_project/cc359_data_224/train_data_224'
    cc359_valid_root_dir = '/data/zhanghao/skull_project/cc359_data_224/valid_data_224'
    cc359_test_root_dir = '/data/zhanghao/skull_project/cc359_data_224/test_data_224'
    seed = 55


class cc359Dataset(Dataset):
    def __init__(self,paths):#传入路径集合paths 
        self.paths = paths

    def __len__(self):

        return len(self.paths)

   

    def __getitem__(self, index):
        path = self.paths[index]

        image = np.load(path)['image']
        mask = np.load(path)['mask']
        
                          

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)


        return{
            # "Id": id,
            "image": image,
            "mask": mask,
            # "domain":domain
        }

class cc359Dataset_all(Dataset):
    def __init__(self,paths):#传入路径集合paths 
        self.paths = paths

    def __len__(self):

        return int(len(self.paths))

   

    def __getitem__(self, index):
        path = self.paths[index]

        image = np.load(path)['image']
        mask = np.load(path)['mask']
        
                          

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)


        return{
            # "Id": id,
            "image": image,
            "mask": mask,
            # "domain":domain
        }

class cc359Dataset_only40(Dataset):
    def __init__(self,paths):#传入路径集合paths 
        self.paths = paths

    def __len__(self):
        return int(40)

    def __getitem__(self, index):
        path = self.paths[index]

        image = np.load(path)['image']
        mask = np.load(path)['mask']
                               
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return{
            # "Id": id,
            "image": image,
            "mask": mask,
            # "domain":domain
        }

class cc359Dataset005(Dataset):
    def __init__(self,paths):#传入路径集合paths 
        self.paths = paths

    def __len__(self):

        return int(len(self.paths)*0.05)

   

    def __getitem__(self, index):
        path = self.paths[index]

        image = np.load(path)['image']
        mask = np.load(path)['mask']
        
                          

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)


        return{
            # "Id": id,
            "image": image,
            "mask": mask,
            # "domain":domain
        }

def getPathList():
    config = GlobalConfig()
    train_paths = []
    valid_paths = []
    test_paths = []
    x = os.listdir(config.cc359_train_root_dir)
    y = os.listdir(config.cc359_valid_root_dir)
    z = os.listdir(config.cc359_test_root_dir)
    for i in range(len(x)):
        path = os.path.join(config.cc359_train_root_dir, x[i])
        train_paths.append(path)
    i = 0
    for i in range(len(y)):
        path = os.path.join(config.cc359_valid_root_dir, y[i])
        valid_paths.append(path)
    i = 0
    for i in range(len(z)):
        path = os.path.join(config.cc359_test_root_dir, z[i])
        test_paths.append(path)
    return train_paths,valid_paths,test_paths

def getDataloader(train_paths,valid_paths,test_paths,B1=1,B2=1,B3=1,pin:bool = True):
    train_dataset = cc359Dataset(train_paths)
    train_loader = DataLoader(
        train_dataset,
        batch_size=B1,  # 每次传入1个nii文件
        # num_workers=num_workers,
        pin_memory=pin,
        shuffle=True
    )
    if valid_paths!=None:
        valid_dataset = cc359Dataset(valid_paths)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=B2,  # 每次传入一个nii文件，
            # num_workers=num_workers,
            pin_memory=pin,
            shuffle = False
        )
        test_dataset = cc359Dataset(test_paths)
        test_loader = DataLoader(
            test_dataset,
            batch_size=B3,  # 每次传入一个nii文件，
            # num_workers=num_workers,
            pin_memory=pin,
            shuffle=False
        )
    if valid_paths==None:
        return train_loader
    else:
        return train_loader,valid_loader,test_loader


def getDataloader_all(train_paths,valid_paths=None,test_paths=None,B1=1,B2=1,B3=1,pin:bool = True):
    train_dataset = cc359Dataset_all(train_paths)
    train_loader = DataLoader(
        train_dataset,
        batch_size=B1,  # 每次传入1个nii文件
        # num_workers=num_workers,
        pin_memory=pin,
        shuffle=True
    )
    if valid_paths!=None:
        valid_dataset = cc359Dataset(valid_paths)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=B2,  # 每次传入一个nii文件，
            # num_workers=num_workers,
            pin_memory=pin,
            shuffle = False
        )
        test_dataset = cc359Dataset(test_paths)
        test_loader = DataLoader(
            test_dataset,
            batch_size=B3,  # 每次传入一个nii文件，
            # num_workers=num_workers,
            pin_memory=pin,
            shuffle=False
        )
    if valid_paths==None:
        return train_loader
    else:
        return train_loader,valid_loader,test_loader

def getDataloader_only40(train_paths,valid_paths=None,test_paths=None,B1=1,B2=1,B3=1,pin:bool = True):
    train_dataset = cc359Dataset_only40(train_paths)
    train_loader = DataLoader(
        train_dataset,
        batch_size=B1,  # 每次传入1个nii文件
        # num_workers=num_workers,
        pin_memory=pin,
        shuffle=True
    )
    if valid_paths!=None:
        valid_dataset = cc359Dataset(valid_paths)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=B2,  # 每次传入一个nii文件，
            # num_workers=num_workers,
            pin_memory=pin,
            shuffle = False
        )
        test_dataset = cc359Dataset(test_paths)
        test_loader = DataLoader(
            test_dataset,
            batch_size=B3,  # 每次传入一个nii文件，
            # num_workers=num_workers,
            pin_memory=pin,
            shuffle=False
        )

    if valid_paths==None:
        return train_loader
    else:
        return train_loader,valid_loader,test_loader


def getDataloader005(train_paths,valid_paths=None,test_paths=None,B1=1,B2=1,B3=1,pin:bool = True):
    train_dataset = cc359Dataset005(train_paths)
    train_loader = DataLoader(
        train_dataset,
        batch_size=B1,  # 每次传入1个nii文件
        # num_workers=num_workers,
        pin_memory=pin,
        shuffle=True
    )
    if valid_paths!=None:
        valid_dataset = cc359Dataset005(valid_paths)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=B2,  # 每次传入一个nii文件，
            # num_workers=num_workers,
            pin_memory=pin,
            shuffle = False
        )
        test_dataset = cc359Dataset005(test_paths)
        test_loader = DataLoader(
            test_dataset,
            batch_size=B3,  # 每次传入一个nii文件，
            # num_workers=num_workers,
            pin_memory=pin,
            shuffle=False
        )
    if valid_paths==None:
        return train_loader
    else:
        return train_loader,valid_loader,test_loader

