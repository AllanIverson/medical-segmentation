import random
import time
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator
import numpy as np
import torch
import os
class GlobalConfig():
    cc359_root_dir = '/data/zhanghao/skull_project/cc359_data_224/'
    cc359_train_root_dir = '/data/zhanghao/skull_project/cc359_data_224/train_data_224_2D'
    cc359_valid_root_dir = '/data/zhanghao/skull_project/cc359_data_224/valid_data_224_2D'
    cc359_test_root_dir = '/data/zhanghao/skull_project/cc359_data_224/test_data_224_2D'
    oasis_root_dir = '/data/zhanghao/skull_project/oasis_data_224/'
    oasis_train_root_dir = '/data/zhanghao/skull_project/oasis_data_224/train_2D_224'
    oasis_valid_root_dir = '/data/zhanghao/skull_project/oasis_data_224/valid_2D_224'
    oasis_test_root_dir = '/data/zhanghao/skull_project/oasis_data_224/test_2D_224'
    seed = 55


class Oasis_cc359_Dataset(Dataset):
    def __init__(self,paths):#传入路径集合paths 
        self.paths = paths

    def __len__(self):
        return len(self.paths)
        

    def __getitem__(self, index):
        path = self.paths[index]
        
       
        image = np.load(path)['image']

        # mask = np.load(path)['mask']  
        # print(time.time()-time_begin) 
        
         
        # domain = np.load(path)['domain']#改成torch.tensor64
        # domain = torch.from_numpy(domain)
        # domain = domain.reshape(1,) 
        # domain = domain.squeeze()                

        # image = torch.from_numpy(image).unsqueeze(0)
        # mask = torch.from_numpy(mask).unsqueeze(0)

        
        image = image.reshape(1,224,224)
        # mask = mask.reshape(1,224,224)


        
        
        return{
            # "Id": id,
            "image": image,
            # "mask": mask,
            # "domain":domain
        }

def getPathList():
    config = GlobalConfig()
    train_paths = []
    valid_paths = []
    test_paths = []

    x = os.listdir(config.oasis_train_root_dir) 
    y = os.listdir(config.oasis_valid_root_dir)
    z = os.listdir(config.oasis_test_root_dir)
    X = os.listdir(config.cc359_train_root_dir)
    Y = os.listdir(config.cc359_valid_root_dir)
    Z = os.listdir(config.cc359_test_root_dir)

    i = 0
    for i in range(len(x)):
        path = os.path.join(config.oasis_train_root_dir, x[i])
        train_paths.append(path)
    i = 0
    for i in range(len(y)):
        path = os.path.join(config.oasis_valid_root_dir, y[i])
        train_paths.append(path)
    i = 0
    for i in range(len(z)):
        path = os.path.join(config.oasis_test_root_dir, z[i])
        train_paths.append(path)
    i = 0
    for i in range(len(X)):
        path = os.path.join(config.cc359_train_root_dir, X[i])
        train_paths.append(path)
    i = 0
    for i in range(len(Y)):
        path = os.path.join(config.cc359_valid_root_dir, Y[i])
        valid_paths.append(path)
    i = 0
    for i in range(len(Z)):
        path = os.path.join(config.cc359_test_root_dir, Z[i])
        test_paths.append(path)
    print("x,y,z,X,Y,Z = ",len(x),len(y),len(z),len(X),len(Y),len(Z))

    random.seed(config.seed)
    random.shuffle(train_paths)
    # random.shuffle(valid_paths)
    # random.shuffle(test_paths)
    return train_paths,valid_paths,test_paths

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def getDataloader(train_paths,valid_paths,test_paths,B1=1,B2=1,B3=1,pin:bool = True):
    train_dataset = Oasis_cc359_Dataset(train_paths)
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=B1,  # 每次传入1个nii文件
        # num_workers=num_workers,
        pin_memory=pin,
        shuffle=True
    )
    valid_dataset = Oasis_cc359_Dataset(valid_paths)
    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=B2,  # 每次传入一个nii文件，
        # num_workers=num_workers,
        pin_memory=pin,
        shuffle = False
    )
    test_dataset = Oasis_cc359_Dataset(test_paths)
    test_loader = DataLoaderX(
        test_dataset,
        batch_size=B3,  # 每次传入一个nii文件，
        # num_workers=num_workers,
        pin_memory=pin,
        shuffle=False
    )
    return train_loader,valid_loader,test_loader



