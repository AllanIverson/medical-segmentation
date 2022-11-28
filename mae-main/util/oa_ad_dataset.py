from random import shuffle
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import os
class GlobalConfig():
    adni_train_dir = '/data2/zhanghao/adni/train_npy/train_data/'
    oasis4_35_train_dir = '/data2/zhanghao/oasis4_35/train_npy/'

    seed = 55

class oa_ad_Dataset(Dataset):
    def __init__(self,paths):#传入路径集合paths 
        self.paths = paths

    def __len__(self):

        return int(len(self.paths))

   

    def __getitem__(self, index):
        path = self.paths[index]

        image = np.load(path)
        image = torch.from_numpy(image)
        image = image.float()
        image = torch.transpose(image,0,1)
        image = image.unsqueeze(0)


        return{
            "image": image,
        }

def getPathList():
    config = GlobalConfig()
    train_paths = []
    x = os.listdir(config.oasis4_35_train_dir)

    for i in range(len(x)):
        path = os.path.join(config.oasis4_35_train_dir, x[i])
        train_paths.append(path)
    

    y = os.listdir(config.adni_train_dir)
    for i in range(len(y)):
        path = os.path.join(config.adni_train_dir, y[i])
        train_paths.append(path)


    return train_paths

def getDataloader_all(train_paths,test_paths=None,B1=1,B3=1,pin:bool = True):
    train_dataset = oa_ad_Dataset(train_paths)
    train_loader = DataLoader(
        train_dataset,
        batch_size=B1,  # 每次传入1个nii文件
        # num_workers=num_workers,
        pin_memory=pin,
        shuffle=True
    )
    if test_paths!=None:
        test_dataset = oa_ad_Dataset(test_paths)
        test_loader = DataLoader(
            test_dataset,
            batch_size=B3,  # 每次传入一个nii文件，
            # num_workers=num_workers,
            pin_memory=pin,
            shuffle=False
        )
    if test_paths==None:
        return train_loader
    else:
        return train_loader,test_loader


