from random import shuffle
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import os
class GlobalConfig():
    adni_train_dir = '/data2/zhanghao/data/adni_affine_192_176_144/train_data/'
    adni_test_dir = '/data2/zhanghao/data/adni_affine_192_176_144/test_data/'
    seed = 55

class adniDataset(Dataset):
    def __init__(self,paths):#传入路径集合paths 
        self.paths = paths

    def __len__(self):
        return int(len(self.paths))
   

    def __getitem__(self, index):
        path = self.paths[index]

        image = np.load(path)

        image = np.pad(image,((0,0),(0,0),(16,16)),'constant')
                               
        image = torch.from_numpy(image).unsqueeze(0)

        return{
            # "Id": id,
            "image": image,
            # "domain":domain
        }

def getPathList():
    config = GlobalConfig()
    train_paths = []
    x = os.listdir(config.adni_train_dir)

    for i in range(len(x)):
        path = os.path.join(config.adni_train_dir, x[i])
        train_paths.append(path)

    test_paths = []
    y = os.listdir(config.adni_test_dir)

    for i in range(len(y)):
        path = os.path.join(config.adni_test_dir, x[i])
        test_paths.append(path)

    return train_paths,test_paths



def getDataloader_all(train_paths,test_paths=None,B1=1,B3=1,pin:bool = True):
    train_dataset = adniDataset(train_paths)
    train_loader = DataLoader(
        train_dataset,
        batch_size=B1,  # 每次传入1个nii文件
        # num_workers=num_workers,
        pin_memory=pin,
        shuffle=True
    )
    if test_paths!=None:
        test_dataset = adniDataset(test_paths)
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





