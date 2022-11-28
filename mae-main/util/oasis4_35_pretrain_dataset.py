from random import shuffle
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import os
class GlobalConfig():
    oasis4_35_root_dir = '/data2/zhanghao/data/oasis_4_35_npy/'
    seed = 55

class oasis4_35Dataset(Dataset):
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
    x = os.listdir(config.oasis4_35_root_dir)

    for i in range(len(x)):
        path = os.path.join(config.oasis4_35_root_dir, x[i])
        train_paths.append(path)

    return train_paths



def getDataloader_all(train_paths,valid_paths=None,test_paths=None,B1=1,B2=1,B3=1,pin:bool = True):
    train_dataset = oasis4_35Dataset(train_paths)
    train_loader = DataLoader(
        train_dataset,
        batch_size=B1,  # 每次传入1个nii文件
        # num_workers=num_workers,
        pin_memory=pin,
        shuffle=True
    )
    if valid_paths!=None:
        valid_dataset = oasis4_35Dataset(valid_paths)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=B2,  # 每次传入一个nii文件，
            # num_workers=num_workers,
            pin_memory=pin,
            shuffle = False
        )
        test_dataset = oasis4_35Dataset(test_paths)
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





