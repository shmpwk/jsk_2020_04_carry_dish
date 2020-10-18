#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
import os
import pickle
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge, CvBridgeError
from torch.utils.tensorboard import SummaryWriter

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if tensor > 0.5:
            rt = tensor - abs(torch.randn(tensor.size()) * self.std + self.mean)
        else: 
            rt = tensor + abs(torch.randn(tensor.size()) * self.std + self.mean)
        return rt

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class MyDataset(Dataset):
    def __init__(self):
        self.d_transformer= AddGaussianNoise(0., 0.01)
        self.datanum = 160
        judge_path = "Data/judge_data"
        self.judge_dataset = np.empty((0,1))
        judge_key = '.txt'
        for j_dir_name, j_sub_dirs, j_files in sorted(os.walk(judge_path)): 
            for jf in sorted(j_files):
                if judge_key == jf[-len(judge_key):]:
                    f = open(os.path.join(j_dir_name, jf), 'r')
                    n = f.read()
                    self.judge_dataset = np.append(self.judge_dataset, int(n))

    def __len__(self):
        return self.datanum  #should be dataset size / batch size

    def __getitem__(self, idx):
        c = self.judge_dataset[idx]
        c = torch.from_numpy(np.array(c)).float()
        c = self.d_transformer(c)
        print(c)
        return c

class GraspSystem():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def imshow(self, img):
            img = img / 2 + 0.5  # [-1,1] を [0,1] へ戻す(正規化解除)
            npimg = img.numpy()  # torch.Tensor から numpy へ変換
            plt.imshow(np.transpose(npimg, (1, 2, 0)))  # チャンネルを最後に並び変える((C,X,Y) -> (X,Y,C))
            plt.show()  # 表示

    # load depth_image and grasp_pos_rot data
    def load_data(self, datasets):

        # Data loader (https://ohke.hateblo.jp/entry/2019/12/28/230000)
        train_dataloader = torch.utils.data.DataLoader(
            datasets, 
            batch_size=2, 
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        labels = next(iter(train_dataloader))
        # Show img
        return train_dataloader


if __name__ == '__main__':
    datasets = MyDataset()
    gs = GraspSystem()
    train_dataloader = gs.load_data(datasets)

