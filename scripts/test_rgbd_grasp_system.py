import numpy as np
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
import os


if __name__ == '__main__':
    # parse
    train_flag = True #int(arg.train)
    gs = GraspSystem()
    loop_num = 100

    # train model or load model
    if train_flag:
        datasets = MyDataset()
        train_dataloader = gs.load_data(datasets)
        gs.make_model()
        gs.train(train_dataloader, loop_num=2)
        gs.save_model()
    else:
        gs.load_model()
        gs.test(test_loader)

