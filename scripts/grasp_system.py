#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Grasp System
# This file is for 
# - init arg parser and set parse
# - make network
# - train model or load model
# - do test
# - select simulation or real robot

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import os


class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.datanum = 100
        #self.imgfiles = sorted(glob('%s/*.png' % imgpath))
        #self.csvfiles = sorted(glob('%s/*.csv' % csvpath))

        """
        Args:
            dataset_path (str): example
                /home/Data
        
        """
        # depth_data_size(100) * (32*32)
        depth_dataset_path = "~/my_ws/src"
        self.depth_dataset = np.empty((0,3))
        key = '.pkl'
        for dir_name, sub_dirs, files in os.walk(depth_dataset_path):
            for f in files:
                if key == f[-len(key):]:
                    np.append(self.dapth_dataset, f, axis = 0) 
                    #os.path.join(dir_name, f[:-len(key)]))
        
        """
        data=np.loadtxt("data.csv",  # 読み込みたいファイルのパス
                  delimiter=",",    # ファイルの区切り文字
                  skiprows=0,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                  #usecols=(1,2,3,4,5,6,7,8,9,10) # 読み込みたい列番号。
                )
        """

        # grasp point data size : 100 * 6(4)   
        grasp_point_path = "~/"
        self.grasp_datset = np.empty((0,3))
        grasp_key = 'point.pkl'
        for g_dir_name, g_sub_dirs, g_files in os.walk(grasp_point_path):
            for gf in g_files:
                if grasp_key == gf[-len(grasp_key):]:
                    np.append(self.grasp_dataset, gf, axis=0)

        # data=np.genfromtxt("sample_writer.csv", filling_values=0) #nanを0に置き換える。data.shape(480,)

        # judge data size : 100 * 1

        judge_path = "~/"
        self.judge_dataset = np.empty((0,3))
        judge_key = 'judge.pkl'
        for j_dir_name. j_sub_dirs, j_files in os.walk(judge_path): 
            for jf in j_files:
                if judge_key == jf[-len(judge_key):]:
                    np.append(self.judge_dataset, hf, axis=0)

    def __len__(self):
        return self.datanum #should be dataset size / batch size

    def __getitem__(self, idx):
        print(idx)
        x = self.depth_dataset[idx]
        y = self.grasp_dataset[idx]
        c = self.judge_dataset[idx]
        """
        y = pd.read_csv(self.csvfiles[idx], header=None),
        yy = np.array(yy, dtype=np.float32)[0]
        x = self.transform(x) if self.transform else x
        c = yy[1:]
        y = yy[:1]
        """
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        c = torch.from_numpy(c)
        return x, y, c

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2) #入力チャンネル数は1, 出力チャンネル数は96 
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1) # output is 1 dim scalar probability

    # depth encording without concate grasp point
    def forward(self, x, y):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
   
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class GraspSystem():
    def __init__(self):
        pass

    # load depth_image and grasp_pos_rot data
    def load_data(self, datasets):

        # Data loader (https://ohke.hateblo.jp/entry/2019/12/28/230000)
        train_dataloader = torch.utils.data.DataLoader(
            datasets, batch_size=16, shuffle=True,
            num_workers=2, drop_last=True
        )

        depth_data, grasp_point, labels = next(iter(train_dataloader))
        print(depth_data.size())  # torch.Size([16, 3, 224, 224])になっているかな
        print(grasp_point.size())
        print(labels.size())
        return train_dataloader


    # make Net class model
    def make_model(self):
        self.model = Net()
        self.criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    def get_batch_train():
        pass


    def get_test():
        pass

    # load traind Network model
    def load_model():
        model_path = 'model.pth'
        # learn GPU, load GPU
        self.model.load_state_dict(torch.load(model_path))
        # learn CPU, load GPU
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))



    def save_model(self):
        model_path = 'model.pth'
        # GPU save
        torch.save(self.model.state_dict(), PATH)
        # CPU save
        #torch.save(self.model.to('cpu').state_dict(), model_path)

    def train(self, train_dataloader, loop_num=10):
        for epoch in range(2):  # 訓練データを複数回(2周分)学習する
            running_loss = 0.0
            
            for i, data in enumerate(trainloader, 0):
                # ローダからデータを取得する; データは [inputs, labels] の形で取得される
                # イテレータを使用していないように見えますが for の内部で使用されています。
                depth_data, grasp_point, labelsb = data 
                


                # 勾配を0に初期化する(逆伝播に備える)
                optimizer.zero_grad()

                # 順伝播 + 逆伝播 + 最適化(訓練)
                outputs = net(depth_data, grasp_point)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 統計を表示する
                running_loss += loss.item()
                if i % 2000 == 1999:    # 2000 ミニバッチ毎に表示する
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training')

    def test(self):

        pass


if __name__ == '__main__':
    # parse
    train_flag = True #int(arg.train)

    gs = GraspSystem()

    # train model or load model
    if train_flag:
        datasets = MyDataset()

        train_dataloader = gs.load_data(datasets)
        gs.tain(train_dataloader, loop_num)
        gs.save_model()

    else:
        pass

    # test
    gs.start()


