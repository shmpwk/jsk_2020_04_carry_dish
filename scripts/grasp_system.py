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

class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.transforms = transforms
        self.datanum = 10
        #self.imgfiles = sorted(glob('%s/*.png' % imgpath))
        #self.csvfiles = sorted(glob('%s/*.csv' % csvpath))

        """
        Args:
            dataset_path (str): example
                /home/Data
        
        """
        # depth_data_size(10) * (32*32)
        """
        # Can depth image (not PIL )use transform? : No, so you should convert from numpy.

        pilImg = Image.fromarray(self.depth_dataset)
        resize_transform = self.transforms.Resize((100, 100))
        display(resize_transform(pilImg))
        """
        
        # It was copied from scripts/depth_pickle_load.py
        """
        self.depth_dataset = np.empty((0,40000))

        for file in os.listdir(".ros/Data/gazebo_depth_image"):
            with open (".ros/Data/gazebo_depth_image/" + file, "rb") as f:
                ff = pickle.load(f)
                ff = np.array(ff).reshape((1, 40000))
                self.depth_dataset = np.append(self.depth_dataset, ff, axis=0)
        self.depth_dataset = self.depth_dataset.reshape((10, 1, 200, 200))
        """
        depth_path = "Data/depth_data"
        self.depth_dataset = np.empty((0,230400))
        depth_key = '.pkl'
        for d_dir_name, d_sub_dirs, d_files in os.walk(depth_path): 
            for df in d_files:
                if depth_key == df[-len(depth_key):]:
                    with open(os.path.join(d_dir_name, df), 'rb') as f:
                        ff = pickle.load(f)

                        WIDTH = 240
                        HEIGHT = 240 
                        bridge = CvBridge()
                        #try:
                        #    depth_image = bridge.imgmsg_to_cv2(ff, 'passthrough')
                        #except CvBridgeError, e:
                        #    rospy.logerr(e)
                        im = ff.reshape((480,640,3))
                        im_gray = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
                        depth_image = im_gray

                        h, w = depth_image.shape

                        x1 = (w / 2) - WIDTH
                        x2 = (w / 2) + WIDTH
                        y1 = (h / 2) - HEIGHT
                        y2 = (h / 2) + HEIGHT
                        depth_data = np.empty((0,230400))

                        for i in range(y1, y2):
                            for j in range(x1, x2):
                                if depth_image.item(i,j) == depth_image.item(i,j):
                                    depth_data = np.append(depth_data, depth_image.item(i,j))
                                            
                        #ims = depth_data.reshape((1, 480, 480))
                        #plt.imshow(np.transpose(ims[0, :, :]))
                        #plt.show()
                                    
                        depth_data = np.array(depth_data).reshape((1, 230400))
                        self.depth_dataset = np.append(self.depth_dataset, depth_data, axis=0)
        self.depth_dataset = self.depth_dataset.reshape((10, 1, 480, 480))
        print("Finished loading all depth data")
        
        # grasp point data size : 10 * 6(4)   
        self.grasp_dataset = np.empty((0,6))

        for file in os.listdir(".ros/Data/grasp_point"):
            with open (".ros/Data/grasp_point/" + file, "rb") as f:
                ff = pickle.load(f)
                ff = np.array(ff).reshape((1, 6))
                self.grasp_dataset = np.append(self.grasp_dataset, ff, axis=0)
        print("Finished loading grasp point")         
        # judge data size : 100 * 1

        """
        judge_path = "~/Data"
        self.judge_dataset = np.empty((0,1))
        judge_key = '.txt'
        for j_dir_name. j_sub_dirs, j_files in os.walk(judge_path): 
            for jf in j_files:
                if judge_key == jf[-len(judge_key):]:
                    np.append(self.judge_dataset, hf, axis=0)
        """

        judge_path = "Data/judge_data"
        self.judge_dataset = np.empty((0,1))
        judge_key = '.txt'
        for j_dir_name, j_sub_dirs, j_files in os.walk(judge_path): 
            for jf in j_files:
                if judge_key == jf[-len(judge_key):]:
                    f = open(os.path.join(j_dir_name, jf), 'r')
                    n = f.read()
                    self.judge_dataset = np.append(self.judge_dataset, int(n))
        print("Finished loading judge data")

        print(self.depth_dataset.shape)
        print(self.grasp_dataset.shape)
        print(self.judge_dataset.shape)

    def __len__(self):
        return self.datanum #should be dataset size / batch size

    def __getitem__(self, idx):
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
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        c = torch.from_numpy(np.array(c)).float()
        return x, y, c

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2) #入力チャンネル数は1, 出力チャンネル数は96 
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        #self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1 = nn.Linear(50176, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.fc4 = nn.Linear(10 + 6, 16)
        self.fc5 = nn.Linear(16, 1) # output is 1 dim scalar probability
        

    # depth encording without concate grasp point
    def forward(self, x, y):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        #depth_data =depth_data.view(depth_data.shape[0], -1)
        print("x", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        z = torch.cat((x, y), dim=1)
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        return z
   
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
            datasets, batch_size=2, shuffle=True,
            num_workers=2, drop_last=True
        )
        depth_data, grasp_point, labels = next(iter(train_dataloader))
        print(depth_data.size())  # torch.Size([10, 1, 480, 480])になっているか
        print(grasp_point.size())
        print(labels.size())
        return train_dataloader

    # make Net class model
    def make_model(self):
        self.model = Net()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

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
        #torch.save(self.model.state_dict(), PATH)
        # CPU save
        #torch.save(self.model.to('cpu').state_dict(), model_path)

    def train(self, train_dataloader, loop_num=10):
        for epoch in range(2):  # 訓練データを複数回(2周分)学習する
            running_loss = 0.0
            
            for i, data in enumerate(train_dataloader, 0):
                # ローダからデータを取得する; データは [inputs, labels] の形で取得される
                # イテレータを使用していないように見えますが for の内部で使用されています。
                depth_data, grasp_point, labels = data 
                
                # 勾配を0に初期化する(逆伝播に備える)
                self.optimizer.zero_grad()

                # 順伝播 + 逆伝播 + 最適化(訓練)
                outputs = self.model(depth_data, grasp_point)
                loss = self.criterion(outputs.view_as(labels), labels)
                loss.backward()
                self.optimizer.step()

                # 統計を表示する
                running_loss += loss.item()
                if i % 2 == 1:    # 2000 ミニバッチ毎に表示する
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2))
                    running_loss = 0.0
        print('Finished Training')

    def test(self):
        pass

if __name__ == '__main__':
    # parse
    train_flag = True #int(arg.train)
    gs = GraspSystem()
    loop_num = 1

    # train model or load model
    if train_flag:
        datasets = MyDataset()

        train_dataloader = gs.load_data(datasets)
        gs.make_model()
        gs.train(train_dataloader, loop_num)
        gs.save_model()
    else:
        pass
    # test
    #gs.start()


