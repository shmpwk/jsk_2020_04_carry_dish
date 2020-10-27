#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Grasp System
# This file is for 
# - init arg parser and set parse
# - make network
# - train model or load model
# - do test
# - select simulation or real robot
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from __future__ import unicode_literals
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
import pickle
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge, CvBridgeError
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from PIL import Image

class MyTransform:
    def __init__(self, hoge):
        self.hoge = fuga
    
    def __call__(self, x):
        hoge = hoge(self.hoge)
        return hoge

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class InflateGraspPoint(object):
    def __init__(self, grasp_point):
        self.aug_grasp_point = np.zeros((10, 4))
        self.x = grasp_point[[0]]
        self.y = grasp_point[[1]]
        self.z = grasp_point[[2]]
        self.theta = grasp_point[[3]]

    def calc(self):
        r = math.sqrt(self.x**2 + self.y**2)
        rad = math.atan2(self.y, self.x)
        rad_delta = math.pi/5
        X = np.zeros(10)
        Y = np.zeros(10)
        for i in range(10):
            X[i] = r * math.cos(rad + rad_delta*i)
            Y[i] = r * math.sin(rad + rad_delta*i)
            self.aug_grasp_point[i, :] = np.array((X[i], Y[i], self.z, self.theta))
        return self.aug_grasp_point

class NormalizedAddGaussianNoise(object):
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
        """
        self.depth_transform = depth_transform
        self.grasp_point_transform = grasp_point_transform
        self.judge_transform = judge_transform
        """
        self.dd_transformer = transforms.Compose([transforms.ToPILImage(), transforms.RandomAffine(degrees=0, translate=(0.001, 0.001)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), AddGaussianNoise(0., 0.001)])
        self.d_transformer= AddGaussianNoise(0., 0.01)
        self.j_transformer= NormalizedAddGaussianNoise(0., 0.01)
        self.datanum = 1600 / 4
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
        self.depth_dataset = np.empty((0,16384)) #230400))
        self.gray_dataset = np.empty((0,16384)) #230400))
        depth_key = 'heightmap_image.pkl'
        color_key = 'extract_color_image.pkl'
        t_cnt = 0
        tmp_cnt = 0
        for d_dir_name, d_sub_dirs, d_files in sorted(os.walk(depth_path)): 
            for df in sorted(d_files):
                if color_key == df[-len(color_key):]:
                    with open(os.path.join(d_dir_name, df), 'rb') as f:
                        fff = pickle.load(f)
                        color_image = fff
                        WIDTH = 64#240
                        HEIGHT = 64#240
                        """
                        bridge = CvBridge()
                        try:
                            color_image = bridge.imgmsg_to_cv2(ff, 'passthrough')
                        except CvBridgeError, e:
                            rospy.logerr(e)
                        """
                        im = fff.reshape((480,640,3))
                        pil_im = Image.fromarray(np.uint8(im))
                        pil_im = pil_im.resize((129, 172))
                        im = np.asarray(pil_im)
                        im_gray = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
                        h, w = im_gray.shape
                        x1 = (w / 2) - WIDTH
                        x2 = (w / 2) + WIDTH
                        y1 = (h / 2) - HEIGHT
                        y2 = (h / 2) + HEIGHT
                        gray_data = np.empty((0,16384))

                        for i in range(y1, y2):
                            for j in range(x1, x2):
                                if im_gray.item(i,j) == im_gray.item(i,j):
                                    gray_data = np.append(gray_data, im_gray.item(i,j))
                                else:
                                    gray_data = np.append(gray_data, 0)
                                            
                        gray_data = np.array(gray_data).reshape((1, 16384)) #230400))
                        #self.depth_dataset = np.append(self.depth_dataset, depth_data, axis=0)
                        if (t_cnt == 1 or t_cnt == 3):
                            self.gray_dataset = np.append(self.gray_dataset, np.tile(gray_data, (500, 1)).reshape(500, 16384), axis=0)
                        else:
                            self.gray_dataset = np.append(self.gray_dataset, np.tile(gray_data, (200, 1)).reshape(200, 16384), axis=0)
                        t_cnt += 1

                if depth_key == df[-len(depth_key):]:
                    with open(os.path.join(d_dir_name, df), 'rb') as f:
                        ff = pickle.load(f)
                        depth_image = ff
                        WIDTH = 64#240
                        HEIGHT = 64#240
                        """
                        bridge = CvBridge()
                        try:
                            depth_image = bridge.imgmsg_to_cv2(ff, 'passthrough')
                        except CvBridgeError, e:
                            rospy.logerr(e)
                        """
                        """
                        im = ff.reshape((480,640,3))
                        im_gray = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
                        depth_image = im_gray
                        """
                        im = ff.reshape((128,128,2))
                        depth_image = im[:, :, 0]
                        h, w = depth_image.shape
                        x1 = (w / 2) - WIDTH
                        x2 = (w / 2) + WIDTH
                        y1 = (h / 2) - HEIGHT
                        y2 = (h / 2) + HEIGHT
                        depth_data = np.empty((0,16384)) #230400))

                        for i in range(y1, y2):
                            for j in range(x1, x2):
                                if depth_image.item(i,j) == depth_image.item(i,j):
                                    depth_data = np.append(depth_data, depth_image.item(i,j))
                                else:
                                    depth_data = np.append(depth_data, 0)
                        depth_data = np.array(depth_data).reshape((1, 16384)) #230400))
                        #self.depth_dataset = np.append(self.depth_dataset, depth_data, axis=0)
                        if (tmp_cnt == 1 or tmp_cnt == 3):
                            self.depth_dataset = np.append(self.depth_dataset, np.tile(depth_data, (500, 1)).reshape(500, 16384), axis=0)
                        else:
                            self.depth_dataset = np.append(self.depth_dataset, np.tile(depth_data, (200, 1)).reshape(200, 16384), axis=0)

                        tmp_cnt += 1
        self.depth_dataset = self.depth_dataset.reshape((1600, 1, 128, 128))
        self.gray_dataset = self.gray_dataset.reshape((1600, 1, 128, 128))

        self.gray_depth_dataset = np.concatenate([self.depth_dataset, self.gray_dataset], 1)
        print("gray depth dataset", self.gray_depth_dataset.shape)
        print("Finished loading all depth data")
        
        # grasp point data size : 10 * 6(4)   
        self.grasp_dataset = np.empty((0,4))

        """for file in os.listdir(".ros/Data/grasp_point"):
            with open (".ros/Data/grasp_point/" + file, "rb") as f:
                ff = pickle.load(f)
                ff = np.array(ff).reshape((1, 4))
                self.grasp_dataset = np.append(self.grasp_dataset, ff, axis=0)
        
        """
        grasp_path = ".ros/Data/grasp_point"
        g_key = '.pkl'
        for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
            for gf in sorted(g_files):
                if g_key == gf[-len(g_key):]:
                    with open(os.path.join(g_dir_name, gf), 'rb') as f:
                        ff = pickle.load(f)
                        #ff = np.array(ff).reshape((1, 4))
                        ff = np.array(ff).reshape((4))
                        fff = InflateGraspPoint(ff)
                        #fff = np.append(fff, np.tile(fff, (10, 1)).reshape(10, 4), axis=0)
                        fff = np.array(fff.calc())
                        self.grasp_dataset = np.append(self.grasp_dataset, fff, axis=0)
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
        for j_dir_name, j_sub_dirs, j_files in sorted(os.walk(judge_path)): 
            for jf in sorted(j_files):
                if judge_key == jf[-len(judge_key):]:
                    f = open(os.path.join(j_dir_name, jf), 'r')
                    n = int(f.read())
                    n = np.atleast_2d(n)
                    nn = np.append(n, np.tile(n, (9, 1)).reshape(9, 1), axis=0)
                    self.judge_dataset = np.append(self.judge_dataset, nn)
        print("Finished loading judge data")

        print(self.gray_depth_dataset.shape)
        print(self.grasp_dataset.shape)
        print(self.judge_dataset.shape)

    def __len__(self):
        return self.datanum #should be dataset size / batch size

    def __getitem__(self, idx):
        x = self.gray_depth_dataset[idx]
        y = self.grasp_dataset[idx]
        c = self.judge_dataset[idx]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        c = torch.from_numpy(np.array(c)).float()
        x = self.dd_transformer(x) 
        y = self.d_transformer(y) 
        c = self.j_transformer(c) 
        
        return x, y, c

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        """
        This imitates alexnet. 
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2) #入力チャンネル数は1, 出力チャンネル数は96 
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        #self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1 = nn.Linear(50176, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.fc4 = nn.Linear(10 + 4, 14)
        self.fc5 = nn.Linear(14, 1) # output is 1 dim scalar probability
        """

        # dynamics-net (icra2019の紐とか柔軟物を操るやつ) by Mr. Kawaharazuka
        self.conv1 = nn.Conv2d(2, 4, 3, 2, 1)
        self.cbn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, 2, 1)
        self.cbn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3, 2, 1)
        self.cbn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, 2, 1)
        self.cbn4 = nn.BatchNorm2d(32)
        #self.conv5 = nn.Conv2d(32, 64, 3, 2, 1)
        #self.cbn5 = nn.BatchNorm2d(64)
        #self.fc1 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8 + 4, 12)
        self.fc5 = nn.Linear(12, 1) # output is 1 dim scalar probability

    # depth encording without concate grasp point
    def forward(self, x, y):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.cbn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.cbn2(x)
        x = F.relu(self.conv3(x))
        x = self.cbn3(x)
        x = F.relu(self.conv4(x))
        x = self.cbn4(x)
        #x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        #x = self.cbn5(x)
        x = x.view(-1, self.num_flat_features(x))
        #depth_data =depth_data.view(depth_data.shape[0], -1)
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
            batch_size=4, 
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        depth_data, grasp_point, labels = next(iter(train_dataloader))
        # Show img
        img = torchvision.utils.make_grid(depth_data)
        img = img / 2 + 0.5  # [-1,1] を [0,1] へ戻す(正規化解除)
        npimg = img.numpy()  # torch.Tensor から numpy へ変換
        ims = npimg#.reshape((1, 480, 480))
        plt.imshow(np.transpose(ims[1, :, :])) # チャンネルを最後に並び変える((C,X,Y) -> (X,Y,C))
        plt.show() #表示
        # Show label
        print(' '.join('%5s' % labels[j] for j in range(2)))
        depth_data = depth_data.to(self.device)
        grasp_point = grasp_point.to(self.device)
        labels = labels.to(self.device)
        print("depth size", depth_data.size())  # torch.Size([10, 1, 480, 480])になっているか
        print("point size", grasp_point.size())
        print("judge size", labels.size())
        return train_dataloader

    # make Net class model
    def make_model(self):
        self.model = Net()
        self.model = self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        #self.train_optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        #self.test_optimizer = optim.SGD(self.model
        summary(self.model, [(2, 128, 128), (4,)])

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
        ## Save only parameter
        #torch.save(self.model.state_dict(), model_path)
        ## Save whole model
        torch.save(self.model, model_path)
        # CPU save
        #torch.save(self.model.to('cpu').state_dict(), model_path)
        print("Finished Saving model")

    def train(self, train_dataloader, loop_num):
        tensorboard_cnt = 0
        for epoch in range(loop_num):  # 訓練データを複数回(2周分)学習する
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                # ローダからデータを取得する; データは [inputs, labels] の形で取得される
                # イテレータを使用していないように見えますが for の内部で使用されています。
                depth_data, grasp_point, labels = data 
                depth_data = depth_data.to(self.device)
                grasp_point = grasp_point.to(self.device)
                labels = labels.to(self.device)
                # 勾配を0に初期化する(逆伝播に備える)
                self.train_optimizer.zero_grad()

                # 順伝播 + 逆伝播 + 最適化(訓練)
                outputs = self.model(depth_data, grasp_point)
                loss = self.criterion(outputs.view_as(labels), labels)
                loss.backward()
                self.train_optimizer.step()
                aa = torch.max(outputs.data, labels)
                print("a", aa)
                #print("b", b)

                # 統計を表示する
                writer = SummaryWriter(log_dir="./Data/loss")
                running_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), tensorboard_cnt) #(epoch + 1) * i)
                if i % 100 == 99:    # 2 ミニバッチ毎に表示する
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
                tensorboard_cnt += 1
        print('Finished Training')
        writer.flush()

    def test(self, test_loader):
        for data in testloader:
            depth_data, grasp_point, labels = data
            outputs = self.model(depth_data, grasp_point)
            # lossのgrasp_point偏微分に対してoptimaizationする．
            depth_data.requires_grad(False)
            grasp_point.requires_grad(True)
            loss = self.criterion(outputs.view_as(labels), labels)
            loss.backward()
            self.train_optimizer.step()
            # 最適化されたuを元に把持を実行し、その結果を予測と比較する
       
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
    # test
    #gs.start()

