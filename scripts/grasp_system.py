#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Grasp System
# This file is for 
# - init arg parser and set parse
# - make network
# - train model or load model
# - do test
# - select simulation or real robot

import torch 
import torch.nn as nn
import torch.nn.functional as F


LOG_FILES = ['../log/log-by-logger/log-by-loggerpy1_0.log',
             '../log/log-by-logger/log-by-loggerpy1_1.log',
             '../log/log-by-logger/log-by-loggerpy1_2.log',
             '../log/log-by-logger/log-by-loggerpy1_4.log',
             '../log/log-by-logger/log-by-loggerpy1_5.log',
             '../log/log-by-logger/log-by-loggerpy1_7.log',
             '../log/log-by-logger/log-by-loggerpy1_6.log',
             '../log/log-by-logger/log-by-loggerpy1_3.log']

class Net(nn.module):
    def __init__(self):
        super(Net, self).__init__()
        self.f1 = nn.Liner(16, 4)  
        self.f2 = nn.Liner(4,4)
        

    def encord(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def forward(self, x):
        output = self.encord(x.view(-1, 784)) #????
         
        return output



class GraspSystem():
    def __init__(self):
        pass

    # load depth_image and grasp_pos_rot data
    def load_data(self, log_files):
        """
        Args:
            dataset_path (str): example
                /home/Data
        """       
        data_file_prefixes = []
        key = '.pkl'
        for die_name, sub_dirs file in os.walk(dataset_path):
            for f in files:
                if key == f[-len(key):]:
                    data_file_prefixes.append(
                            os.path.join(dir_name, f[:-len(key)]))


    # make Net class model
    def make_model(self):
        self.model = Net()
        self.criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    def get_batch_train():
        pass
        for epoch in range(2):  # 訓練データを複数回(2周分)学習する
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # ローダからデータを取得する; データは [inputs, labels] の形で取得される
                # イテレータを使用していないように見えますが for の内部で使用されています。
                inputs, labels = data

                # 勾配を0に初期化する(逆伝播に備える)
                optimizer.zero_grad()

                # 順伝播 + 逆伝播 + 最適化(訓練)
                outputs = net(inputs)
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


    def get_test():
        pass

    # load traind Network model
    def load_model():
        pass

    def save_model(self):
        pass
        #serializers.save_hdf5()

    def train(self, loop_num=10):
        pass

    def test(self):
        pass


if __name__ == '__main__':
    # parse
    train_flag = True #int(arg.train)

    gs = GraspSystem()

    # train model or load model
    if train_flag:
        gs.load_data(LOGFILE)
        gs.tain(loop_num=10)
        gs.save_model()

    else:
        pass

    # test
    gs.start()


