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
        self.conv1 = nn.Conv2d(1, 6, 5)   # 入力のチャネル数は1，出力のチャネル数は6，5x5の畳み込み層
        self.conv2 = nn.Conv2d(6, 16, 5)  # 入力のチャネル数は6，出力のチャネル数は16，5x5の畳み込み層


class GraspSystem():
    def __init__(self):
        pass

    # load current state data
    def load_data(self, log_files):
        pass

    # make Net class model
    def make_model(self):
        pass

    def get_batch_train():
        pass

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


