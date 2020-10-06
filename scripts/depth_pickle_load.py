#! /usr/bin/env python

import pickle
import os 
import numpy as np
from PIL import Image
from sklearn import preprocessing

grasp_dataset = np.empty((0,10000))

for file in os.listdir(".ros/Data/gazebo_depth_image"):
    with open (".ros/Data/gazebo_depth_image/" + file, "rb") as f:
        ff = pickle.load(f)
        #girasp_dataset = np.append(grasp_dataset, ff, axis=0)
#print(grasp_dataset)
fff = np.array(ff).reshape((100, 100))
print(fff)
mm =preprocessing.MinMaxScaler()
ffff = mm.fit_transform(fff)
#print(ffff)
im = Image.fromarray(ffff)


#im.save(".ros/Data/gazebo_depth_image/test.jpg")

