#! /usr/bin/env python

import pickle
import os 
import numpy as np
from PIL import Image
from sklearn import preprocessing
import matplotlib.pyplot as plt

grasp_dataset = np.empty((0,10000))

for file in os.listdir(".ros/Data/depth_image"):
    with open (".ros/Data/depth_image/" + file, "rb") as f:
        ff = pickle.load(f)
        ff = np.array(ff).reshape((1, 10000))
        grasp_dataset = np.append(grasp_dataset, ff, axis=0)
print(grasp_dataset.shape)
ims = grasp_dataset.reshape((1, 100, 100))
plt.imshow(np.transpose(ims[0, :, :]))
plt.show()

"""
mm =preprocessing.MinMaxScaler()
ffff = mm.fit_transform(fff)
#print(ffff)
im = Image.fromarray(ffff)
"""

#im.save(".ros/Data/gazebo_depth_image/test.jpg")

