#! /usr/bin/env python

import pickle
import os 
import numpy as np
from PIL import Image
from sklearn import preprocessing
import matplotlib.pyplot as plt

grasp_dataset = np.empty((0,6))

for file in os.listdir(".ros/Data/grasp_point"):
    with open (".ros/Data/grasp_point/" + file, "rb") as f:
        ff = pickle.load(f)
        ff = np.array(ff).reshape((1, 6))
        grasp_dataset = np.append(grasp_dataset, ff, axis=0)
print(grasp_dataset.shape)
#ims = grasp_dataset.reshape((10, 200, 200))
#plt.imshow(np.transpose(ims[0, :, :]))
#plt.show()
#im = Image.fromarray(ffff)

