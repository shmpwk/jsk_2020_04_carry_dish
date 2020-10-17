#! /usr/bin/env python

import pickle
import os 
import numpy as np
from PIL import Image
from sklearn import preprocessing
import matplotlib.pyplot as plt

grasp_dataset = np.empty((0,4))

grasp_path = ".ros/Data/grasp_point"
g_key = '.pkl'
for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
    for gf in sorted(g_files):
        if g_key == gf[-len(g_key):]:
            with open(os.path.join(g_dir_name, gf), 'rb') as f:
                ff = pickle.load(f)
                ff = np.array(ff).reshape((1, 4))
                grasp_dataset = np.append(grasp_dataset, ff, axis=0)
                print(gf)
print(grasp_dataset.shape)
#ims = grasp_dataset.reshape((10, 200, 200))
#plt.imshow(np.transpose(ims[0, :, :]))
#plt.show()
#im = Image.fromarray(ffff)

