#! /usr/bin/env python

import pickle
import os 
import numpy as np

grasp_dataset = np.empty((0,6))

for file in os.listdir(".ros/Data/grasp_point"):
    with open (".ros/Data/grasp_point/" + file, "rb") as f:
        ff = pickle.load(f)
        grasp_dataset = np.append(grasp_dataset, ff, axis=0)
print(grasp_dataset)
