#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inflate grasp point *100
"""

import numpy as np
import math

grasp_point = np.array((1,2,3,4))
#grasp_point = grasp_point.reshape((1,4))
aug_grasp_point = np.zeros((100,4))
x = grasp_point[[0]]
y = grasp_point[[1]]
z = grasp_point[[2]]
theta = grasp_point[[3]]

r = math.sqrt(x**2 + y**2)
rad = math.atan2(y, x)
rad_delta = math.pi/50
X = np.zeros(100)
Y = np.zeros(100)
for i in range(100):
    X[i] = r * math.cos(rad + rad_delta*i)
    Y[i] = r * math.sin(rad + rad_delta*i)
    aug_grasp_point[i, :] = np.array((X[i], Y[i], z, theta))

print(aug_grasp_point[4,:])

