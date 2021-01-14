#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os 
import numpy as np
from PIL import Image
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance
#seabornでグラフをきれいにしたいだけのコード
import seaborn as sns
sns.set_style("darkgrid")

#3次元プロットするためのモジュール
from mpl_toolkits.mplot3d import Axes3D 

grasp_dataset = np.empty((0,4))

grasp_path = "Data/inferred_grasp_point"
g_key = '.pkl'
for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
    for gf in sorted(g_files):
        if g_key == gf[-len(g_key):]:
            with open(os.path.join(g_dir_name, gf), 'rb') as f:
                ff = pickle.load(f)
                ff = np.array(ff).reshape((1, 4))
                grasp_dataset = np.append(grasp_dataset, ff, axis=0)
                print(ff)
gx = grasp_dataset[:,0]
gy = grasp_dataset[:,1]
gz = grasp_dataset[:,2]
gt = grasp_dataset[:,3]
grasp_dataset = np.empty((0,4))

grasp_path = "Data/inferred_point"
g_key = '.pkl'
for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
    for gf in sorted(g_files):
        if g_key == gf[-len(g_key):]:
            with open(os.path.join(g_dir_name, gf), 'rb') as f:
                ff = pickle.load(f)
                ff = np.array(ff).reshape((1, 4))
                grasp_dataset = np.append(grasp_dataset, ff, axis=0)
                print(ff)
ix = grasp_dataset[:,0]
iy = grasp_dataset[:,1]
iz = grasp_dataset[:,2]
it = grasp_dataset[:,3]
#ims = grasp_dataset.reshape((10, 200, 200))
#plt.imshow(np.transpose(ims[0, :, :]))
#plt.show()
#im = Image.fromarray(ffff)

grasp_dataset = np.empty((0,4))

grasp_path = "Data/all_edge_point"
g_key = '.pkl'
for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
    for gf in sorted(g_files):
        if g_key == gf[-len(g_key):]:
            with open(os.path.join(g_dir_name, gf), 'rb') as f:
                ff = pickle.load(f)

x = ff[1:,0]
y = ff[1:,1]
z = ff[1:,2]

grasp_dataset = np.empty((0,3))

grasp_path = "Data/trans"
g_key = '.pkl'
for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
    for gf in sorted(g_files):
        if g_key == gf[-len(g_key):]:
            with open(os.path.join(g_dir_name, gf), 'rb') as f:
                ff = pickle.load(f)
                print(ff)

cx = ff[:,0]
cy = ff[:,1]
cz = ff[:,2]

dgx = 2*(gx-cy)/10
dgy = 2*(gy+cx)/10
dgz = distance.euclidean(gx-cy, gy+cx) * np.tan(math.pi-gt)/10

#dgz = math.sqrt((gx-cy)**2+(gy+cx)**2) * tan(math.pi-gt)
#soa = np.array([[0, 0, 0.02, 0.05, 0.03, 0],
#                [0, 0, 0.3, 0.01, 0.03, 0]])
soa = np.array([[gx-cy, gy+cx, gz-cz, dgx, dgy, dgz]])
X, Y, Z, U, V, W = zip(*soa)

#グラフの枠を作っていく
fig = plt.figure()
ax = Axes3D(fig)
ax.quiver(X, Y, Z, U, V, W)

#.plotで描画
#linestyle='None'にしないと初期値では線が引かれるが、3次元の散布図だと大抵ジャマになる
#markerは無難に丸
#ax.plot(x, y, z, marker="o", s=20,  linestyle='None')
ax.scatter(x-cy, y+cx, z-cz, s=5, c="green")
ax.scatter(ix-cy, iy+cx, iz-cz, s=40, c="blue")
ax.scatter(gx-cy, gy+cx, gz-cz, s=70, c="red")
#最後に.show()を書いてグラフ表示
plt.show()
