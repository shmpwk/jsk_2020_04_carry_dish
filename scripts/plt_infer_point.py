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

grasp_path = "Data/plt/inferred_grasp_point"
g_key = '.pkl'
for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
    for gf in sorted(g_files):
        if g_key == gf[-len(g_key):]:
            with open(os.path.join(g_dir_name, gf), 'rb') as f:
                ff = pickle.load(f)
                ff = np.array(ff).reshape((1, 4))
                grasp_dataset = np.append(grasp_dataset, ff, axis=0)
gx = grasp_dataset[:,0]
gy = grasp_dataset[:,1]
gz = grasp_dataset[:,2]
gt = grasp_dataset[:,3]
gxp = grasp_dataset[:9,0]
gyp = grasp_dataset[:9,1]
gzp = grasp_dataset[:9,2]
gtp = grasp_dataset[:9,3]
gxl = grasp_dataset[9,0]
gyl = grasp_dataset[9,1]
gzl = grasp_dataset[9,2]
gtl = grasp_dataset[9,3]
grasp_dataset = np.empty((0,4))

grasp_path = "Data/plt/inferred_point"
g_key = '.pkl'
for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
    for gf in sorted(g_files):
        if g_key == gf[-len(g_key):]:
            with open(os.path.join(g_dir_name, gf), 'rb') as f:
                ff = pickle.load(f)
                ff = np.array(ff).reshape((1, 4))
                grasp_dataset = np.append(grasp_dataset, ff, axis=0)
ix = grasp_dataset[:9,0]
iy = grasp_dataset[:9,1]
iz = grasp_dataset[:9,2]
it = grasp_dataset[:9,3]
ixl = grasp_dataset[9,0]
iyl = grasp_dataset[9,1]
izl = grasp_dataset[9,2]
itl = grasp_dataset[9,3]
#ims = grasp_dataset.reshape((10, 200, 200))
#plt.imshow(np.transpose(ims[0, :, :]))
#plt.show()
#im = Image.fromarray(ffff)

grasp_dataset = np.empty((0,4))

grasp_path = "Data/plt/all_edge_point"
g_key = '.pkl'
for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
    for gf in sorted(g_files):
        if g_key == gf[-len(g_key):]:
            with open(os.path.join(g_dir_name, gf), 'rb') as f:
                ff = pickle.load(f)

x = ff[1:,0]
y = ff[1:,1]
z = ff[1:,2]

grasp_dataset = np.empty((0,4))

grasp_path = "Data/plt/obj_pcl"
g_key = '.pkl'
for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
    for gf in sorted(g_files):
        if g_key == gf[-len(g_key):]:
            with open(os.path.join(g_dir_name, gf), 'rb') as f:
                ff = pickle.load(f)

px = ff[1:,0]
py = ff[1:,1]
pz = ff[1:,2]

grasp_dataset = np.empty((0,7))

grasp_path = "Data/plt/trans"
g_key = '.pkl'
for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
    for gf in sorted(g_files):
        if g_key == gf[-len(g_key):]:
            with open(os.path.join(g_dir_name, gf), 'rb') as f:
                ff = pickle.load(f)

cx = ff[:,0]
cy = ff[:,1]
cz = ff[:,2]
cs = ff[:,3]
ct = ff[:,4]
cu = ff[:,5]
cv = ff[:,6]
print(cx,cy,cz,cs,ct,cu,cv)

grasp_dataset = np.empty((0,3))

grasp_path = "Data/plt/box_pos"
g_key = '.pkl'
for g_dir_name, g_sub_dirs, g_files in sorted(os.walk(grasp_path)): 
    for gf in sorted(g_files):
        if g_key == gf[-len(g_key):]:
            with open(os.path.join(g_dir_name, gf), 'rb') as f:
                ff = pickle.load(f)
bx = ff[0]
by = ff[1]
bz = ff[2]

dgx = ((gx-cy)-(bx-cy))
dgy = ((gy+cx)-(by+cx))
dgz = np.sqrt(dgx**2 + dgy**2) * np.tan(math.pi-gt)
n = np.sqrt(dgx**2 + dgy**2 + dgz**2) 
#dgz = math.sqrt((gx-cy)**2+(gy+cx)**2) * tan(math.pi-gt)
#soa = np.array([[0, 0, 0.02, 0.05, 0.03, 0],
#                [0, 0, 0.3, 0.01, 0.03, 0]])
soa = np.array([[gx-cy, gy+cx, cz-gz, -dgx/n/50, -dgy/n/50, -dgz/n/50]])
soal = np.array([[(gx-cy)[9], (gy+cx)[9], (cz-gz)[9], -(dgx/n/50)[9], -(dgy/n/50)[9], -(dgz/n/50)[9]]])
#soal = soa
X, Y, Z, U, V, W = zip(*soa)[:][:][:]
Xl, Yl, Zl, Ul, Vl, Wl = zip(*soal)[:][:][:]

print(np.array((X, Y, Z, U, V, W)).shape)
print(np.array((Xl, Yl, Zl, Ul, Vl, Wl)).shape)

#グラフの枠を作っていく
fig = plt.figure()
ax = Axes3D(fig)
ax.quiver(X, Y, Z, U, V, W, color="orange")
ax.quiver(Xl, Yl, Zl, Ul, Vl, Wl, color="red", linewidth=5)

#.plotで描画
#linestyle='None'にしないと初期値では線が引かれるが、3次元の散布図だと大抵ジャマになる
#markerは無難に丸
#ax.plot(x, y, z, marker="o", s=20,  linestyle='None')
ax.scatter(x-cy, y+cx, cz-z, s=0.8, c="green")
ax.scatter(px-cy, py+cx, cz-pz, s=0.05, c="black")
ax.scatter(ix-cy, iy+cx, cz-iz, s=40, c="cyan")
ax.scatter(ixl-cy, iyl+cx, cz-izl, s=40, c="blue")
ax.scatter(gxp-cy, gyp+cx, cz-gzp, s=50, c="orange")
ax.scatter(gxl-cy, gyl+cx, cz-gzl, s=70, c="red")
ax.scatter(bx-cy, by+cx, cz-bz, s=100, c="olive")
"""
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# make the panes transparent
ax.xaxis.set_pane_color((0, 0, 0, 0.0))
ax.yaxis.set_pane_color((0, 0, 0, 0.0))
ax.zaxis.set_pane_color((0, 0, 0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (0,0,0,0)
ax.yaxis._axinfo["grid"]['color'] =  (0,0,0,0)
ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,0)
"""
# transparent backgroud. see https://stackoverrun.com/ja/q/10585327
ax.axis("off")
fig.patch.set_visible(False)
ax.patch.set_visible(False)
ax.set_axis_off()
ax._axis3don = False
#最後に.show()を書いてグラフ表示
plt.show()

