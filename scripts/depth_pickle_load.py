#! /usr/bin/env python

import pickle
import os 
import numpy as np
from PIL import Image
from sklearn import preprocessing
import matplotlib.pyplot as plt

depth_path = "Data/depth_data"
depth_dataset = np.empty((0,230400))
depth_key = '.pkl'
for d_dir_name, d_sub_dirs, d_files in os.walk(depth_path): 
    for df in d_files:
        if depth_key == df[-len(depth_key):]:
            with open(os.path.join(d_dir_name, df), 'rb') as f:
                ff = pickle.load(f)

                WIDTH = 240
                HEIGHT = 240 
                #try:
                #    depth_image = bridge.imgmsg_to_cv2(ff, 'passthrough')
                #except CvBridgeError, e:
                #    rospy.logerr(e)
                im = ff.reshape((480,640,3))
                im_gray = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
                depth_image = im_gray

                h, w = depth_image.shape

                x1 = (w / 2) - WIDTH
                x2 = (w / 2) + WIDTH
                y1 = (h / 2) - HEIGHT
                y2 = (h / 2) + HEIGHT
                depth_data = np.empty((0,230400))

                for i in range(y1, y2):
                    for j in range(x1, x2):
                        if depth_image.item(i,j) == depth_image.item(i,j):
                            depth_data = np.append(depth_data, depth_image.item(i,j))
                                    
                #ims = depth_data.reshape((1, 480, 480))
                #plt.imshow(np.transpose(ims[0, :, :]))
                #plt.show()
                            
                depth_data = np.array(depth_data).reshape((1, 230400))
                depth_dataset = np.append(depth_dataset, depth_data, axis=0)
                
                ims = depth_dataset.reshape((1, 480, 480))
                plt.imshow(np.transpose(ims[0, :, :]))
                plt.show()

"""
for file in os.listdir(".ros/Data/depth_image"):
    with open (".ros/Data/depth_image/" + file, "rb") as f:
        ff = pickle.load(f)
        ff = np.array(ff).reshape((1, 480*640))
        grasp_dataset = np.append(grasp_dataset, ff, axis=0)
        """

"""
mm =preprocessing.MinMaxScaler()
ffff = mm.fit_transform(fff)
#print(ffff)
im = Image.fromarray(ffff)
"""

#im.save(".ros/Data/gazebo_depth_image/test.jpg")

