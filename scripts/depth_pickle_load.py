#! /usr/bin/env python

import pickle
import os 
import numpy as np
from PIL import Image
from sklearn import preprocessing
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
import matplotlib

def colorize_depth(depth, min_value=None, max_value=None, dtype=np.uint8):
    """Colorize depth image with JET colormap."""
    min_value = np.nanmin(depth) if min_value is None else min_value
    max_value = np.nanmax(depth) if max_value is None else max_value
    if np.isinf(min_value) or np.isinf(max_value):
        warnings.warn('Min or max value for depth colorization is inf.')
    if max_value == min_value:
        eps = np.finfo(depth.dtype).eps
        max_value += eps
        min_value -= eps

    colorized = depth.copy()
    print("col", colorized)
    nan_mask = np.isnan(colorized)
    colorized[nan_mask] = 0
    colorized = 1. * (colorized - min_value) / (max_value - min_value)
    colorized = matplotlib.cm.jet(colorized)[:, :, :3]
    if dtype == np.uint8:
        colorized = (colorized * 255).astype(dtype)
    else:
        assert np.issubdtype(dtype, np.floating)
        colorized = colorized.astype(dtype)
    print("colotized", colorized)
    print("nan_mask", nan_mask)
    colorized[nan_mask] = (0, 0, 0)
    return colorized

depth_path = "Data/depth_data"
depth_dataset = np.empty((0,230400))
#depth_key = 'extract_depth_image.pkl'
#color_key = 'extract_color_image.pkl'
depth_key = '.pkl'
for d_dir_name, d_sub_dirs, d_files in sorted(os.walk(depth_path)): 
    for df in sorted(d_files):
        #if depth_key == df[-len(depth_key):]:
        if depth_key == df[-len(depth_key):]:
            with open(os.path.join(d_dir_name, df), 'rb') as f:
                ff = pickle.load(f)
                depth_image = ff
                WIDTH = 64#240
                HEIGHT = 64#240
                """
                bridge = CvBridge()
                try:
                    depth_image = bridge.imgmsg_to_cv2(ff, 'passthrough')
                except CvBridgeError, e:
                    rospy.logerr(e)
                """
                
                im = ff.reshape((128,128,2))
                depth_image = im[:, :, 0]
                #im_gray = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
                #depth_image = im_gray
                
                print("depth image shape", depth_image.shape)
                h, w = depth_image.shape

                x1 = (w / 2) - WIDTH
                x2 = (w / 2) + WIDTH
                y1 = (h / 2) - HEIGHT
                y2 = (h / 2) + HEIGHT
                depth_data = np.empty((0,16384))

                for i in range(y1, y2):
                    for j in range(x1, x2):
                        if depth_image.item(i,j) == depth_image.item(i,j):
                            depth_data = np.append(depth_data, depth_image.item(i,j))
                        else:
                            depth_data = np.append(depth_data, 0)
                                    
                ims = depth_data.reshape((1, 128, 128)) #480, 480))
                print(ims)
                #im = colorize_depth(ims)
                #plt.imshow(ims)
                #plt.show()
                plt.imshow(np.transpose(ims[0, :, :]))
                plt.show()
                """            
                depth_data = np.array(depth_data).reshape((1, 230400))
                depth_dataset = np.append(depth_dataset, depth_data, axis=0)
                
                ims = depth_dataset.reshape((1, 480, 480))
                plt.imshow(np.transpose(ims[0, :, :]))
                plt.show()
                """
                print("printed")

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

