#! /usr/bin/env python
# -*- coding: utf-8 -*-
 
import rospy
import cv2
import sys
import csv
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point
import pickle
from absl import app
from absl import flags
import datetime
import time
import numpy as np
import pickle
import os 
import time 
import matplotlib.pyplot as plt

#FLAGS = flags.FLAGS

#flags.DEFINE_string(
#        'depth_topic', '/head_mount_kinect/depth/image_raw', 'depth topic name')

def ImageCallback(depth_data):
    WIDTH = 100
    HEIGHT = 100 
    depth_image = depth_data

    #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) 
    #h, w, c = color_image.shape
    h, w = depth_image.shape

    x1 = (w / 2) - WIDTH
    x2 = (w / 2) + WIDTH
    y1 = (h / 2) - HEIGHT
    y2 = (h / 2) + HEIGHT
    sum = 0.0
    points = []
    #depth_data = []
    depth_data = np.empty(0)

    for i in range(y1, y2):
        for j in range(x1, x2):
            #color_image.itemset((i, j, 0), 0) #color roi without blue 
            #color_image.itemset((i, j, 1), 0) #color roi without green
            if depth_image.item(i,j) == depth_image.item(i,j):
                #point = Point(i, j, depth_image.item(i,j))
                #points.append(point)
                #depth_data.append(depth_image.item(i,j))
                depth_data = np.append(depth_data, depth_image.item(i,j))
            else:
                depth_data = np.append(depth_data, 0)

    ave = sum / ((WIDTH * 2) * (HEIGHT * 2)) #average distance 
    now = datetime.datetime.now()
    filename = 'Data/gazebo_depth_image/depth_image_' + now.strftime('%Y%m%d_%H%M%S') + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(depth_data, f) #datasize=480
    time.sleep(2)
    print("depth saved at node script")
    print("depthdata",depth_data.reshape((200,200)).shape)
    print("depthimage", depth_image.shape)

    cv2.normalize(depth_image, depth_image, 0, 1, cv2.NORM_MINMAX)
    #cv2.namedWindow("color_image")
    cv2.namedWindow("depth_image")
    #cv2.imshow("color_image", color_image)
    #cv2.imshow("depth_image", depth_data)
    cv2.imshow("depth_image", depth_image)
    #plt.imshow(depth_data)
    #plt.show()
    cv2.waitKey(1000)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

#def main():
if __name__ == '__main__':
    """ 
    sub_rgb = message_filters.Subscriber("/head_mount_kinect/rgb/image_raw",Image)
    sub_depth = message_filters.Subscriber("/head_mount_kinect/depth/image_raw",Image)
    sub_points = message_filters.Subscriber("/head_mount_kinect/depth_registered/points", PointCloud2)
    mf = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth, sub_points], 100, 10.0)
    mf.registerCallback(ImageCallback)
    #pub = rospy.Publisher("/depth_crds", Point)
    rospy.spin()
    """
    depth_path = "Data/depth_data"
    depth_key = '.pkl'
    for dir_name, sub_dirs, files in os.walk(depth_path): 
        for f in files:
            if depth_key == f[-len(depth_key):]:
                with open(os.path.join(dir_name, f), 'rb') as f:
                    data = pickle.load(f)
                    ImageCallback(data)

"""
if __name__ == '__main__':
    print("=============================================started!!")
    #app.run(main)
    main"""
