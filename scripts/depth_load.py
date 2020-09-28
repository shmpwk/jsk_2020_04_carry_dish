#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
succeeded to load pickle and png, but cannot use cv2 so cannot resize to square.
"""

import rospy
import cv2
import sys
import csv
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point
import os 
import pickle
from PIL import Image

def load_depth():
    depth_dataset_path = os.path.expanduser('~/Data/depth_data')

    depth_key = '.pkl'
    rgb_key = '.png'
    for dir_name, sub_dirs, files in os.walk(depth_dataset_path):
        for f in files:
            load_file = os.path.join(dir_name, f)
            if depth_key ==f[-len(depth_key):]:
                depth_data = pickle.load(open(load_file, 'rb'))
                print(depth_data)
            
            if rgb_key == f[-len(rgb_key):]:
                print(f)
                rgb_data = Image.open(load_file)
                print(rgb_data)
    WIDTH = 50
    HEIGHT = 25
    bridge = CvBridge()
    
    try:
        #color_image = bridge.imgmsg_to_cv2(rgb_data, 'passthrough')
        depth_image = bridge.imgmsg_to_cv2(depth_data, 'passthrough')
    except CvBridgeError, e:
        rospy.logerr(e)

    #color_image.flags.writeable = True #It leads errors 
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) 
    h, w, c = color_image.shape

    x1 = (w / 2) - WIDTH
    x2 = (w / 2) + WIDTH
    y1 = (h / 2) - HEIGHT
    y2 = (h / 2) + HEIGHT
    sum = 0.0
    points = []

    for i in range(y1, y2):
        for j in range(x1, x2):
            color_image.itemset((i, j, 0), 0) #color roi without blue 
            color_image.itemset((i, j, 1), 0) #color roi without green
            #color_image.itemset((100,100,2), 0)

            if depth_image.item(i,j) == depth_image.item(i,j):
                point = Point(i, j, depth_image.item(i,j))
                points.append(point)

    #pub.publisher(points)            
    ave = sum / ((WIDTH * 2) * (HEIGHT * 2)) #average distance 
    #print("%f [m]" % ave)
    #print(points)
    with open('sample_depth_image.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(depth_image) #datasize=480
    #with open('sample_points.csv', 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerows(points)

    cv2.normalize(depth_image, depth_image, 0, 1, cv2.NORM_MINMAX)
    cv2.namedWindow("color_image")
    cv2.namedWindow("depth_image")
    cv2.imshow("color_image", color_image)
    cv2.imshow("depth_image", depth_image)
    cv2.waitKey(10)
     

if __name__ == '__main__':
    load_depth()

