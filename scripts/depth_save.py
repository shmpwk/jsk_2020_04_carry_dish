#! /usr/bin/env python
# -*- coding: utf-8 -*-
 
import rospy
import cv2
import sys
import csv
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

def ImageCallback(rgb_data , depth_data):
    WIDTH = 50
    HEIGHT = 25
    bridge = CvBridge()
    try:
        color_image = bridge.imgmsg_to_cv2(rgb_data, 'passthrough')
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
    rospy.init_node('depth_estimater', anonymous=True)
    sub_rgb = message_filters.Subscriber("/camera/rgb/image_raw",Image)
    sub_depth = message_filters.Subscriber("/camera/depth_registered/sw_registered/image_rect",Image)
    mf = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth], 100, 10.0)
    mf.registerCallback(ImageCallback)
    pub = rospy.Publisher("/depth_crds", Point)
    rospy.spin()

