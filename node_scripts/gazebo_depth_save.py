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

#FLAGS = flags.FLAGS

#flags.DEFINE_string(
#        'depth_topic', '/head_mount_kinect/depth/image_raw', 'depth topic name')

def ImageCallback(depth_data):
    WIDTH = 100
    HEIGHT = 100 
    bridge = CvBridge()
    try:
        #color_image = bridge.imgmsg_to_cv2(rgb_data, 'passthrough')
        depth_image = bridge.imgmsg_to_cv2(depth_data, 'passthrough')
    except CvBridgeError, e:
        rospy.logerr(e)

    #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) 
    #h, w, c = color_image.shape
    h, w = depth_image.shape

    x1 = (w / 2) - WIDTH
    x2 = (w / 2) + WIDTH
    y1 = (h / 2) - HEIGHT
    y2 = (h / 2) + HEIGHT
    sum = 0.0
    points = []
    depth_data = []

    for i in range(y1, y2):
        for j in range(x1, x2):
            #color_image.itemset((i, j, 0), 0) #color roi without blue 
            #color_image.itemset((i, j, 1), 0) #color roi without green
            if depth_image.item(i,j) == depth_image.item(i,j):
                print("depth image item", depth_image.item(i,j))
                #point = Point(i, j, depth_image.item(i,j))
                #points.append(point)
                depth_data.append(depth_image.item(i,j))

    print("depth_data was saved?????????????", depth_data) 
    ave = sum / ((WIDTH * 2) * (HEIGHT * 2)) #average distance 
    now = datetime.datetime.now()
    filename = 'Data/gazebo_depth_image/depth_image_' + now.strftime('%Y%m%d_%H%M%S') + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(depth_data, f) #datasize=480
    print("depth saved at node script")

    cv2.normalize(depth_image, depth_image, 0, 1, cv2.NORM_MINMAX)
    #cv2.namedWindow("color_image")
    cv2.namedWindow("depth_image")
    #cv2.imshow("color_image", color_image)
    cv2.imshow("depth_image", depth_image)
    cv2.waitKey(10)

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
    try:
        time.sleep(50)
        rospy.init_node('depth_estimater', anonymous=True)
        while not rospy.is_shutdown():
            #topic_name = "{}".format(FLAGS.depth_topic) 
            topic_name = "/head_mount_kinect/depth/image_raw"
            data = rospy.wait_for_message(topic_name,Image)
            #data = rospy.wait_for_message("/head_mount_kinect/depth/image_raw", Image)
            ImageCallback(data)
            break
    except rospy.ROSInterruptException: pass
"""
if __name__ == '__main__':
    print("=============================================started!!")
    #app.run(main)
    main"""
