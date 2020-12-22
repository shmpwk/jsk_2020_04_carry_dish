#!/usr/bin/env python

import rospy
import tf

from jsk_recognition_msgs.msg import *
from geometry_msgs.msg import *

def divided_box_cb(msg):
    div_box = msg
    if (msg.dimensions.x > msg.dimensions.y):
        div_box.dimensions.x = msg.dimensions.x / 50
    else:
        div_box.dimensions.y = msg.dimensions.y / 50
    bbox_pub.publish(div_box)

rospy.init_node('divided_bbox')
listener = tf.TransformListener()
position_sub = rospy.Subscriber('/bounding_box_marker/selected_box', BoundingBox, divided_box_cb)
bbox_pub = rospy.Publisher(
    'divided_bbox', BoundingBox, queue_size=1)
rospy.spin()
    
    bbox_pub.publish(div_box)

rospy.init_node('divided_bbox')
listener = tf.TransformListener()
position_sub = rospy.Subscriber('/bounding_box_marker/selected_box', BoundingBox, divided_box_cb)
bbox_pub = rospy.Publisher(
    'divided_bbox', BoundingBox, queue_size=1)
rospy.spin()
