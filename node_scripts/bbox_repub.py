#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
from absl import app
from absl import flags

from jsk_recognition_msgs.msg import *
from geometry_msgs.msg import *

dFLAGS = flags.FLAGS

flags.DEFINE_string(
        'obj', 'dish', 'obj to pick up: dish, cutlery, tray')

def divided_box_cb(msg):
    div_box = msg
    if (dFLAGS.obj == 'dish'):
        if (msg.dimensions.x > msg.dimensions.y):
            div_box.pose.position.x = msg.pose.position.x + msg.dimensions.x / 4
            div_box.dimensions.y = msg.dimensions.y / 2
        else:
            div_box.dimensions.x = msg.dimensions.x / 2
            div_box.pose.position.y = msg.pose.position.y + msg.dimensions.y / 4
        
    elif (dFLAGS.obj == 'cutlery'):
        if (msg.dimensions.x > msg.dimensions.y):
            div_box.dimensions.x = msg.dimensions.x / 80
        else:
            div_box.dimensions.y = msg.dimensions.y / 80
     
    elif (dFLAGS.obj == 'tray'):
        if (msg.dimensions.x > msg.dimensions.y):
            div_box.pose.position.y = msg.pose.position.y + msg.dimensions.y / 8 * 7
            div_box.dimensions.x = msg.dimensions.x / 6
        else:
            div_box.dimensions.y = msg.dimensions.y / 6
            div_box.pose.position.x = msg.pose.position.x + msg.dimensions.x / 8 * 7
     
    bbox_pub.publish(div_box)

def main(argv):
    rospy.init_node('divided_bbox')
    listener = tf.TransformListener()
    position_sub = rospy.Subscriber('/bounding_box_marker/selected_box', BoundingBox, divided_box_cb)
    bbox_pub = rospy.Publisher(
        'divided_bbox', BoundingBox, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
    print("=============================================started!!")
    app.run(main)
