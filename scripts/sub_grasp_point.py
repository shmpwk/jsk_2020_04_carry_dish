#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Pose

def grasp_point_cb (msg):
    print msg.position

rospy.init_node("grasp_point")
rospy.Subscriber("grasp_point", Pose, grasp_point_cb)
rospy.spin( )

