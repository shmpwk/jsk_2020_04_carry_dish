#!/usr/bin/env python

import rospy
import tf

from jsk_recognition_msgs.msg import *
from geometry_msgs.msg import *
from std_msgs.msg import *

def wrench_cb(msg):
    x_force =  msg.wrench.force.x
    y_force =  msg.wrench.force.y
    z_force =  msg.wrench.force.z
    xy = x_force**2 + y_force**2
    yz = y_force**2 + z_force**2
    zx = z_force**2 + x_force**2
    #print("xy", xy)
    #print("yz", yz)
    #print("zx", zx)
    print (y_force)
    if y_force > 3:
        pub.publish(1)
        print("touch")
    else :
        pub.publish(0)
        print("nothing")
    print("==========")

rospy.init_node('touch_detect')
position_sub = rospy.Subscriber('/right_endeffector/wrench', WrenchStamped, wrench_cb)
pub = rospy.Publisher(
    'r_contact', Bool, queue_size=1)
rospy.spin()
    
