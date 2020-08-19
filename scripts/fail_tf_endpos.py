#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import Pose

if __name__=='__main__':
    try:
        rospy.init_node('hoge')
        target_frame  = "/r_gripper_tool_frame"
        source_frame = "/base_footprint"

        listener = tf.TransformListener()
        listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
        tf = listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
        pub = rospy.Publisher('endcrd', Pose, queue_size=1)
        
        pub.publish(tf)

    except rospy.ROSInterruptException: pass
