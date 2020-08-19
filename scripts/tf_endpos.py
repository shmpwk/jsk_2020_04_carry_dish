#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import PoseStamped
from std.msg import string

if __name__=='__main__':
    rospy.init_node('hoge')
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        target_pose = TransformSamped()
        starget_pose.header.frame_id = "/r_gripper_tool_frame"
        target_pose.pose.

        source_pose = PoseStamped()
        source_pose.header.frame_id = "/base_footprint"
        source_pose.header.stamp = rospy.Time.now()
        source_pose.pose.

        try:
            target_frame  = 
            source_frame =

            listener = tf.TransformListener()
            listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
            tf = listener.transformPose(source_frame, target_frame, rospy.Time(0))
            pub = rospy.Publisher('endcrd', PoseStamped, queue_size=1)
            
            pub.publish(tf)

        except rospy.ROSInterruptException: pass
