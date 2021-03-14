#!/usr/bin/env python
import skrobot
from skrobot.interfaces import PR2ROSRobotInterface
import rospy
import tf
import numpy as np
from geometry_msgs.msg import *

rospy.init_node('pub_pose')
pose_pub = rospy.Publisher('/pr2_pose', PoseStamped, queue_size=1)
model = skrobot.models.PR2()
print(model.rarm.end_coords.copy_worldcoords())
print(model.rarm.end_coords.worldpos())
print(model.rarm.end_coords.worldrot())
model.reset_manip_pose()
print(model.rarm.end_coords.worldpos())
print(model.rarm.end_coords.worldrot())

interface = PR2ROSRobotInterface(model)
posestamped = PoseStamped()
while not rospy.is_shutdown():
    model.angle_vector(interface.angle_vector())
    print(model.rarm.end_coords.worldpos())
    print(model.rarm.end_coords.worldrot())
    posestamped.pose.position.x = np.array(model.rarm.end_coords.worldpos())[0]
    posestamped.pose.position.y = np.array(model.rarm.end_coords.worldpos())[1]
    posestamped.pose.position.z = np.array(model.rarm.end_coords.worldpos())[2]
    eul = tf.transformations.euler_from_matrix(model.rarm.end_coords.worldrot())
    posestamped.pose.orientation.x = tf.transformations.quaternion_from_euler(np.array(eul)[0], np.array(eul)[1], np.array(eul)[2])[0]
    posestamped.pose.orientation.y = tf.transformations.quaternion_from_euler(np.array(eul)[0], np.array(eul)[1], np.array(eul)[2])[1]
    posestamped.pose.orientation.z = tf.transformations.quaternion_from_euler(np.array(eul)[0], np.array(eul)[1], np.array(eul)[2])[2]
    posestamped.pose.orientation.w = tf.transformations.quaternion_from_euler(np.array(eul)[0], np.array(eul)[1], np.array(eul)[2])[3]
    posestamped.header.stamp = rospy.Time(0)
    posestamped.header.frame_id = "base_link"
    pose_pub.publish(posestamped)
     
