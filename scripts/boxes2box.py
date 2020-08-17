#! /usr/bin/env python

import rospy
import rosparam #not need
import tf
from geometry_msgs.msg import Quaternion
from jsk_recognition_msgs.msg import BoundingBoxArray


def set(b):
    pos_list = [b.pose.position.x, b.pose.position.y, b.pose.position.z]
    # Convert Quaternion to Euler Angles
    e = tf.transformations.euler_from_quaternion((b.pose.orientation.x, b.pose.orientation.y, b.pose.orientation.z, b.pose.orientation.w))
    rot_list = [e[0], e[1], e[2]]
    rospy.set_param("initial_pos", pos_list) #Why cannot use rosparam.set_param instead?
    rospy.set_param("initial_rot", rot_list)
    rospy.set_param("dimention_x", b.dimensions.x)
    rospy.set_param("dimention_y", b.dimensions.y)
    rospy.set_param("dimention_z", b.dimensions.z)
    print(rospy)
    #rospy.set_param_name("attention_clipper")


# boxes is an BoundingBox Array
def cb (msg):
    boxes_list = msg.boxes
    if boxes_list:
        b = boxes_list[0]
        set(b)

if __name__ == '__main__':
    rospy.init_node('boundingboxarray_subscriber')
    rospy.Subscriber('/segmentation_decomposer/boxes',  BoundingBoxArray, cb)
    rospy.Publisher(
    rospy.spin()


~           
