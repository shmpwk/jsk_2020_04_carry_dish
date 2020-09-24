#! /usr/bin/env python

# load edge pointcloud data
# select one point
# choose rotation
# publish grasp point by service 

import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 
from geometry_msgs.msg import Pose
import tf
import numpy as np

def choose_point_callback(data):
    assert isinstance(data, PointCloud2)
    """
    Randomly choose position (x, y, z) data from point cloud
    """
    gen = point_cloud2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)
    A = np.arange(3).reshape(1,3)
    for l in gen:
        l = np.array(l)
        l = l.reshape(1,3)
        A = np.append(A, l, axis=0)
        
    idx = np.random.randint(10, size=1) #To do : change 10 to data size
    Ax = A[idx, 0]
    Ay = A[idx, 1]
    Az = A[idx, 2]

    """
    Romdomly choose rotation theta from 0, 45, 90. (currently, 90 for test).
    Other roatation angle is fixed toward middle point.

    """

    theta = -1.54 
    phi = 1.2
    psi = 0
    q = tf.transformations.quaternion_from_euler(theta, phi, psi)



    pose = Pose()
    pose.position.x = Ax
    pose.position.y = Ay
    pose.position.z = Az 

    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    
    pub.publish(pose)

    

# get depth and get grasp point 
if __name__=="__main__":
    #subscribe edge pointcloud data
    try:
        rospy.init_node('grasp_point_server')
        rospy.Subscriber('/camera/depth_registered/points', PointCloud2, choose_point_callback, queue_size=10)
        pub = rospy.Publisher('/grasp_point', Pose, queue_size=10)
        rospy.spin()
    except rospy.ROSInterruptException: pass

"""
def grasp_point_server():
    rospy.init_node('grasp_point_server')
    s = rospy.Service('grasp_point', GetGraspPoint, handle_grasp_point)
    print("ready to get grasp point")
    rospy.spin()

#if __name__=="__main__":
#    #grasp_point_server()
#    handle_grasp_point()
"""
