#! /usr/bin/env python

# load edge pointcloud data
# select one point
# choose rotation
# publish grasp point by service 

import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 
from geometry_msgs.msg import PoseStamped
import tf
import numpy as np
import csv
import pickle

def choose_point_callback(data):
    assert isinstance(data, PointCloud2)
    """
    Randomly choose position (x, y, z) data from object edge point cloud.
    Numpy array A is selected position.
    """
    gen = point_cloud2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)
    length = 1 
    A = np.arange(3).reshape(1,3)
    for l in gen:
        l = np.array(l)
        l = l.reshape(1,3)
        A = np.append(A, l, axis=0)
        length += 1
        
    idx = np.random.randint(length, size=1) #To do : change 10 to data length
    Ax = A[idx, 0]
    Ay = A[idx, 1]
    Az = A[idx, 2]

    """
    Romdomly choose rotation theta from 0, 45, 90. (currently, 90 for test).
    Other roatation angle is fixed toward middle point.
    But currently, rotation is fixed for test. 
    """
    # euler angle will be strange when converting in eus program. Adjust parameter until solving this problem.  
    theta = 0#-1.54 
    phi = 0#1.2
    psi = 0
    q = tf.transformations.quaternion_from_euler(theta, phi, psi)

    posestamped = PoseStamped()
    pose = posestamped.pose
    pose.position.x = Ax
    pose.position.y = Ay
    pose.position.z = Az 
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    header = posestamped.header
    header.stamp = rospy.Time.now()
    header.frame_id = "head_mount_kinect_rgb_optical_frame"

    print("publish grasp point")

    """
    Save 
    pointcloud in boundingbox
    grasp point
    Currently save at .ros folder.
    """

    """
    with open('edge_pointcloud.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(gen) #shape(64751, 3)
    """

    grasp_posrot = np.array((Ax, Ay, Az, theta, phi, psi)).reshape(1,6) 


    """
    with open('grasp_pointcloud_pos_rot.csv', 'w') as f: 
        writer = csv.writer(f)
        writer.writerows(grasp_posrot) #shape(1, 4)?
        """
    with open("grasp_pointcloud_pos_rot.pkl", "wb") as f:
        pickle.dump(grasp_posrot, f)
        print("saved grasp point")


    pub.publish(posestamped)

if __name__=="__main__":
    #subscribe edge pointcloud data
    try:
        rospy.init_node('grasp_point_server')
        rospy.Subscriber('supervoxel_segmentation/output/cloud', PointCloud2, choose_point_callback, queue_size=10)
        pub = rospy.Publisher('/grasp_point', PoseStamped, queue_size=100)
        """while not rospy.is_shutdown():
            data = rospy.wait_for_message('supervoxel_segmentation/output/cloud', PointCloud2)
            choose_point_callback(data)
            break"""
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
