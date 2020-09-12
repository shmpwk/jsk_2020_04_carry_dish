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

def choose_point_callback(data):
    assert isinstance(data, PointCloud2)
    gen = point_cloud2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)

    pose = Pose()
    pose.position.x = 5
    pose.position.y = 4
    pose.position.z = 3
    pose.orientation.x = 1
    pose.orientation.y = 1
    pose.orientation.z = 1
    pose.orientation.w = 1
    
    pub.publish(pose)

    

# get depth and get grasp point 
if __name__=="__main__":
    #subscribe edge pointcloud data
    try:
        rospy.init_node('grasp_point_server')
        rospy.Subscriber('/camera/depth_registered/points', PointCloud2, choose_point_callback, queue_size=1)
        pub = rospy.Publisher('/grasp_point', Pose, queue_size=1)
        rospy.spin()
    except rospy.ROSInterruptException: pass

def grasp_point_server():
    rospy.init_node('grasp_point_server')
    s = rospy.Service('grasp_point', GetGraspPoint, handle_grasp_point)
    print("ready to get grasp point")
    rospy.spin()

#if __name__=="__main__":
#    #grasp_point_server()
#    handle_grasp_point()
