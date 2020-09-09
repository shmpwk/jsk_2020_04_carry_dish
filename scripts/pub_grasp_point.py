#! /usr/bin/env python

# load edge pointcloud data
# select one point
# choose rotation
# publish grasp point by service 

import rospy
import geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 

def choose_point_callback(msg):
    gen = point_cloud2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)


    point = Point()



# get depth and get grasp point 
def handle_grasp_point(req):
    #req.boxからその部分のpoint cloud を得る
    #subscribe edge pointcloud data

    try:
        rospy.Subscriber('/edgepoint2', PointCloud2, choose_point_callback, queue_size=1)

    point = choose_point()
    return GraspPointResponse(point)

def grasp_point_server():
    rospy.init_node('grasp_point_server')
    s = rospy.Service('grasp_point', GetGraspPoint, handle_grasp_point)
    print("ready to get grasp point")
    rospy.spin()

if __name__=="__main__":
    grasp_point_server()
