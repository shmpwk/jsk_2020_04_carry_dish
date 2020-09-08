#!/usr/bin/env python
# refference https://answers.ros.org/question/240491/point_cloud2read_points-and-then/

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import rospy
import csv

def callback_pointcloud(data):
    assert isinstance(data, PointCloud2)
    gen = point_cloud2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)
    print type(gen)
    with open('pointcloudxyz.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(gen) #shape(64751, 3)


    #for p in gen:
    #  print p 

if __name__ == '__main__':
    rospy.init_node('pointcloud_to_csv')
    #pub = rospy.Publisher('~output', PointStamped, queue_size=1)
    sub = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, callback_pointcloud, queue_size=1)
    rospy.spin()
  
