#!/usr/bin/env python
# refference https://answers.ros.org/question/240491/point_cloud2read_points-and-then/

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import rospy
import csv
import random
import numpy as np

def choose_point(xyz):
    #print(random.choice(xyz))
    print(xyz)


def callback_pointcloud(data):
    assert isinstance(data, PointCloud2)
    gen = point_cloud2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)
    data_line = 0
    whole_data1 = gen    
    whole_data2 = gen
    """ 
    for l in whole_data1:
        data_line += 1"""
    A = np.arange(3).reshape(1,3)
    for ll in whole_data2:
        ll  = np.array(ll)
        ll = ll.reshape(1,3)
        A = np.append(A, ll, axis=0)
    #A = np.array(gen)
    idx = np.random.randint(100, size=1)
    print(A[idx, :])
    print(A.shape)

    #choose_point(list(gen.next()))
    #print(list(gen.next()))
    """
    with open('pointcloudxyz.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(gen) #shape(64751, 3)
    """
    with open('one_pointcloudxyz.csv', 'w') as ff: 
        writerf = csv.writer(ff)
        writerf.writerows(A[idx, :]) #shape(1, 3)?    
 
if __name__ == '__main__':
    rospy.init_node('pointcloud_to_csv')
    #pub = rospy.Publisher('~output', PointStamped, queue_size=1)
    sub_once = None 
    #sub = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, callback_pointcloud, sub_once)

    while not rospy.is_shutdown():
        data = rospy.wait_for_message('organized_edge_detector/output', PointCloud2)
        callback_pointcloud(data)
        break
    rospy.spin()
  
