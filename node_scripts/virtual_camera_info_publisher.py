#!/usr/bin/env python

import rospy

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2


class VirtualCameraInfoPublisher(object):

    def __init__(self):
        self.frame_id = rospy.get_param('~frame_id', 'virtual_camera_optical_frame')
        self.pub = rospy.Publisher('~output/camera_info', CameraInfo, queue_size=1)
        self.sub = rospy.Subscriber('~input', PointCloud2, self._cb, queue_size=10)

    def _cb(self, cloud_msg):
        info_msg = CameraInfo()
        info_msg.header = cloud_msg.header
        info_msg.header.frame_id = self.frame_id
        fx = 589.3664541825391
        fy = 589.3664541825391
        info_msg.height = 480
        info_msg.width = 640
        info_msg.distortion_model = "plumb_bob"
        cx = info_msg.width // 2 
        cy = info_msg.height // 2 
        info_msg.D = [1e-08, 1e-08, 1e-08, 1e-08, 1e-08]
        info_msg.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info_msg.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub.publish(info_msg)


if __name__ == '__main__':
    rospy.init_node('virtual_camera_info_publisher')
    app = VirtualCameraInfoPublisher()
    rospy.spin()
