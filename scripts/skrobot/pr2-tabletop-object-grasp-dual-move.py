#! /usr/bin/env python
import skrobot
import rospy
from skrobot_pr2_utils import *
import time
import numpy as np

#robot_model = skrobot.models.PR2()

#viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
#viewer.add(robot_model)

class GraspDemo(object):
    def __init__(self):
        self.robot_model = pr2_init()
        self.ri = skrobot.interfaces.ros.PR2ROSRobotInterface(self.robot_model)

    def pr2_pregrasp_pose(self):
        self.robot_model.angle_vector(np.array(
        [-6.2634935e+00, -8.4950066e+00,  3.5984734e+01,  3.1591320e+00,
          1.9961237e+02,  1.8404451e+02,  6.3117599e+00,  3.2616870e+02,
          3.5873029e+02,  1.2591792e+01,  3.6077728e+02,  3.8010056e+02,
          2.9972118e-01,  0.0000000e+00, -5.0567411e-02,  1.0113834e+00,
          1.4531954e-01, -3.1579876e-01,  3.1479496e-01, -3.4505635e-01,
         -2.2965016e+00, -1.4735943e+00, -1.9999944e+00, -3.7221497e-01,
          0.0000000e+00,  0.0000000e+00,  5.0013804e-01,  5.0013804e-01,
          5.0013804e-01,  5.0013804e-01,  8.5792243e-02,  4.6800205e-01,
          1.2305890e-01,  8.5547125e-01,  1.6163381e+00, -1.2373285e+00,
         -1.4197469e+00, -1.0793933e-01,  0.0000000e+00,  0.0000000e+00,
          4.9880299e-01,  4.9880299e-01,  4.9880299e-01,  4.9880299e-01,
          8.5585549e-02]))
        # time_scale must be 1.0
        self.ri.angle_vector(self.robot_model.angle_vector(), time=1.0, time_scale=1.0)
        print("pregrasp pose")

if __name__=='__main__':
    rospy.init_node('grasp_planner', anonymous=True)
    demo = GraspDemo()
    demo.pr2_pregrasp_pose()
    time.sleep(3)
