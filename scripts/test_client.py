#! /usr/bin/env python

import rospy
import actionlib
from behavior_tree_core.msg import *

if __name__=='__main__':

  rospy.init_node('client_test')
  client = actionlib.SimpleActionClient('action_client', BTAction)
  client.wait_for_server()

  goal = BTAction()
  client.send_goal(goal)
  client.wait_for_result(rospy.Duration.from_sec(5.0))

  result =client.get_result()
  print("result", result)

