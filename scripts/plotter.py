#! /usr/bin/env python
try:
  from jsk_rviz_plugins.msg import *
except:
  import roslib;roslib.load_manifest("jsk_rviz_plugins")
  from jsk_rviz_plugins.msg import *

from geometry_msgs.msg import *
from std_msgs.msg import ColorRGBA, Float32
import rospy
import message_filters
import datetime
import pickle

pre_val1 = 0 
pre_val2 = 0
pre_val3 = 0
pre_del_val1 = 0
pre_del_val2 = 0
pre_del_val3 = 0
prepre_del_val1 = 0
prepre_del_val2 = 0
prepre_del_val3 = 0
preprepre_del_val1 = 0
preprepre_del_val2 = 0
preprepre_del_val3 = 0
pre_val4 = 0 
pre_val5 = 0
pre_val6 = 0
pre_del_val4 = 0
pre_del_val5 = 0
pre_del_val6 = 0
prepre_del_val4 = 0
prepre_del_val5 = 0
prepre_del_val6 = 0
preprepre_del_val4 = 0
preprepre_del_val5 = 0
preprepre_del_val6 = 0


def plot_cb (msg1, msg2):
    val1 = float(msg1.wrench.force.x) #sub msg is float64 but pub msg is float32
    val2 = float(msg1.wrench.force.y) #sub msg is float64 but pub msg is float32
    val3 = float(msg1.wrench.force.z) #sub msg is float64 but pub msg is float32
    val4 = float(msg2.wrench.force.x) #sub msg is float64 but pub msg is float32
    val5 = float(msg2.wrench.force.y) #sub msg is float64 but pub msg is float32
    val6 = float(msg2.wrench.force.z) #sub msg is float64 but pub msg is float32
    global pre_val1
    global pre_val2
    global pre_val3
    global pre_del_val1
    global pre_del_val2
    global pre_del_val3
    global prepre_del_val1
    global prepre_del_val2
    global prepre_del_val3
    global preprepre_del_val1
    global preprepre_del_val2
    global preprepre_del_val3
    global pre_val4
    global pre_val5
    global pre_val6
    global pre_del_val4
    global pre_del_val5
    global pre_del_val6
    global prepre_del_val4
    global prepre_del_val5
    global prepre_del_val6
    global preprepre_del_val4
    global preprepre_del_val5
    global preprepre_del_val6


    del_val1 = (val1 - pre_val1)*10
    del_val2 = (val2 - pre_val2)*10
    del_val3 = (val3 - pre_val3)*10
    pub_val1 = ((del_val1 + pre_del_val1 + prepre_del_val1 + preprepre_del_val1) / 4)
    pub_val2 = ((del_val2 + pre_del_val2 + prepre_del_val2 + preprepre_del_val2) / 4)
    pub_val3 = ((del_val3 + pre_del_val3 + prepre_del_val3 + preprepre_del_val3) / 4)
    pre_val1 = val1
    pre_val2 = val2
    pre_val3 = val3
    pre_del_val1 = del_val1
    pre_del_val2 = del_val2
    pre_del_val3 = del_val3
    prepre_del_val1 = pre_del_val1
    prepre_del_val2 = pre_del_val2
    prepre_del_val3 = pre_del_val3
    preprepre_del_val1 = prepre_del_val1
    preprepre_del_val2 = prepre_del_val2
    preprepre_del_val3 = prepre_del_val3
    value1_pub.publish(pub_val1)
    value2_pub.publish(pub_val2)
    value3_pub.publish(pub_val3)
 
    del_val4 = (val4 - pre_val4)*10
    del_val5 = (val5 - pre_val5)*10
    del_val6 = (val6 - pre_val6)*10
    pub_val4 = ((del_val4 + pre_del_val4 + prepre_del_val4 + preprepre_del_val4) / 4)
    pub_val5 = ((del_val5 + pre_del_val5 + prepre_del_val5 + preprepre_del_val5) / 4)
    pub_val6 = ((del_val6 + pre_del_val6 + prepre_del_val6 + preprepre_del_val6) / 4)
    pre_val4 = val4
    pre_val5 = val5
    pre_val6 = val6
    pre_del_val4 = del_val4
    pre_del_val5 = del_val5
    pre_del_val6 = del_val6
    prepre_del_val4 = pre_del_val4
    prepre_del_val5 = pre_del_val5
    prepre_del_val6 = pre_del_val5
    preprepre_del_val4 = prepre_del_val4
    preprepre_del_val5 = prepre_del_val5
    preprepre_del_val6 = prepre_del_val6
    value4_pub.publish(pub_val4)
    value5_pub.publish(pub_val5)
    value6_pub.publish(pub_val6)
 
    print(pub_val1, pub_val2, pub_val3)
    print(pub_val4, pub_val5, pub_val6)
    now = datetime.datetime.now()
    #filename = 'Data/' + now.strftime('%Y%m%d_%H%M%S') + '.pkl'
    #with open(filename, 'wb') as f:
    #    pickle.dump(pub_val1, f) #datasize=480

if __name__ == '__main__':
    rospy.init_node("plotter_sample")
    value1_pub = rospy.Publisher("value1_load", Float32, queue_size=1)
    value2_pub = rospy.Publisher("value2_load", Float32, queue_size=1)
    value3_pub = rospy.Publisher("value3_load", Float32, queue_size=1)
    value4_pub = rospy.Publisher("value4_load", Float32, queue_size=1)
    value5_pub = rospy.Publisher("value5_load", Float32, queue_size=1)
    value6_pub = rospy.Publisher("value6_load", Float32, queue_size=1)
    sub1 = message_filters.Subscriber('/right_endeffector/wrench', WrenchStamped)
    sub2 = message_filters.Subscriber('/left_endeffector/wrench', WrenchStamped)
    #sub3 = message_filters.Subscriber('/rgripper/finger3_joint_controller/state', JointState)
    delay = 1 / 30
    ts = message_filters.ApproximateTimeSynchronizer([sub1,sub2], 100, 0.1)
    ts.registerCallback(plot_cb)
    #rospy.Subscriber('/lgripper/finger1_joint_controller/state',  JointState, plot_cb)
    rospy.spin()

