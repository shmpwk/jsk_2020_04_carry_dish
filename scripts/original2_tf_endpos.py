import rospy
import tf

rospy.init_node('hoge')
target_frame  = "/r_gripper_tool_frame"
source_frame = "/base_footprint"

listener = tf.TransformListener()
listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
tf = listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
print(tf)

