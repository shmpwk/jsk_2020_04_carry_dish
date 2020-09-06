import rospy
import tf
from geometry_msgs.msg import PoseStamped

rospy.init_node('hoge')
target_frame  = "/r_gripper_tool_frame"
source_frame = "/base_footprint"
source_pose = PoseStamped()
target_pose = PoseStamped()
print(source_pose)
listener = tf.TransformListener()
listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
tf = listener.transformPose(target_frame, source_pose)
print(tf)
