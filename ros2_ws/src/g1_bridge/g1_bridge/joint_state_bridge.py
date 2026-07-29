"""Republishes the G1's raw DDS LowState (rt/lowstate) as sensor_msgs/JointState.

The robot's onboard software publishes motor state directly over DDS with
Unitree's own IDL types -- it never goes through rclpy, so it never carries
proper ROS graph info and robot_state_publisher/RViz can't consume it
directly. This node is the missing link: subscribe to /lowstate, pull the
first 29 motor_state entries (indices 0-28 match G1JointIndex from the SDK
examples, in the same order as the joints in g1_description's URDF), and
publish a standard JointState that robot_state_publisher understands.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import JointState
from unitree_hg.msg import LowState

# Order must match motor_state[0..28] (G1JointIndex) and the revolute joints
# in g1_description/urdf/g1_29dof_with_inspire_hand.urdf.
JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]


class JointStateBridge(Node):

    def __init__(self):
        super().__init__('g1_joint_state_bridge')

        # The robot's DDS publisher is best-effort; a RELIABLE subscriber
        # would silently never connect to it.
        lowstate_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.lowstate_sub = self.create_subscription(
            LowState, 'lowstate', self.on_lowstate, lowstate_qos)

        self.get_logger().info(
            f'Bridging /lowstate -> /joint_states for {len(JOINT_NAMES)} joints')

    def on_lowstate(self, msg: LowState):
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = JOINT_NAMES
        out.position = [msg.motor_state[i].q for i in range(len(JOINT_NAMES))]
        out.velocity = [msg.motor_state[i].dq for i in range(len(JOINT_NAMES))]
        out.effort = [msg.motor_state[i].tau_est for i in range(len(JOINT_NAMES))]
        self.joint_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
