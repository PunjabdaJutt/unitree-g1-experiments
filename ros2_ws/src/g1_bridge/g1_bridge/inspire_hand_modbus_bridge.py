"""Publishes the G1's Inspire hand joint angles onto /joint_states, read
directly from the hands over Modbus TCP instead of DDS.

On this robot the Inspire hands don't appear on the robot's own DDS bus at
all (rt/inspire/state never publishes) -- see tools/inspire_hand_modbus_read.py
for the full network writeup. They're reachable directly as two Modbus TCP
devices on the robot's 192.168.123.0/24 network, factory host octets .210
and .211, port 6000. This node polls ANGLE_ACT (register 1546, 6 shorts,
0=closed/1000=open) from both, expands each hand's 6 motor readings into
its 12 URDF joint angles (inspire_hand_kinematics.py), and publishes them
as a normal sensor_msgs/JointState on the same 'joint_states' topic
joint_state_bridge.py publishes the 29 body joints to -- robot_state_publisher
and mujoco_mirror both already consume that topic and merge by joint name,
so this is the only piece needed to make hand state show up in RViz/MuJoCo.

Confirmed by observation against the physical robot: .210 is the anatomical
LEFT hand, .211 is RIGHT (the reverse of the factory host-octet order they're
usually listed in elsewhere in this repo, e.g. inspire_hand_modbus_control.py's
--hands default). Override via `--ros-args -p left_hand_ip:=... -p right_hand_ip:=...`
if this ever changes (e.g. a hand gets swapped/re-cabled).

Run (inside the ROS2 container, after colcon build + source install/setup.bash):
  ros2 run g1_bridge inspire_hand_modbus_bridge
"""
import rclpy
from pymodbus.client import ModbusTcpClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

from g1_bridge.inspire_hand_kinematics import expand_hand_angles

PORT = 6000
DEVICE_ID = 1
ANGLE_ACT = 1546
NUM_DOF = 6
POLL_DT = 0.05  # 20 Hz -- plenty for visualization, gentle on the embedded Modbus server


class InspireHandModbusBridge(Node):

    def __init__(self):
        super().__init__('g1_inspire_hand_modbus_bridge')

        self.declare_parameter('right_hand_ip', '192.168.123.211')
        self.declare_parameter('left_hand_ip', '192.168.123.210')
        right_ip = self.get_parameter('right_hand_ip').value
        left_ip = self.get_parameter('left_hand_ip').value

        self._hands = {
            'right': ModbusTcpClient(right_ip, port=PORT, timeout=1),
            'left': ModbusTcpClient(left_ip, port=PORT, timeout=1),
        }
        self._connected = {'right': False, 'left': False}

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.create_timer(POLL_DT, self._poll)

        self.get_logger().info(
            f"Polling Inspire hands over Modbus TCP: right={right_ip} left={left_ip}")

    def _ensure_connected(self, side: str) -> bool:
        client = self._hands[side]
        if client.connected:
            return True
        ok = client.connect()
        if ok != self._connected[side]:
            level = self.get_logger().info if ok else self.get_logger().warn
            level(f"{side} hand ({client.comm_params.host}) "
                  f"{'connected' if ok else 'not reachable'}")
            self._connected[side] = ok
        return ok

    def _read_hand(self, side: str):
        if not self._ensure_connected(side):
            return None
        client = self._hands[side]
        result = client.read_holding_registers(ANGLE_ACT, count=NUM_DOF, device_id=DEVICE_ID)
        if result.isError():
            client.close()
            self._connected[side] = False
            return None
        return [v / 1000.0 for v in result.registers]

    def _poll(self):
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        for side in ('right', 'left'):
            motor_qs = self._read_hand(side)
            if motor_qs is None:
                continue
            for name, angle in expand_hand_angles(side, motor_qs).items():
                out.name.append(name)
                out.position.append(angle)
        if out.name:
            self.joint_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = InspireHandModbusBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for client in node._hands.values():
            client.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
