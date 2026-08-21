"""Interactive open/close control for the G1's Inspire 6-DOF hands via rclpy.

Same DDS topics as the raw unitree_sdk2py version
(low_level/inspire_hand_open_close.py in .unitree_sdk2_python_src): publishes
MotorCmds on rt/inspire/cmd, subscribes MotorStates on rt/inspire/state. This
version goes through rclpy/rmw_cyclonedds_cpp instead of the vendored
unitree_sdk2py + pip cyclonedds bindings, so it isn't affected by the
libddsc ABI mismatch between the pip cyclonedds wheel and ROS2 Jazzy's own
CycloneDDS build.

  - 12 motors total, indices 0-5 = right hand, 6-11 = left hand.
  - Per-hand joint order: pinky, ring, middle, index, thumb_bend, thumb_rotation.
  - q is normalized: 0.0 = fully closed, 1.0 = fully open.

Run (inside the ROS2 container/workspace, after colcon build + source install/setup.bash):
  ros2 run g1_bridge inspire_hand_node
"""
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from unitree_go.msg import MotorCmd, MotorCmds, MotorStates

NUM_MOTORS = 12
RIGHT_HAND_IDX = list(range(0, 6))
LEFT_HAND_IDX = list(range(6, 12))
JOINT_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]

OPEN_Q = 1.0
CLOSE_Q = 0.0

HAND_KP = 0.6
HAND_KD = 0.05

CONTROL_DT = 0.02      # 50 Hz command rate
RAMP_SPEED = 0.6       # q units per second -- lower = gentler motion


class InspireHandNode(Node):

    def __init__(self):
        super().__init__('g1_inspire_hand_node')

        self.q_state = [0.5] * NUM_MOTORS
        self.q_target = [0.5] * NUM_MOTORS
        self.q_cmd = [0.5] * NUM_MOTORS
        self.state_received = False

        state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.cmd_pub = self.create_publisher(MotorCmds, 'inspire/cmd', 10)
        self.state_sub = self.create_subscription(
            MotorStates, 'inspire/state', self._on_state, state_qos)

        self.timer = self.create_timer(CONTROL_DT, self._write)

    def _on_state(self, msg: MotorStates):
        for i, s in enumerate(msg.states[:NUM_MOTORS]):
            self.q_state[i] = s.q
        if not self.state_received:
            self.q_cmd = list(self.q_state)
            self.q_target = list(self.q_state)
            self.state_received = True

    def _write(self):
        max_step = RAMP_SPEED * CONTROL_DT
        cmd = MotorCmds()
        for i in range(NUM_MOTORS):
            err = self.q_target[i] - self.q_cmd[i]
            step = max(-max_step, min(max_step, err))
            self.q_cmd[i] += step
            motor = MotorCmd()
            motor.mode = 1
            motor.q = self.q_cmd[i]
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = HAND_KP
            motor.kd = HAND_KD
            motor.reserve = [0, 0, 0]
            cmd.cmds.append(motor)
        self.cmd_pub.publish(cmd)

    def set_hand(self, indices, q):
        for i in indices:
            self.q_target[i] = q

    def open_hand(self, side="both"):
        if side in ("right", "both"):
            self.set_hand(RIGHT_HAND_IDX, OPEN_Q)
        if side in ("left", "both"):
            self.set_hand(LEFT_HAND_IDX, OPEN_Q)

    def close_hand(self, side="both"):
        if side in ("right", "both"):
            self.set_hand(RIGHT_HAND_IDX, CLOSE_Q)
        if side in ("left", "both"):
            self.set_hand(LEFT_HAND_IDX, CLOSE_Q)


HELP_TEXT = """
Commands:
  open           - open both hands
  close          - close both hands
  open left      - open left hand only
  open right     - open right hand only
  close left     - close left hand only
  close right    - close right hand only
  status         - print current commanded/measured q
  help           - show this message
  quit / exit    - stop and exit
"""


def main(args=None):
    rclpy.init(args=args)
    node = InspireHandNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("WARNING: Ensure fingers are clear of pinch points before running this script.")
    input("Press Enter to continue...")

    print("[INFO] Waiting for first hand state message...")
    timeout = time.time() + 5.0
    while not node.state_received and time.time() < timeout:
        time.sleep(0.05)
    if not node.state_received:
        print("[WARN] No state received on inspire/state -- "
              "starting from a neutral guess. Check the hand driver is running.")

    print(HELP_TEXT)
    try:
        while True:
            try:
                cmd = input("hand> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if cmd in ("quit", "exit"):
                break
            elif cmd == "open":
                node.open_hand("both")
            elif cmd == "close":
                node.close_hand("both")
            elif cmd == "open left":
                node.open_hand("left")
            elif cmd == "open right":
                node.open_hand("right")
            elif cmd == "close left":
                node.close_hand("left")
            elif cmd == "close right":
                node.close_hand("right")
            elif cmd == "status":
                for i in range(NUM_MOTORS):
                    side = "R" if i in RIGHT_HAND_IDX else "L"
                    name = JOINT_NAMES[i % 6]
                    print(f"  [{side}] {name:15s} q_meas={node.q_state[i]:+.3f} "
                          f"q_cmd={node.q_cmd[i]:+.3f} q_target={node.q_target[i]:+.3f}")
            elif cmd in ("help", ""):
                print(HELP_TEXT)
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for options.")
    finally:
        print("[INFO] Exiting.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
