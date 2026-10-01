# g1_controller.py
"""Cartesian waypoint execution on the real G1, right arm + Inspire hand.

Unitree SDK2 has no built-in Cartesian/pose command for the arms -- only
joint-space position/velocity/torque via rt/arm_sdk (see
g1_hw_arm_hand_test.py, this repo's existing verbatim pattern, which this
file reuses rather than reinventing). So "send a Cartesian waypoint" here
means: solve it to a 17-DOF joint target with the existing DLS arm IK
(ph.kinematics.ik_two_hands_dls, pelvis frame), then ramp rt/arm_sdk's
LowCmd_ position command from the current joint state to that target over a
fixed duration -- same weight-enable / position-PD / graceful-winddown
pattern as g1_hw_arm_hand_test.py's Custom class, condensed into a reusable
controller. move_to_pose() blocks until the ramp completes, so callers get
"wait for each waypoint before proceeding" for free.

Only the right arm is driven to IK targets; the left arm holds its start
pose throughout (matches this repo's current hardware setup -- see
g1_hw_arm_hand_test.py's "LEFT SIDE DISABLED (hardware problem)" and
"the left Inspire hand is not connected at all"). Re-enable left-arm IK
targets in move_to_pose() (pass a real pL_target instead of the held one)
once that's fixed.

WARNING: drives real hardware. Same preconditions as g1_hw_arm_hand_test.py
-- clear workspace, e-stop in reach, robot confirmed in Locked Standing.
"""
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # g1_hw_arm_hand_test, hw_safety are siblings here

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from g1_hw_arm_hand_test import G1JointIndex, ARM_JOINT_NAMES, ARM_JOINTS  # noqa: E402
import hw_safety  # noqa: E402

_OUTER_REPO_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 5)))
sys.path.insert(0, os.path.join(_OUTER_REPO_ROOT, "tools"))
from inspire_hand_modbus_control import InspireHand, DEFAULT_SPEED, DEFAULT_FORCE  # noqa: E402

# ph is imported as scripts.g1_control.ph (this submodule's own convention --
# see g1_hw_ph_finger_control.py) rather than a bare top-level "ph", so it
# doesn't collide with any other "ph" on sys.path.
_PH_SUBMODULE_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PH_SUBMODULE_ROOT not in sys.path:
    sys.path.insert(0, _PH_SUBMODULE_ROOT)
from scripts.g1_control.ph import ik_two_hands_dls, fk_chain, default_urdf, IK17_JOINT_NAMES  # noqa: E402

# IK17_JOINT_NAMES (ph/robot_indices.py) and ARM_JOINT_NAMES (g1_hw_arm_hand_test.py)
# are the same 17 joints in the same waist+left+right order (the "_joint"
# suffix is the only difference) -- verified by inspection, not re-derived
# by name matching, since ARM_JOINT_NAMES has no suffix to match against.
# If either list is ever reordered independently this mapping breaks.
assert len(IK17_JOINT_NAMES) == len(ARM_JOINT_NAMES) == len(ARM_JOINTS) == 17
FINGER_ORDER = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
FINGER_LIMITS = {  # rad, lo=open hi=closed -- see g1_hw_arm_hand_test.py FINGER_LIMITS
    "pinky": (0.0, 1.7), "ring": (0.0, 1.7), "middle": (0.0, 1.7), "index": (0.0, 1.7),
    "thumb_bend": (0.0, 0.6), "thumb_rot": (-0.1, 1.3),
}
MODBUS_OPEN, MODBUS_CLOSED = 1000, 0


def _to_modbus_angle(finger: str, q_rad: float) -> int:
    lo, hi = FINGER_LIMITS[finger]
    ratio = float(np.clip((q_rad - lo) / (hi - lo), 0.0, 1.0))
    return int(round(MODBUS_OPEN - ratio * (MODBUS_OPEN - MODBUS_CLOSED)))


class G1ArmHandController:
    def __init__(self, network_interface: Optional[str] = None,
                 right_hand_ip: str = "192.168.123.211",
                 arm_kp: float = 60.0, arm_kd: float = 1.5,
                 hand_speed: int = DEFAULT_SPEED, hand_force: int = DEFAULT_FORCE,
                 control_dt: float = 0.02, enable_ramp_T: float = 2.0):
        self.network_interface = network_interface
        self.arm_kp, self.arm_kd = arm_kp, arm_kd
        self.control_dt = control_dt
        self.enable_ramp_T = enable_ramp_T

        self.urdf = default_urdf()
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.hand = InspireHand(right_hand_ip)
        self.hand_speed, self.hand_force = hand_speed, hand_force

        self.low_state: Optional[LowState_] = None
        self._have_low_state = False
        self._q17 = np.zeros(17)      # last commanded / current joint vector
        self._pL_hold = None          # left wrist target, frozen at connect()-time pose
        self._weight = 0.0

    # -- lifecycle -----------------------------------------------------
    def connect(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        if self.network_interface:
            ChannelFactoryInitialize(0, self.network_interface)
        else:
            ChannelFactoryInitialize(0)

        # Safety gate (see hw_safety.py / safety_config.json) -- do this
        # before touching arm_sdk or the hand, same order as
        # g1_hw_arm_hand_test.py.
        hw_safety.flush_stale_arm_sdk()
        hw_safety.require_locked_standing()

        self.arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_pub.Init()
        self.low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_sub.Init(self._on_low_state, 10)

        if not self.hand.connect():
            raise RuntimeError(f"failed to connect to right Inspire hand at {self.hand.ip}")
        self.hand.configure(self.hand_speed, self.hand_force)

        while not self._have_low_state:
            time.sleep(0.1)
        self._q17 = np.array([self.low_state.motor_state[j].q for j in ARM_JOINTS])
        q_by_name = dict(zip(IK17_JOINT_NAMES, self._q17))
        self._pL_hold = fk_chain(self.urdf, "pelvis", "left_wrist_yaw_link", q_by_name)[:3, 3]

        # Ramp arm_sdk weight 1 -> 0 happens implicitly via _ramp_to(); do
        # the initial enable ramp explicitly by holding q17 fixed at start.
        self._ramp_weight(0.0, 1.0, self.enable_ramp_T, hold_q=self._q17)

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg
        self._have_low_state = True

    # -- Cartesian waypoint execution -----------------------------------
    def move_to_pose(self, position: np.ndarray, R: Optional[np.ndarray] = None,
                      duration: float = 3.0) -> bool:
        """Blocks until the right wrist reaches `position` (pelvis frame,
        meters) [+ orientation R], ramping joints linearly over `duration`
        seconds. Returns whether the IK solve converged (see IKInfo) --
        always ramps to the solved joints regardless, since a
        near-converged solution is usually still a reasonable target, but
        check this return value before trusting a `grasp` waypoint."""
        q17_target, info = ik_two_hands_dls(
            pL_target=self._pL_hold, pR_target=np.asarray(position, dtype=float),
            RR_target=R, q0=self._q17,
        )
        q_start = self._q17.copy()
        n_steps = max(1, int(round(duration / self.control_dt)))
        for step in range(1, n_steps + 1):
            ratio = step / n_steps
            q_cmd = (1.0 - ratio) * q_start + ratio * q17_target
            self._publish(q_cmd, weight=1.0)
            time.sleep(self.control_dt)
        self._q17 = q17_target
        return info.converged

    def set_hand(self, opening: Dict[str, float]):
        """opening: {finger_name: radians}, missing fingers hold their last
        commanded value. Sent once -- the Inspire firmware ramps to it at
        hand_speed itself (see FORCE_SET/ANGLE_SET semantics in
        tools/inspire_hand_modbus_control.py), so this call returns
        immediately without waiting for the hand to finish moving."""
        merged = {**{f: FINGER_LIMITS[f][0] for f in FINGER_ORDER}, **opening}
        self.hand.set_angles([_to_modbus_angle(f, merged[f]) for f in FINGER_ORDER])

    def open_hand(self):
        self.set_hand({f: FINGER_LIMITS[f][0] for f in FINGER_ORDER})

    def close_hand(self, close_fraction: float = 0.9):
        self.set_hand({f: FINGER_LIMITS[f][0] + close_fraction * (FINGER_LIMITS[f][1] - FINGER_LIMITS[f][0])
                        for f in FINGER_ORDER})

    # -- low-level publish helpers --------------------------------------
    def _publish(self, q17: np.ndarray, weight: float):
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = weight
        for k, joint in enumerate(ARM_JOINTS):
            mc = self.low_cmd.motor_cmd[joint]
            mc.tau, mc.dq = 0.0, 0.0
            mc.q = float(q17[k])
            mc.kp, mc.kd = self.arm_kp, self.arm_kd
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_pub.Write(self.low_cmd)
        self._weight = weight

    def _ramp_weight(self, w0: float, w1: float, T: float, hold_q: np.ndarray):
        n_steps = max(1, int(round(T / self.control_dt)))
        for step in range(1, n_steps + 1):
            w = w0 + (w1 - w0) * (step / n_steps)
            self._publish(hold_q, weight=w)
            time.sleep(self.control_dt)

    # -- shutdown ---------------------------------------------------------
    def shutdown(self, ramp_T: float = 2.0):
        """Wind the arm_sdk weight down to 0 while holding the last
        commanded pose, then publish a final neutral command -- same
        rationale as g1_hw_arm_hand_test.Custom.Shutdown(): an instant
        weight drop hands control to the whole-body FSM's own (similarly
        stiff) arm controller from wherever the arm was, which snaps.
        Call this on every exit path, including exceptions."""
        if hasattr(self, "arm_pub"):
            self._ramp_weight(self._weight, 0.0, ramp_T, hold_q=self._q17)
            hw_safety.publish_neutral_arm_sdk(self.arm_pub)
        self.hand.close()
