# g1_hw_arm_ph_control.py
"""Bench test: PH-style gravity-comp + impedance TORQUE control for the
right arm's 7 DOF (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw),
vs. the plain position-PD every other script in this directory uses for
the arm (g1_hw_ph_finger_control.py, grasp_ph.py). Written to answer a
concrete open question: this repo's finger control (HandAdmittancePH) is
PH torque wrapped in an admittance law because the real Inspire hand
firmware genuinely has no torque input (see PIPELINE.md) -- but nothing in
this repo had ever tested whether the ARM motors accept real torque. The
`rt/arm_sdk` wire format (unitree_hg MotorCmd_: mode/q/dq/tau/kp/kd per
joint) supports it in principle; every script so far just never set `tau`
to anything but 0.

Same script, TWO control laws, switched with --mode, so a run in each mode
against the SAME segment choreography is a direct, same-logging-schema
comparison ("normal arm" vs "PH arm"):

  --mode position (the "normal arm" baseline): rt/arm_sdk position-PD,
      tau=0, kp=--arm_kp/kd=--arm_kd -- literally the same law
      g1_hw_ph_finger_control.py's arm block and grasp_ph.py's Custom use.

  --mode ph: rt/arm_sdk TORQUE feedforward,
      tau_cmd = g(q) - Kp_ph*(q - q_des) - Kd_ph*qdot,
      clamped to +-MAX_TAU_NM per joint, PLUS a low-authority PD floor
      (kp=--safety_kp/kd=--safety_kd, deliberately far below --arm_kp) sent
      alongside it on the SAME MotorCmd -- so the firmware's own position
      servo, not just software, catches a torque law that turns out to be
      wrong. g(q) is Pinocchio's generalized gravity vector for the 7 right
      arm joints (gravity_right_arm_pin/_remote, ph/pin_kinematics.py),
      computed with the waist at its LIVE measured angle (not assumed
      zero -- gravity genuinely depends on it, unlike the shoulder-frame
      FK/IK elsewhere in this repo, which only cares about relative
      geometry). Kp_ph/Kd_ph are --ph_kp/--ph_kd, PER-JOINT arrays in
      RIGHT_ARM_JOINT_NAMES order (see DEFAULT_PH_KP/KD below) --
      "position mode" of PHContactForceController's own law
      (u = g_c(q) - Kp*e - Kd*qdot), just without that class's HAND12/
      fingertip-Jacobian machinery, which doesn't apply to a 7-DOF rigid
      arm chain.

  Both modes ALWAYS use --mode position for the first ("enable-ramp") and
  last ("disable-ramp"/winddown) segments regardless of --mode -- the
  torque law only ever runs for the segments in between, once the arm_sdk
  weight is already fully ramped to 1.0. This is deliberate: the risky,
  never-before-tried part of this script is the middle segments, not the
  ramp in/out, and there's no reason to test both at once.

WHY THIS MATTERS BEYOND "does it work": if --mode ph tracks the same
targets as --mode position with comparable settle time and the arm doesn't
sag between waypoints (gravity comp working) or oscillate/runaway (gains
sane), that's the actual answer to "can this arm run under Port-Hamiltonian
control" -- see the conversation this script came out of for the fuller
context on why the hand and arm cases are NOT the same question.

Test motion: hold_in (ramp weight 0->1 at HOME) -> move_to_pose (HOME -> DEFAULT_TEST_POSE,
grasp_absolute.py's TRAJ_POSE_2 right-arm entries by default -- see --test_pose_json,
always position-PD) -> hold_torque (STAY at that pose -- the actual gravity-comp/tracking
test, --mode-controlled) -> return (back to HOME, always position-PD) -> release (ramp
weight 1->0 at HOME, always position-PD). HOME itself is whatever Start() reads as the
arm's actual current position, not a hardcoded pose -- only the hold_torque target
(DEFAULT_TEST_POSE) is a fixed absolute pose, chosen because it's already been exercised
on this hardware under plain position-PD (grasp_absolute.py), so a --mode ph run's
tracking/settling has a known-reachable target to be judged against.

WARNING: --mode ph sends a torque this repo has NEVER sent to a real G1
arm joint before. Start with the arm hanging freely (nothing holding
it back, nothing DEFAULT_TEST_POSE could hit), default gains, and the
e-stop within reach. Watch the first `hold_torque` segment closely -- if
the arm sags, the gravity term is under-driving; if it creeps/oscillates,
over-driving. MAX_TAU_NM defaults to 40% of the URDF's own <limit effort=...>
per joint (shoulder/elbow/wrist_roll: 25 Nm -> 10 Nm; wrist_pitch/yaw:
5 Nm -> 2 Nm) -- generous enough to cover gravity comp at most reachable
poses (measured ~2.2-3.4 Nm on shoulder_pitch/elbow at a hanging-ish pose,
see pin_kinematics.gravity_right_arm_pin) plus some impedance correction,
while staying well inside the hardware's own ceiling. A runaway watchdog
(--max_qdot_abort/--max_err_abort) forces an immediate graceful winddown
if the arm ever moves or deviates from target faster than a real tracking
response should.

Run (terminal 1, conda `pinocchio` env, once):
    scripts/g1_control/ph/pin_kinematics_server.py
Run (terminal 2, .venv):
    python3 g1_hw_arm_ph_control.py eth0 --mode position   # baseline
    python3 g1_hw_arm_ph_control.py eth0 --mode ph         # PH torque test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import hw_safety  # sibling of this script -- see its own docstring / safety_config.json
from log_joint_positions import G1_JOINT_NAMES  # full 29-DOF body order, for logging only

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "ph"))
import pin_kinematics_client as pik  # noqa: E402 -- talks to pin_kinematics_server.py; does NOT
                                      # import pinocchio itself, safe in .venv (see its own docstring)

LOG_DIR = os.path.join(_HERE, "grasp_logs")
STATE_LOG_HZ = 10.0


# =============================================================================
# Arm + waist joint indices (rt/arm_sdk) -- duplicated (not imported) from
# g1_hw_ph_finger_control.py's own copy, itself "verbatim from
# g1_hw_arm_hand_test.py": small and stable, and importing that file would
# pull in its pymodbus/Inspire-hand dependency chain for a script that never
# touches the hands. See pin_kinematics.py's own docstring for the repo's
# general stance on this kind of small-table duplication vs. cross-import.
# =============================================================================
class G1JointIndex:
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    kNotUsedJoint = 29  # 1: enable arm_sdk, 0: release back to whole-body FSM


ARM_JOINT_NAMES: List[str] = [
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
ARM_JOINTS: List[int] = [
    G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch,
    G1JointIndex.LeftShoulderPitch, G1JointIndex.LeftShoulderRoll, G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow, G1JointIndex.LeftWristRoll, G1JointIndex.LeftWristPitch, G1JointIndex.LeftWristYaw,
    G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll, G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow, G1JointIndex.RightWristRoll, G1JointIndex.RightWristPitch, G1JointIndex.RightWristYaw,
]

# The 7-DOF subset this script actually drives with a torque law -- same
# RIGHT_ARM_JOINT_NAMES order as grasp_ph.py / pin_kinematics.py (there
# "_joint"-suffixed, here not -- see grasp_ph.py's own comment on the two
# conventions).
RIGHT_ARM_JOINT_NAMES: List[str] = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
RIGHT_ARM_JOINTS: List[int] = [
    G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll, G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow, G1JointIndex.RightWristRoll, G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]
_RIGHT_ARM_IN_ARM = [ARM_JOINT_NAMES.index(n) for n in RIGHT_ARM_JOINT_NAMES]
_WAIST_IN_ARM = [ARM_JOINT_NAMES.index(n) for n in ("waist_yaw", "waist_roll", "waist_pitch")]

# URDF <limit effort=...> per right-arm joint (g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf) --
# the hardware's own torque ceiling, read from the same URDF pin_kinematics.py loads. Every
# other DOF (waist, left arm) never gets a nonzero tau in this script, so isn't listed here.
URDF_EFFORT_LIMIT_NM: Dict[str, float] = {
    "right_shoulder_pitch": 25.0, "right_shoulder_roll": 25.0, "right_shoulder_yaw": 25.0,
    "right_elbow": 25.0, "right_wrist_roll": 25.0, "right_wrist_pitch": 5.0, "right_wrist_yaw": 5.0,
}
DEFAULT_TAU_FRACTION = 0.4  # first-bench-test cap: 40% of the URDF's own effort limit per joint

# PH position-mode law gains (Nm/rad, Nm*s/rad), RIGHT_ARM_JOINT_NAMES order -- same role as
# PHContactForceController's Kp/Kd for the hand, sized down for an arm joint's much larger
# inertia/lever-arm than a finger's. Heavier for the shoulder/elbow (bigger links, more gravity
# torque to correct against), lighter for the wrist (light links, low URDF effort ceiling above).
# NOT bench-tuned -- same caveat as every other gain in this repo's hardware scripts.
DEFAULT_PH_KP = np.array([15.0, 15.0, 10.0, 10.0, 4.0, 3.0, 3.0])
DEFAULT_PH_KD = np.array([1.2, 1.2, 0.8, 0.8, 0.4, 0.3, 0.3])

# Low-authority PD floor sent ALONGSIDE the torque command in --mode ph (see module docstring's
# "hybrid-safe" rationale) -- deliberately far below --arm_kp/--arm_kd (60/1.5, --mode position's
# own gains), so the torque law does almost all the work and this only catches a badly wrong tau.
DEFAULT_SAFETY_KP = 3.0
DEFAULT_SAFETY_KD = 0.3

# Runaway watchdog (both modes, but only --mode ph is untested enough to likely trip it): if
# either bound is exceeded on any right-arm joint, immediately start_winddown() instead of
# continuing the segment. Generous relative to the move this script's default segments make, so
# a real tracking response never trips it, but a torque law that's actually wrong (arm swinging,
# not settling) does.
MAX_QDOT_ABORT_RAD_S = 3.0
MAX_ERR_ABORT_RAD = 0.6

# Absolute right-arm target pose for the move -> hold_torque -> return test -- TRAJ_POSE_2's
# right-arm entries (grasp_absolute.py's own TRAJ_POSE_2, waist excluded: this script is
# right-arm-only, see the module docstring's scoping note), an existing pose already exercised
# on this hardware under plain position-PD, rather than an arbitrary offset from wherever HOME
# happens to be -- so a --mode ph run's tracking/settling can be judged against a known-reachable
# target, not a guess.
DEFAULT_TEST_POSE: Dict[str, float] = {
    "right_shoulder_pitch": -0.55, "right_shoulder_roll": 0.090, "right_shoulder_yaw": 0.119,
    "right_elbow": 0.760, "right_wrist_roll": -0.003, "right_wrist_pitch": -0.493, "right_wrist_yaw": 0.024,
}


@dataclass
class Segment:
    name: str
    T: float
    law: str  # "position" | "mode" -- "mode" defers to Custom.control_mode (--mode)
    pose: Optional[Dict[str, float]] = None  # RIGHT_ARM_JOINT_NAMES subset -> ABSOLUTE rad target;
    # any joint not listed keeps HOME's value. None/{} means "target = HOME" entirely.
    # resolved once, in Custom.Start() (needs HOME, known only after the first rt/lowstate read)
    target7: np.ndarray = field(default=None, repr=False)


def build_segments(test_pose: Dict[str, float]) -> List[Segment]:
    """hold_in (always position-PD, ramp weight 0->1) -> move_to_pose (HOME -> test_pose, always
    position-PD -- a big absolute move is exactly the kind of thing NOT to make under an
    untested torque law) -> hold_torque (STAY at test_pose -- the actual gravity-comp/tracking
    test, --mode-controlled) -> return (test_pose -> HOME, always position-PD) -> release (always
    position-PD, ramp weight 1->0). Only hold_torque switches law with --mode; every other
    segment stays plain position-PD regardless (see module docstring)."""
    return [
        Segment("hold_in", 2.0, law="position", pose=None),
        Segment("move_to_pose", 4.0, law="position", pose=test_pose),
        Segment("hold_torque", 5.0, law="mode", pose=test_pose),
        Segment("return", 4.0, law="position", pose=None),
        Segment("release", 2.0, law="position", pose=None),
    ]


class Custom:
    def __init__(self, segments: List[Segment], control_mode: str, control_dt: float = 0.02,
                 arm_kp: float = 60.0, arm_kd: float = 1.5,
                 ph_kp: np.ndarray = DEFAULT_PH_KP, ph_kd: np.ndarray = DEFAULT_PH_KD,
                 safety_kp: float = DEFAULT_SAFETY_KP, safety_kd: float = DEFAULT_SAFETY_KD,
                 tau_fraction: float = DEFAULT_TAU_FRACTION,
                 max_qdot_abort: float = MAX_QDOT_ABORT_RAD_S, max_err_abort: float = MAX_ERR_ABORT_RAD,
                 shutdown_ramp_T: float = 2.0):
        assert control_mode in ("position", "ph")
        self.segments = segments
        self.control_mode = control_mode
        self.control_dt = control_dt
        self.arm_kp = arm_kp
        self.arm_kd = arm_kd
        self.ph_kp = np.asarray(ph_kp, dtype=float).reshape(7)
        self.ph_kd = np.asarray(ph_kd, dtype=float).reshape(7)
        self.safety_kp = safety_kp
        self.safety_kd = safety_kd
        self.max_tau = np.array([tau_fraction * URDF_EFFORT_LIMIT_NM[n] for n in RIGHT_ARM_JOINT_NAMES])
        self.max_qdot_abort = max_qdot_abort
        self.max_err_abort = max_err_abort

        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self._have_low_state = False

        self.t = 0.0
        self.seg_idx = 0
        self.seg_t0 = 0.0
        self.arm_q_start = np.zeros(len(ARM_JOINTS))
        self.arm_q_target = np.zeros(len(ARM_JOINTS))
        self._tick_count = 0
        self.enable_ramp_T = 2.0
        self.disable_ramp_T = 2.0
        self.done = False
        self.aborted = False

        # last commanded right-arm torque (Nm) / q target (rad), for logging + winddown-hold
        self.last_tau7 = np.zeros(7)
        self.last_g7 = np.zeros(7)
        self._q_cmd_last: Optional[np.ndarray] = None

        self.shutdown_ramp_T = float(shutdown_ramp_T)
        self._weight = 0.0
        self._winddown = False
        self._winddown_w0 = 0.0
        self._winddown_t0 = 0.0
        self._winddown_T = 2.0
        self._winddown_hold: Optional[np.ndarray] = None

    def Init(self):
        self.arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_pub.Init()
        self.low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_sub.Init(self._on_low_state, 10)

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg
        self._have_low_state = True

    def Start(self):
        if self.control_mode == "ph" and not pik.ping():
            raise pik.KinematicsServerNotRunning()

        while not self._have_low_state:
            time.sleep(0.1)
        self.arm_q_start = np.array([self.low_state.motor_state[j].q for j in ARM_JOINTS])
        self.arm_q_target = self.arm_q_start.copy()
        home7 = self.arm_q_start[_RIGHT_ARM_IN_ARM].copy()
        print(f"[INFO] right-arm HOME (measured, {self.control_mode} mode): "
              f"{dict(zip(RIGHT_ARM_JOINT_NAMES, home7.round(3)))}")

        for seg in self.segments:
            q7 = home7.copy()
            for name, v in (seg.pose or {}).items():
                q7[RIGHT_ARM_JOINT_NAMES.index(name)] = float(v)
            seg.target7 = q7

        self._load_segment(0)
        self.thread = RecurrentThread(interval=self.control_dt, target=self._tick, name="hw_arm_ph_control")
        self.thread.Start()

    def _load_segment(self, i: int):
        seg = self.segments[i]
        self.seg_idx = i
        self.seg_t0 = self.t
        self.arm_q_start = self.arm_q_target.copy()
        self.arm_q_target[_RIGHT_ARM_IN_ARM] = seg.target7
        law = "position" if seg.law == "position" else self.control_mode
        print(f"[SEG] -> {seg.name} (T={seg.T:.1f}s, law={law}, "
              f"target7={seg.target7.round(3)})")

    def start_winddown(self, T: float = 2.0):
        self._winddown_hold = (self._q_cmd_last if self._q_cmd_last is not None
                                else self.arm_q_target).copy()
        self._winddown_w0 = self._weight
        self._winddown_t0 = self.t
        self._winddown_T = max(T, 1e-6)
        self._winddown = True

    def winddown_done(self) -> bool:
        return self._winddown and self._weight <= 0.0

    def _write_position_law(self, q_arm_cmd: np.ndarray):
        for k, j in enumerate(ARM_JOINTS):
            mc = self.low_cmd.motor_cmd[j]
            mc.tau = 0.0
            mc.q = float(q_arm_cmd[k])
            mc.dq = 0.0
            mc.kp = self.arm_kp
            mc.kd = self.arm_kd

    def _tick(self):
        if self._winddown:
            ratio = float(np.clip((self.t - self._winddown_t0) / self._winddown_T, 0.0, 1.0))
            self._weight = self._winddown_w0 * (1.0 - ratio)
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = self._weight
            self._write_position_law(self._winddown_hold)  # always plain PD during winddown
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.arm_pub.Write(self.low_cmd)
            return

        if self.done:
            return

        self.t += self.control_dt
        seg = self.segments[self.seg_idx]
        seg_t = self.t - self.seg_t0
        law = "position" if seg.law == "position" else self.control_mode

        # arm_sdk weight: ramp in over the first segment, ramp out over the last
        if self.seg_idx == 0:
            w = float(np.clip(seg_t / max(self.enable_ramp_T, 1e-6), 0.0, 1.0))
        elif self.seg_idx == len(self.segments) - 1:
            w = 1.0 - float(np.clip(seg_t / max(self.disable_ramp_T, 1e-6), 0.0, 1.0))
        else:
            w = 1.0
        self._weight = w
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = w

        ratio = float(np.clip(seg_t / max(seg.T, 1e-6), 0.0, 1.0))
        q_arm_cmd = (1.0 - ratio) * self.arm_q_start + ratio * self.arm_q_target
        self._q_cmd_last = q_arm_cmd.copy()

        if law == "position":
            self._write_position_law(q_arm_cmd)
            self.last_tau7[:] = 0.0
        else:
            q7_meas = np.array([self.low_state.motor_state[j].q for j in RIGHT_ARM_JOINTS])
            qd7_meas = np.array([self.low_state.motor_state[j].dq for j in RIGHT_ARM_JOINTS])
            waist_q = np.array([self.low_state.motor_state[j].q for j in
                                 (G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch)])
            q7_des = q_arm_cmd[_RIGHT_ARM_IN_ARM]

            if np.any(np.abs(qd7_meas) > self.max_qdot_abort) or \
               np.any(np.abs(q7_meas - q7_des) > self.max_err_abort):
                print(f"[ABORT] runaway watchdog tripped -- |qdot| max {np.max(np.abs(qd7_meas)):.2f} "
                      f"rad/s (limit {self.max_qdot_abort:.2f}), |q err| max "
                      f"{np.max(np.abs(q7_meas - q7_des)):.2f} rad (limit {self.max_err_abort:.2f}). "
                      f"Starting graceful winddown -- do not re-run --mode ph until this is understood.")
                self.aborted = True
                self.start_winddown(self.shutdown_ramp_T)
                return

            g7 = pik.gravity_right_arm_remote(q7_meas, waist_q=waist_q)
            tau7 = g7 - self.ph_kp * (q7_meas - q7_des) - self.ph_kd * qd7_meas
            tau7 = np.clip(tau7, -self.max_tau, self.max_tau)
            self.last_g7 = g7
            self.last_tau7 = tau7

            # non-right-arm joints (waist, left arm): plain position-PD hold, untouched by this
            # script's torque law -- same "never touches DOF it isn't explicitly testing"
            # convention as grasp_ph.py's right-arm-only scoping.
            self._write_position_law(q_arm_cmd)
            for k, j in enumerate(RIGHT_ARM_JOINTS):
                mc = self.low_cmd.motor_cmd[j]
                mc.tau = float(tau7[k])
                mc.q = float(q7_des[k])
                mc.dq = 0.0
                mc.kp = self.safety_kp
                mc.kd = self.safety_kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_pub.Write(self.low_cmd)

        self._tick_count += 1
        if seg_t >= seg.T:
            if self.seg_idx + 1 < len(self.segments):
                self._load_segment(self.seg_idx + 1)
            else:
                print("[INFO] choreography complete.")
                self.done = True

    def Shutdown(self):
        if hasattr(self, "thread") and hasattr(self, "arm_pub"):
            self.start_winddown(self.shutdown_ramp_T)
            while not self.winddown_done():
                time.sleep(0.05)
        if hasattr(self, "thread"):
            self.thread.Wait(1.0)
        if hasattr(self, "arm_pub"):
            hw_safety.publish_neutral_arm_sdk(self.arm_pub)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("network_interface", nargs="?", default=None,
                         help="NIC on the robot's LAN (e.g. eth0). Omit to use the default.")
    parser.add_argument("--mode", choices=["position", "ph"], default="position",
                         help="'position' = normal arm_sdk position-PD baseline (tau=0). "
                              "'ph' = gravity-comp + PH-impedance TORQUE feedforward, safety-PD "
                              "floor underneath. Default 'position' -- pass --mode ph explicitly.")
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--arm_kp", type=float, default=60.0, help="--mode position gain.")
    parser.add_argument("--arm_kd", type=float, default=1.5, help="--mode position gain.")
    parser.add_argument("--ph_kp", type=float, nargs=7, default=list(DEFAULT_PH_KP),
                         metavar=tuple(RIGHT_ARM_JOINT_NAMES),
                         help="--mode ph position-mode-law Kp, Nm/rad, RIGHT_ARM_JOINT_NAMES order.")
    parser.add_argument("--ph_kd", type=float, nargs=7, default=list(DEFAULT_PH_KD),
                         metavar=tuple(RIGHT_ARM_JOINT_NAMES),
                         help="--mode ph position-mode-law Kd, Nm*s/rad, RIGHT_ARM_JOINT_NAMES order.")
    parser.add_argument("--safety_kp", type=float, default=DEFAULT_SAFETY_KP,
                         help="--mode ph low-authority PD floor sent alongside tau (see module docstring).")
    parser.add_argument("--safety_kd", type=float, default=DEFAULT_SAFETY_KD)
    parser.add_argument("--tau_fraction", type=float, default=DEFAULT_TAU_FRACTION,
                         help="Per-joint |tau| clamp as a fraction of the URDF's own <limit effort=...>.")
    parser.add_argument("--max_qdot_abort", type=float, default=MAX_QDOT_ABORT_RAD_S)
    parser.add_argument("--max_err_abort", type=float, default=MAX_ERR_ABORT_RAD)
    parser.add_argument("--test_pose_json", type=str, default=None,
                         help='JSON dict overriding DEFAULT_TEST_POSE (grasp_absolute.py\'s '
                              'TRAJ_POSE_2 right-arm entries by default) -- ABSOLUTE joint '
                              'targets in radians, e.g. \'{"right_elbow": 0.76}\'. Any joint not '
                              'listed keeps its HOME (measured-at-Start()) value. Keys are '
                              'RIGHT_ARM_JOINT_NAMES.')
    parser.add_argument("--log", action="store_true",
                         help="Log right-arm q/qdot/tau/g(q) + full 29-DOF body pose to "
                              "grasp_logs/arm_ph_<mode>_<ts>.csv at 10 Hz -- same LOG_DIR as "
                              "grasp_ph.py, run --mode position and --mode ph and diff the CSVs.")
    args = parser.parse_args()

    test_pose = DEFAULT_TEST_POSE
    if args.test_pose_json:
        test_pose = json.loads(args.test_pose_json)
        unknown = set(test_pose) - set(RIGHT_ARM_JOINT_NAMES)
        if unknown:
            print(f"[ABORT] --test_pose_json has unknown joint name(s): {unknown}")
            sys.exit(1)

    print(f"WARNING: this drives REAL arm hardware in --mode {args.mode}. " +
          ("This is the position-PD baseline (same law as every other script here) -- "
           "still real motion, keep the e-stop in reach."
           if args.mode == "position" else
           "THIS SENDS REAL TORQUE (tau != 0) TO THE ARM MOTORS -- never done from this repo "
           "before. Start with the arm hanging free of obstacles. Keep the e-stop within reach "
           "and watch the first 'hold_torque' segment closely for sag/creep/oscillation."))
    input("Press Enter to continue...")

    if args.network_interface:
        ChannelFactoryInitialize(0, args.network_interface)
    else:
        ChannelFactoryInitialize(0)

    hw_safety.flush_stale_arm_sdk()
    try:
        hw_safety.require_locked_standing()
    except hw_safety.RobotNotReadyError as e:
        print(f"[ABORT] {e}")
        sys.exit(1)

    custom = Custom(
        build_segments(test_pose), control_mode=args.mode, control_dt=args.control_dt,
        arm_kp=args.arm_kp, arm_kd=args.arm_kd,
        ph_kp=np.array(args.ph_kp), ph_kd=np.array(args.ph_kd),
        safety_kp=args.safety_kp, safety_kd=args.safety_kd, tau_fraction=args.tau_fraction,
        max_qdot_abort=args.max_qdot_abort, max_err_abort=args.max_err_abort,
    )
    custom.Init()
    custom.Start()

    log_f = None
    log_path = None
    if args.log:
        os.makedirs(LOG_DIR, exist_ok=True)
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"arm_ph_{args.mode}_{run_ts}.csv")
        log_f = open(log_path, "w", newline="")
        log_writer = csv.writer(log_f)
        # q_des/q_err: same convention as grasp_ph.py's arm_positions_*.csv -- one column pair
        # per rt/arm_sdk-controlled joint (ARM_JOINT_NAMES == G1_JOINT_NAMES[12:29], verified
        # index-for-index equal there), showing what this tick actually commanded (the
        # interpolated target, not just the segment's end goal) vs. where the joint really was.
        header = ["timestamp", "t_s", "segment", "mode"] + G1_JOINT_NAMES
        header += [f"{n}_qdes_rad" for n in ARM_JOINT_NAMES]
        header += [f"{n}_qerr_rad" for n in ARM_JOINT_NAMES]
        for name in RIGHT_ARM_JOINT_NAMES:
            header += [f"{name}_tau_cmd_Nm", f"{name}_g_Nm"]
        log_writer.writerow(header)
        print(f"[INFO] logging to {log_path}")
        last_log_t = 0.0

    try:
        while True:
            if args.log:
                now_log = time.time()
                if now_log - last_log_t >= 1.0 / STATE_LOG_HZ:
                    last_log_t = now_log
                    if custom.low_state is not None:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")
                        seg_name = custom.segments[custom.seg_idx].name if not custom.done else "done"
                        q_body = [custom.low_state.motor_state[i].q for i in range(len(G1_JOINT_NAMES))]
                        # _q_cmd_last: the actual ratio-interpolated q_des this tick's rt/arm_sdk
                        # write used (Custom._tick(), ARM_JOINT_NAMES order) -- None only before
                        # the very first tick, falls back to the segment's target for that row.
                        q_des17 = (custom._q_cmd_last if custom._q_cmd_last is not None
                                   else custom.arm_q_target)
                        q_err17 = [q_body[12 + k] - float(q_des17[k]) for k in range(17)]
                        row = [ts, f"{custom.t:.3f}", seg_name, custom.control_mode] + \
                              [f"{v:.6f}" for v in q_body] + \
                              [f"{v:.6f}" for v in q_des17] + [f"{v:.6f}" for v in q_err17]
                        for k in range(7):
                            row += [f"{custom.last_tau7[k]:.4f}", f"{custom.last_g7[k]:.4f}"]
                        log_writer.writerow(row)
                        log_f.flush()
            if custom.done or custom.aborted:
                if custom.aborted:
                    while not custom.winddown_done():
                        time.sleep(0.05)
                print("Done!" if not custom.aborted else "Stopped by runaway watchdog.")
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("[INFO] Ctrl+C -- stopping via graceful winddown.")
    finally:
        if log_f is not None:
            log_f.close()
            print(f"[INFO] log saved: {log_path}")
        custom.Shutdown()


if __name__ == "__main__":
    main()
