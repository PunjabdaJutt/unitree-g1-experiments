# g1_hw_ph_finger_control.py
"""

This script runs the REAL PH control loop every tick -- same `ph/` code as
simulation, unmodified -- using the hand's own position feedback as q/qdot,
and converts the computed torque into a position setpoint via a second-
order virtual admittance law (see `HandAdmittancePH` for the exact code):

    tau_ideal   = ph_ctrl.compute(q_sim, qd_sim, dt)
    tau_contact = -contact_gain * max(|F_ACT| - force_guard_frac*FORCE_SET, 0)
    M_a*ddq_cmd + B_a*dq_cmd + K_track*(q_cmd - q_act) = tau_ideal + tau_contact
    q_cmd      += dq_cmd * dt   (rate- and range-clamped)

`tau_contact` is a SOFTWARE guard, independent of the firmware's own
FORCE_SET stall ceiling, that starts resisting further closing *before*
that ceiling is hit, using real `FORCE_ACT` feedback (now actually read
every tick, see `Custom._read_hands()`/`_tick()`). Once `|FORCE_ACT|`
reaches `FORCE_SET`, the reference itself freezes (`q_cmd := q_act`) so the
position target stops advancing past where the finger actually is --
preventing wind-up (a large `q_cmd - q_act` gap) if contact force later
drops.

This is still an APPROXIMATION of the PH law (admittance control on top of
an opaque internal position servo), not true torque control, and none of
`--admit_mass`/`--admit_damping`/`--admit_ktrack`/`--force_guard_frac` have
an analytically-correct default -- the hand's internal position-servo
bandwidth is undocumented, and the sign of `FORCE_ACT`'s "closing"
direction is unconfirmed (the guard uses `abs(FORCE_ACT)` specifically so
it doesn't depend on getting that sign right -- verify the real sign on
the bench before ever relying on a one-directional guard instead). Start
LOW and bench-tune with the e-stop in reach. This has not been run against
physical hardware from here (no physical access) -- verify offline pieces
only (see bottom of file / the plan) before trusting it near real objects.

Hand I/O: direct Modbus TCP, not DDS
--------------------------------------
g1_hw_arm_hand_test.py talks to the hands over rt/inspire/cmd / rt/inspire/state
DDS and blocks at startup until hand state arrives. But
ros2_ws/.../inspire_hand_modbus_bridge.py's own comment states rt/inspire/state
does NOT actually publish on this robot's DDS bus (that's why that bridge
polls Modbus directly instead) -- reusing the DDS hand path here would likely
hang forever. So this script talks Modbus directly for both read and write,
mirroring tools/inspire_hand_modbus_control.py's register map (reimplemented
locally, not cross-repo-imported -- this nested repo stays dependency-free of
the outer tools/ directory, same as g1_hw_arm_hand_test.py already is).

Arms: rt/arm_sdk position-PD, verbatim from g1_hw_arm_hand_test.py -- legs
stay under the robot's own locomotion FSM, this script never touches
rt/lowcmd. Default segments hold the arms at their start pose the whole demo
(safest default) -- edit `default_segments()`'s `arm_targets` to move them.

Run (network_interface is whatever NIC is on the robot's LAN, e.g. eth0):
    python scripts/g1_control/hardware/g1_hw_ph_finger_control.py eth0

WARNING: this drives real hardware. Clear the workspace and keep the e-stop
within reach before running -- there are no collision or joint-limit checks
against the physical environment, only the software clamps below.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

# hw_safety.py is a sibling of this script (scripts/g1_control/hardware/) --
# see safety_config.json for the incident (arms snapping on an FSM switch)
# this exists to prevent.
import hw_safety

from pymodbus.client import ModbusTcpClient

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.g1_control.ph import (
    parse_urdf, build_tree, PHSystem, FingertipJacobian, PHContactForceController,
    IK17_JOINT_NAMES, HAND12_NAMES, HAND12_LIMITS_LO, HAND12_LIMITS_HI,
    fk_chain, ik_two_hands_dls,
)

DEFAULT_URDF = os.path.join(
    REPO_ROOT, "unitree_g1_inspire_assets", "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")
# Copied verbatim from lift_fingerPH.py (~L756) / mujoco_common.py -- kept
# local here (not imported) so this script has no simulation-stack
# dependency (no mujoco needed on the robot's control computer).
TIP_LINK_NAMES = [
    "L_thumb_distal", "L_thumb_distal",
    "L_index_intermediate", "L_middle_intermediate", "L_ring_intermediate", "L_pinky_intermediate",
    "R_thumb_distal", "R_thumb_distal",
    "R_index_intermediate", "R_middle_intermediate", "R_ring_intermediate", "R_pinky_intermediate",
]
THUMB_YAW_IDX = (0, 6)
THUMB_PITCH_IDX = (1, 7)


# =============================================================================
# Arm + waist joint indices (rt/arm_sdk) -- copied verbatim from
# g1_hw_arm_hand_test.py, same proven pattern.
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
# ph/'s IK17_JOINT_NAMES has a "_joint" suffix (e.g. "waist_yaw_joint"),
# arm_sdk's own convention doesn't -- map by stripping it, rather than
# hand-duplicating a second 17-name list that could silently drift.
_IK17_TO_ARM_ORDER = [ARM_JOINT_NAMES.index(n[:-6]) for n in IK17_JOINT_NAMES]
# Inverse permutation: ARM_JOINT_NAMES order -> IK17_JOINT_NAMES order, for
# writing an ik_two_hands_dls solution (IK17 order) back into
# Custom.arm_q_target (ARM_JOINT_NAMES order) via plain fancy-indexing --
# see Custom._load_segment()'s Cartesian-target branch.
_ARM_TO_IK17_ORDER = [IK17_JOINT_NAMES.index(n + "_joint") for n in ARM_JOINT_NAMES]


# =============================================================================
# Inspire Hand Modbus TCP interface (register map from
# tools/inspire_hand_modbus_control.py -- RH56DFTP manual section 2.6).
# =============================================================================
MODBUS_PORT = 6000
MODBUS_DEVICE_ID = 1
ANGLE_SET = 1486
FORCE_SET = 1498
SPEED_SET = 1522
ANGLE_ACT = 1546
FORCE_ACT = 1582
ERROR = 1606

MODBUS_JOINT_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]
NUM_DOF = 6
OPEN_ANGLE = 1000  # ANGLE_SET raw units: 1000=open, 0=closed -- see InspireHandModbus.open_hand()
HAND_LOG_EVERY_TICKS = 10  # ~5 Hz at the default 0.02s control_dt

# --plot support (see __main__): same LOG_DIR/hand_state_*.csv schema and
# STATE_LOG_HZ sampling grasp_ph.py/grasp_absolute.py already use, so
# plot_hand_dynamics.py (same directory) can read whatever this run wrote
# without caring which of the three scripts produced it.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_logs")
STATE_LOG_HZ = 10.0

# Named hand poses, raw ANGLE_SET units (this file's convention: 0=closed,
# 1000=open -- see OPEN_ANGLE above, confirmed on real hardware via
# tools/inspire_hand_modbus_calibrate.py). MODBUS_JOINT_NAMES order
# (pinky, ring, middle, index, thumb_bend, thumb_rotation), L hand's 6 then
# R hand's 6 -- 12 values total, same layout as INSPIRE_POSES_SEQUENCE below
# expects.
#
# 2026-09-16: "open" and "power_grasp" were swapped here -- "open"'s values
# were mostly-0 (mostly CLOSED under this file's own convention) and
# "power_grasp"'s were mostly-800 (mostly OPEN), confirmed on a real run's
# hand_state_phdemo_*.csv log (default_segments()'s "open" segment was
# measured driving the hand from ~950-1000 raw down to ~410-450 over its
# whole window, i.e. closing, not opening). Fixed by swapping the two
# value-lists between the pose names below -- the ANGLE_SET convention
# itself was already hardware-confirmed correct, only the pose-to-values
# assignment was wrong.
INSPIRE_POSES: Dict[str, List[int]] = {
    "open": [800, 800, 800, 800, 700, 500] * 2,
    "power_grasp": [0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 500],
    "pinch": [0, 0, 0, 400, 450, 500, 0, 0, 0, 400, 450, 500],
    # Measured, not guessed: right hand (.211) holding a real cylindrical
    # object, via tools/inspire_hand_modbus_control.py's `diag` command --
    # drove each finger toward fully closed (target 0) and recorded where it
    # actually settled once it hit the object (thumb_bend=780, pinky=193,
    # ring=192, middle=168, index=181); thumb_rotation was instead driven to
    # a chosen opposition angle (target 400, settled 384 -- a position pose,
    # not a contact test). Left hand's block is a placeholder, same as
    # "open" -- no object was in the left hand, only the right-hand slice
    # here is empirically grounded. See grasp_ph.py's build_segments() for
    # where this becomes the real grasp-closing target.
    "grip_cylinder": [0, 0, 0, 0, 0, 500, 193, 192, 168, 181, 780, 384],
}

# Force-mode grasp target, right hand only (left hand's slice stays 0 Nm --
# a zero-target virtual spring back to wherever the left hand was when force
# mode was entered, not a hard position hold). NOT bench-tuned -- same
# caveat as --admit_mass/--admit_damping/--admit_ktrack/--force_guard_frac
# in __main__ below. Tested here first (default_segments()'s "grip_cylinder"
# segment) before reuse in grasp_absolute.py's build_segments(), which
# imports this constant rather than keeping a second copy.
#
# Replaces a fixed-POSITION grasp target (grip_cylinder's pose in rad
# above): that target is identical regardless of what -- or whether
# anything -- is actually between the fingers, so a grasp_logs comparison
# across a rigid object vs. two "deformable" ones showed no
# stiffness-attributable signal, just contact-timing differences from
# object size/geometry (see PHContactForceController.set_force_mode()'s
# docstring in ph_impedance.py for the underlying law). Under force mode,
# the finger's FREE-SPACE equilibrium is q_c* = q0_spr + tau_d/K_spr (K_spr
# from Custom.__init__'s K_spr_arr: 2.5 Nm/rad for every HAND12 DOF except
# thumb_pitch's 14.0) -- so the SAME uniform tau_d below settles at a
# smaller angle on thumb_pitch (its K_spr is ~5.6x stiffer) than on the
# other 5 DOF. Deliberately simple (one number, not individually derived
# per DOF) for this first bench pass -- pick something low enough that
# nothing slams into a joint limit, watch --plot's force/position trace,
# and raise it from there once it's confirmed safe.
GRASP_TAU_D = np.zeros(12)
GRASP_TAU_D = [0,0,0,0,0,0,1,2,2,2,2,2]   # R: thumb_yaw, thumb_pitch, index, middle, ring, pinky -- all 1.0 Nm

ERROR_FLAG_NAMES = ["locked-rotor", "overtemp", "overcurrent", "abnormal", "comms"]
# Manual specs FORCE_ACT as -4000..4000g (see tools/inspire_hand_modbus_control.py's
# module docstring) -- a reading outside that band is a sensor/wiring fault, not a
# real grip force.
FORCE_ACT_MAX_G = 500
# Runtime sensor-fault gate for _tick()'s force guard -- deliberately looser than
# FORCE_ACT_MAX_G above: real captures on this rig (grasp_logs/hand_state_*.csv,
# see plot_hand_dynamics.py) show thumb_yaw sitting at a genuine ~700-850g baseline
# even at rest, which the stricter 500g manual spec would itself flag as faulty --
# so this is set past that observed real range, and only catches the actual fault
# spikes seen on the same logs (5000-6000g, an order of magnitude beyond anything
# real). Same threshold plot_hand_dynamics.py uses to gate its own force subplot --
# kept as a separate constant there (not imported) so that plotting script stays
# free of this control script's heavy hardware-SDK/pymodbus import chain.
FORCE_FAULT_ABS_G = 1000.0
# FORCE_ACT/FORCE_SET are natively grams-force on the wire (the Modbus register map
# above, and hand.configure()'s FORCE_SET write, stay in grams -- that's the real
# hardware unit). HandAdmittancePH's admittance law adds tau_contact directly to
# tau_ideal (a torque, from PHContactForceController), so grams get converted to
# newtons at that boundary via this constant -- see Custom._tick() and
# Custom.__init__'s HandAdmittancePH construction.
GRAMS_TO_NEWTONS = 9.80665 / 1000.0


class InspireHandModbus:
    def __init__(self, ip: str):
        self.ip = ip
        self.client = ModbusTcpClient(ip, port=MODBUS_PORT, timeout=3)

    def connect(self) -> bool:
        return self.client.connect()

    def close(self) -> None:
        self.client.close()

    def configure(self, speed: int, force: int) -> None:
        self.client.write_registers(SPEED_SET, [speed] * NUM_DOF, device_id=MODBUS_DEVICE_ID)
        self.client.write_registers(FORCE_SET, [force] * NUM_DOF, device_id=MODBUS_DEVICE_ID)

    def set_angles(self, angles: List[int]) -> bool:
        registers = [int(a) & 0xFFFF for a in angles]
        result = self.client.write_registers(ANGLE_SET, registers, device_id=MODBUS_DEVICE_ID)
        return not result.isError()

    def open_hand(self) -> bool:
        """Direct hard-open, bypassing the PH admittance loop entirely --
        same pattern as tools/inspire_hand_modbus_control.py's own
        InspireHand.open_hand(): write ANGLE_SET=1000 (fully open) on all 6
        DOF straight to the firmware's position servo. Used once the
        segment choreography finishes, as a deterministic guarantee the
        hand actually ends up fully open -- unlike hand_mode="hold"
        (e.g. the default "release" segment), which just leaves the
        admittance q_cmd wherever tau_ideal happened to relax to, not
        necessarily fully open."""
        return self.set_angles([OPEN_ANGLE] * NUM_DOF)

    def read_angles(self) -> Optional[List[int]]:
        result = self.client.read_holding_registers(ANGLE_ACT, count=NUM_DOF, device_id=MODBUS_DEVICE_ID)
        return None if result.isError() else result.registers

    def read_forces(self) -> Optional[List[int]]:
        """Signed grams per DOF, MODBUS_JOINT_NAMES order. See FORCE_ACT_MAX_G:
        values outside +/-4000g are a sensor fault, not a real reading."""
        result = self.client.read_holding_registers(FORCE_ACT, count=NUM_DOF, device_id=MODBUS_DEVICE_ID)
        if result.isError():
            return None
        return [v - 0x10000 if v >= 0x8000 else v for v in result.registers]

    def read_errors(self) -> Optional[List[int]]:
        result = self.client.read_holding_registers(ERROR, count=3, device_id=MODBUS_DEVICE_ID)
        if result.isError():
            return None
        unpacked = []
        for v in result.registers:
            unpacked.append(v & 0xFF)
            unpacked.append((v >> 8) & 0xFF)
        return unpacked


def error_flags(byte_val: int) -> List[str]:
    return [name for i, name in enumerate(ERROR_FLAG_NAMES) if byte_val & (1 << i)]


# =============================================================================
# HAND12_NAMES <-> Modbus joint-order/unit mapping (safety-relevant -- see
# module docstring: these are two DIFFERENT orderings of the same 12 DOFs).
#   ph/robot_indices.py HAND12_NAMES: thumb_yaw, thumb_pitch, index, middle,
#     ring, pinky -- L block then R block. Units: radians (URDF limits).
#   Modbus MODBUS_JOINT_NAMES: pinky, ring, middle, index, thumb_bend,
#     thumb_rotation, per hand. Units: raw 0-1000 (1000=open, 0=closed --
#     inverted sense from "bigger radians = more closed").
# =============================================================================
_JOINT_NAME_MAP = {
    "thumb_proximal_yaw": "thumb_rotation",
    "thumb_proximal_pitch": "thumb_bend",
    "index_proximal": "index",
    "middle_proximal": "middle",
    "ring_proximal": "ring",
    "pinky_proximal": "pinky",
}


def _hand12_to_modbus_idx(hand12_name: str) -> int:
    base = hand12_name[2:-6]  # strip "L_"/"R_" prefix and "_joint" suffix
    return MODBUS_JOINT_NAMES.index(_JOINT_NAME_MAP[base])


# Per-side (6,) index arrays: HAND12_NAMES[0:6] is the L block, [6:12] is R.
HAND12_TO_MODBUS_L = [_hand12_to_modbus_idx(n) for n in HAND12_NAMES[:6]]
HAND12_TO_MODBUS_R = [_hand12_to_modbus_idx(n) for n in HAND12_NAMES[6:]]


def hand12_rad_to_raw(q_hand12: np.ndarray) -> np.ndarray:
    """(12,) radians, HAND12_NAMES order -> (12,) raw ints 0-1000, same order."""
    frac_closed = np.clip((q_hand12 - HAND12_LIMITS_LO) / (HAND12_LIMITS_HI - HAND12_LIMITS_LO), 0.0, 1.0)
    return np.clip(np.round(1000.0 * (1.0 - frac_closed)), 0, 1000).astype(int)


def hand12_raw_to_rad(raw_hand12_order: np.ndarray) -> np.ndarray:
    """(12,) raw ints, HAND12_NAMES order -> (12,) radians, same order."""
    frac_closed = 1.0 - np.asarray(raw_hand12_order, dtype=float) / 1000.0
    return HAND12_LIMITS_LO + frac_closed * (HAND12_LIMITS_HI - HAND12_LIMITS_LO)


def modbus_to_hand12_order(raw_L: List[int], raw_R: List[int]) -> np.ndarray:
    """Each hand's own (6,) Modbus-order reading -> flat (12,) HAND12_NAMES-
    order array (still raw units -- convert separately via hand12_raw_to_rad)."""
    out = np.zeros(12, dtype=float)
    for i in range(6):
        out[i] = raw_L[HAND12_TO_MODBUS_L[i]]
        out[6 + i] = raw_R[HAND12_TO_MODBUS_R[i]]
    return out


def hand12_order_to_modbus(raw_hand12_order: np.ndarray):
    """Inverse of modbus_to_hand12_order: (12,) raw ints, HAND12_NAMES order
    -> (raw_L, raw_R), each a (6,) list in Modbus order."""
    raw_L = [0] * 6
    raw_R = [0] * 6
    for i in range(6):
        raw_L[HAND12_TO_MODBUS_L[i]] = int(raw_hand12_order[i])
        raw_R[HAND12_TO_MODBUS_R[i]] = int(raw_hand12_order[6 + i])
    return raw_L, raw_R


def pose_to_hand12_rad(pose_modbus_order: List[int]) -> np.ndarray:
    """An INSPIRE_POSES entry -- (12,) raw ANGLE_SET ints, MODBUS_JOINT_NAMES
    order, L hand's 6 then R hand's 6 -- -> (12,) radians, HAND12_NAMES
    order, i.e. exactly what Segment.hand_target / ph_ctrl.q_des_hand12
    expect for hand_mode="position". Just splits the flat 12 back into
    per-hand 6-tuples and reuses the same modbus_to_hand12_order() +
    hand12_raw_to_rad() conversion _tick() already applies to real Modbus
    reads every tick -- one conversion path, not a second hand-rolled one."""
    assert len(pose_modbus_order) == 12, f"expected 12 raw values (L6+R6), got {len(pose_modbus_order)}"
    raw_L, raw_R = pose_modbus_order[:6], pose_modbus_order[6:]
    return hand12_raw_to_rad(modbus_to_hand12_order(raw_L, raw_R))


# =============================================================================
# Admittance wrapper around PHContactForceController (see module docstring
# for why this exists instead of direct torque control). Second-order
# virtual mass-spring-damper law with a software force guard -- NOT
# bench-tuned, see argparse help for every gain below and PIPELINE.md's
# Stage 2/3 tuning procedure.
# =============================================================================
class HandAdmittancePH:
    def __init__(self, ph_ctrl: PHContactForceController,
                 M_a: np.ndarray, B_a: np.ndarray, K_track: np.ndarray,
                 qdot_max: np.ndarray, force_guard_frac: float, force_limit_n: float,
                 contact_gain: float = 5.0):
        self.ph_ctrl = ph_ctrl
        self.M_a = np.asarray(M_a, dtype=float)
        self.B_a = np.asarray(B_a, dtype=float)
        self.K_track = np.asarray(K_track, dtype=float)
        self.qdot_max = np.asarray(qdot_max, dtype=float)
        self.force_limit_n = float(force_limit_n)
        self.force_guard_n = float(force_guard_frac) * self.force_limit_n
        self.contact_gain = float(contact_gain)
        self.q_cmd_hand12: Optional[np.ndarray] = None
        self.dq_cmd = np.zeros(12)

    def reset(self, q_hand12_meas: np.ndarray) -> None:
        self.q_cmd_hand12 = np.array(q_hand12_meas, dtype=float)
        self.dq_cmd[:] = 0.0

    def step(self, q_sim29: np.ndarray, qd_sim29: np.ndarray,
             f_hand12_meas_n: np.ndarray, dt: float):
        """q_sim29/qd_sim29: (29,) state in IK17_JOINT_NAMES + HAND12_NAMES
        order (arm state from rt/lowstate, hand state from Modbus -- built by
        the caller). f_hand12_meas_n: (12,) FORCE_ACT reading, HAND12 order,
        in NEWTONS -- FORCE_ACT is natively grams-force on the wire
        (InspireHandModbus.read_forces()); the caller converts via
        GRAMS_TO_NEWTONS before calling this (see Custom._tick()), so this
        stays in the same physical units as tau_ideal (a torque, from
        PHContactForceController) that tau_contact below gets added to.
        Returns (q_cmd_hand12, tau_ideal_12).

        Second-order virtual admittance:
            M_a*ddq_cmd + B_a*dq_cmd + K_track*(q_cmd - q_act)
                = tau_ideal + tau_contact
        K_track pulls the *reference* q_cmd back toward the hand's actually-
        measured q_act, so q_cmd can't silently drift away from reality if
        the real position servo lags. tau_contact is a software guard that
        starts resisting once |F_ACT| passes force_guard_frac*force_limit_n
        -- BEFORE the firmware's own FORCE_SET stall ceiling -- and the
        reference is hard-frozen to q_act (not just rate-limited) once
        |F_ACT| reaches force_limit_n, so q_cmd stops advancing past where
        the finger actually is (prevents wind-up if force later drops).
        Uses abs(F_ACT) rather than assuming a sign convention for
        "closing" direction -- read_forces()'s own docstring doesn't say
        which sign is which, so this resists in either direction rather
        than risk guarding the wrong one; confirm the real sign on the
        bench before trusting a one-directional guard instead.
        """
        if self.q_cmd_hand12 is None:
            self.reset(q_sim29[17:])

        tau_ideal_12 = self.ph_ctrl.compute(q_sim29, qd_sim29, dt=dt)

        excess_n = np.maximum(np.abs(f_hand12_meas_n) - self.force_guard_n, 0.0)
        tau_contact = -self.contact_gain * excess_n

        q_act = q_sim29[17:]
        q_err = q_act - self.q_cmd_hand12
        ddq = (tau_ideal_12 + tau_contact + self.K_track * q_err
               - self.B_a * self.dq_cmd) / self.M_a
        self.dq_cmd = self.dq_cmd + ddq * dt

        dq_step = np.clip(self.dq_cmd * dt, -self.qdot_max * dt, self.qdot_max * dt)
        self.q_cmd_hand12 = np.clip(self.q_cmd_hand12 + dq_step,
                                     HAND12_LIMITS_LO, HAND12_LIMITS_HI)

        at_limit = np.abs(f_hand12_meas_n) >= self.force_limit_n
        self.q_cmd_hand12 = np.where(at_limit, q_act, self.q_cmd_hand12)
        self.dq_cmd = np.where(at_limit, 0.0, self.dq_cmd)

        return self.q_cmd_hand12, tau_ideal_12


# =============================================================================
# Trajectory segments -- editable, same dataclass pattern as
# g1_hw_arm_hand_test.py's default_segments().
# =============================================================================
@dataclass
class Segment:
    name: str
    T: float
    arm_targets: Optional[Dict[str, float]] = None  # subset of ARM_JOINT_NAMES -> radians; omitted = hold
    # Cartesian alternative to arm_targets, pelvis frame -- solved via
    # ik_two_hands_dls at segment load time (see Custom._load_segment()).
    # arm_pR_target alone (arm_RR_target=None) solves position-only.
    # Mutually exclusive with arm_targets in practice; if both are left
    # None the arm holds its current target, same as today.
    arm_pR_target: Optional[np.ndarray] = None   # (3,) right-wrist position target
    arm_RR_target: Optional[np.ndarray] = None   # (3,3) right-wrist orientation target
    hand_mode: str = "hold"  # "hold" | "position" | "force"
    hand_target: Optional[np.ndarray] = None  # (12,) HAND12_NAMES order: rad (position) or Nm (force)


def default_segments() -> List[Segment]:
    """Hands-only demo: open -> grip_cylinder -> open -> release. "open"/
    "open_4" stay position-mode (hand_mode="position", impedance-regulated
    tracking of a fixed joint-angle pose -- fine for driving to an
    unobstructed target). "grip_cylinder" is FORCE-mode (hand_mode="force",
    hand_target=GRASP_TAU_D, a per-DOF target grip torque) rather than a
    fixed closed-pose position target -- see GRASP_TAU_D's own comment for
    why: a position target can't reflect what's actually being grasped, so
    final closure only ever depended on contact-timing/geometry, not the
    object's own stiffness. Arms hold their start pose throughout (safest
    default) -- edit `arm_targets` below to also test arm positioning;
    camera-driven reaching on real hardware is a later step, not this
    script. See INSPIRE_POSES' own comment for the raw-angle convention
    open/open_4's pose is defined in."""
    pose_open = pose_to_hand12_rad(INSPIRE_POSES["open"])
    return [
        Segment("hold", 2.0, hand_mode="hold"),
        Segment("grip_cylinder", 5.0, hand_mode="force", hand_target=GRASP_TAU_D),
        Segment("release", 2.0, hand_mode="hold"),
    ]


# ========================================================================
# Main controller
# =============================================================================
class Custom:
    def __init__(self, segments: List[Segment], hand_ips: Dict[str, str],
                 control_dt: float = 0.2, arm_kp: float = 60.0, arm_kd: float = 1.5,
                 admit_mass: float = 4.5, admit_damping: float = 10, admit_ktrack: float = 2.0,
                 force_guard_frac: float = 0.85, hand_qdot_max: float = 3.0,
                 modbus_speed: int = 450,modbus_force_limit_g: int = 700,
                 qdot_ema_alpha: float = 0.3, shutdown_ramp_T: float = 2.0,
                 use_waist: bool = True, lock_right_wrist_roll: Optional[float] = None):
        self.segments = segments
        self.control_dt = control_dt
        self.arm_kp = arm_kp
        self.arm_kd = arm_kd
        self.qdot_ema_alpha = qdot_ema_alpha
        # Same use_waist convention as lift_fingerPH.py's ik_two_hands_dls
        # call: False zeroes the waist columns of the IK Jacobian (see
        # kinematics.py), so every _load_segment() re-solve leaves the
        # waist's 3 joints exactly at whatever q0 already had -- i.e. locked
        # at wherever they were when Start() first ran, since nothing here
        # ever writes a different waist value into arm_q_target on its own.
        self.use_waist = use_waist

        # 2026-09-10: hard lock on right_wrist_roll's COMMANDED target, applied
        # unconditionally at the end of every _load_segment() call (see below)
        # regardless of which branch set arm_q_target (a baked Segment.arm_targets
        # value, a live seg.arm_pR_target IK re-solve, or neither -- left over
        # from the previous segment). This is intentionally NOT the same
        # mechanism as ik_two_hands_dls's lock_joint_names/use_waist (which only
        # constrain a value fed INTO one specific IK solve, and silently stop
        # constraining anything the moment a caller forgets to pass them, or a
        # future segment sets arm_pR_target without also passing
        # lock_joint_names to the seg.arm_pR_target branch above -- exactly the
        # gap that let wrist_roll drift). Overriding the already-computed
        # arm_q_target here instead means the lock holds no matter how that
        # value was produced, including future code paths that don't know this
        # lock exists. None (default) disables it entirely -- existing callers
        # unaffected.
        self.lock_right_wrist_roll = lock_right_wrist_roll
        self._right_wrist_roll_idx = (
            ARM_JOINT_NAMES.index("right_wrist_roll") if lock_right_wrist_roll is not None else None
        )

        self.hands: Dict[str, InspireHandModbus] = {
            side: InspireHandModbus(ip) for side, ip in hand_ips.items()
        }
        self.modbus_speed = modbus_speed
        self.modbus_force_limit_g = modbus_force_limit_g

        urdf_path = DEFAULT_URDF
        urdf_model = parse_urdf(urdf_path)
        self.urdf = urdf_model  # reused by _load_segment()'s Cartesian-target IK, not reparsed
        tree = build_tree(urdf_model)
        ph_sys = PHSystem(tree, control_joint_names=HAND12_NAMES)
        ftip_jac = FingertipJacobian(
            ph_sys=ph_sys, tip_link_names=TIP_LINK_NAMES,
            hand12_tree_idx=np.array(ph_sys.control_indices),
        )
        sim_joint_names = IK17_JOINT_NAMES + HAND12_NAMES
        ph_kp_arr = np.full(12, 12.0)
        ph_kp_arr[list(THUMB_YAW_IDX)] = 4.0
        K_spr_arr = np.full(12, 1.5)
        K_spr_arr[list(THUMB_PITCH_IDX)] = 14.0
        self.ph_ctrl = PHContactForceController(
            ph_sys=ph_sys, ftip_jac=ftip_jac, sim_joint_names=sim_joint_names,
            Kp=ph_kp_arr, Kd=np.full(12, 1.0), Ki=np.full(12, 0.1), i_clamp=0.35,
            Kf=2.0, K_spr=K_spr_arr,
        )
        self.hand_ctrl = HandAdmittancePH(
            self.ph_ctrl,
            M_a=np.full(12, admit_mass), B_a=np.full(12, admit_damping),
            K_track=np.full(12, admit_ktrack), qdot_max=np.full(12, hand_qdot_max),
            force_guard_frac=force_guard_frac,
            force_limit_n=float(modbus_force_limit_g) * GRAMS_TO_NEWTONS,
        )

        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self._have_low_state = False

        self.t = 0.0
        self.seg_idx = 0
        self.seg_t0 = 0.0
        self.arm_q_start = np.zeros(len(ARM_JOINTS))
        self.arm_q_target = np.zeros(len(ARM_JOINTS))
        self._pL_hold: Optional[np.ndarray] = None  # set in Start() from the real left-wrist FK
        self._q_hand12_prev = np.zeros(12)
        self._qd_hand12_filt = np.zeros(12)
        self._tick_count = 0
        self.enable_ramp_T = 2.0
        self.disable_ramp_T = 2.0
        self.done = False

        # Latest hand Modbus reading, stashed every _tick() purely so an
        # external read-only consumer (e.g. grasp_ph.py's state logging) can
        # see it without opening a second concurrent connection to the same
        # hand -- pymodbus's TcpClient isn't safe to hit from two threads at
        # once, and _tick() already reads these every control_dt for its own
        # admittance step. None until the first successful tick.
        self.last_raw_L: Optional[List[int]] = None
        self.last_raw_R: Optional[List[int]] = None
        self.last_f_L: Optional[List[int]] = None
        self.last_f_R: Optional[List[int]] = None

        # Graceful wind-down on Shutdown() -- same pattern as
        # g1_hw_arm_hand_test.Custom.start_winddown() (see safety_config.json's
        # graceful_winddown): ramp arm_sdk weight to 0 over shutdown_ramp_T
        # while holding the arm at its last commanded pose, instead of
        # releasing to weight=0 in one abrupt step (which hands control
        # straight to the whole-body FSM's own arm controller from wherever
        # the arm happened to be -- a hard snap if it happens mid-motion,
        # e.g. Ctrl+C during "approach"/"descend"/"lift").
        self.shutdown_ramp_T = float(shutdown_ramp_T)
        self._weight = 0.0
        self._q_cmd_last: Optional[np.ndarray] = None
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
        for side, hand in self.hands.items():
            if not hand.connect():
                raise RuntimeError(f"Failed to connect to {side} hand at {hand.ip}")
            hand.configure(self.modbus_speed, self.modbus_force_limit_g)
            print(f"[INFO] connected {side} hand at {hand.ip}")

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg
        self._have_low_state = True

    def _read_hands(self):
        raw_L = self.hands["L"].read_angles()
        raw_R = self.hands["R"].read_angles()
        f_L = self.hands["L"].read_forces()
        f_R = self.hands["R"].read_forces()
        err_L = self.hands["L"].read_errors()
        err_R = self.hands["R"].read_errors()
        return raw_L, raw_R, f_L, f_R, err_L, err_R

    def Start(self):
        while not self._have_low_state:
            time.sleep(0.1)
        self.arm_q_start = np.array([self.low_state.motor_state[j].q for j in ARM_JOINTS])
        self.arm_q_target = self.arm_q_start.copy()

        # Left wrist frozen at its real current position for the whole run
        # -- same _pL_hold pattern as g1_controller.G1ArmHandController,
        # needed by ik_two_hands_dls whenever a segment sets arm_pR_target.
        q17_start = dict(zip(IK17_JOINT_NAMES, self.arm_q_start[_IK17_TO_ARM_ORDER]))
        self._pL_hold = fk_chain(self.urdf, "pelvis", "left_wrist_yaw_link", q17_start)[:3, 3]

        raw_L, raw_R, _f_L, _f_R, err_L, err_R = self._read_hands()
        if raw_L is None or raw_R is None:
            raise RuntimeError("Failed to read initial hand angles over Modbus.")
        q_hand12_0 = hand12_raw_to_rad(modbus_to_hand12_order(raw_L, raw_R))
        self.hand_ctrl.reset(q_hand12_0)
        self._q_hand12_prev = q_hand12_0.copy()

        self._load_segment(0)
        self.thread = RecurrentThread(interval=self.control_dt, target=self._tick, name="hw_ph_finger_control")
        self.thread.Start()

    def _load_segment(self, i: int):
        seg = self.segments[i]
        self.seg_idx = i
        self.seg_t0 = self.t
        self.arm_q_start = self.arm_q_target.copy()
        if seg.arm_pR_target is not None:
            q17_target, info = ik_two_hands_dls(
                pL_target=self._pL_hold, pR_target=seg.arm_pR_target,
                RR_target=seg.arm_RR_target,
                q0=self.arm_q_target[_IK17_TO_ARM_ORDER], urdf=self.urdf,
                use_waist=self.use_waist,
                # Tell the solver itself not to spend any of the other 16
                # joints' motion compensating for a wrist_roll move that the
                # hard override below is just going to discard anyway --
                # without this, IK could converge to a solution that only
                # works WITH a wrist_roll contribution it never actually gets
                # to keep, leaving eR_p_norm/eR_w_norm worse than reported.
                lock_joint_names=(["right_wrist_roll"] if self.lock_right_wrist_roll is not None else None),
            )
            if not info.converged:
                print(f"[WARN] IK did not converge for segment '{seg.name}' "
                      f"(iters={info.iters}, errors={info.final_errors}) -- "
                      f"using best-effort solution.")
            self.arm_q_target = q17_target[_ARM_TO_IK17_ORDER]
        elif seg.arm_targets:
            for name, val in seg.arm_targets.items():
                self.arm_q_target[ARM_JOINT_NAMES.index(name)] = float(val)
        if self.lock_right_wrist_roll is not None:
            # Unconditional override -- see __init__'s comment. Applied AFTER
            # both branches above (and after the "neither" no-op case, where
            # this still simply re-affirms the same already-locked value), so
            # nothing that runs before this line can leave wrist_roll anywhere
            # else, regardless of which branch fired.
            self.arm_q_target[self._right_wrist_roll_idx] = self.lock_right_wrist_roll
        if seg.hand_mode == "position":
            self.ph_ctrl.set_position_mode()
            self.ph_ctrl.q_des_hand12 = seg.hand_target.copy()
            self.ph_ctrl.reset_integral()
        elif seg.hand_mode == "force":
            self.ph_ctrl.set_force_mode(seg.hand_target.copy())
        # "hold": leave ph_ctrl mode/target as-is -- admittance output settles
        # near its current q_cmd once tau_ideal relaxes toward 0.
        print(f"[SEG] -> {seg.name} (T={seg.T:.1f}s, hand_mode={seg.hand_mode})")

    def start_winddown(self, T: float = 2.0):
        """Freeze the arm at its last commanded pose and ramp the arm_sdk
        weight down to 0 over T seconds, instead of releasing it in one
        abrupt step. Same rationale/pattern as
        g1_hw_arm_hand_test.Custom.start_winddown(): an instant weight drop
        hands control straight to the whole-body FSM's own arm controller
        (similarly high stiffness) from whatever pose the arm happened to
        be in, which looks like a hard snap on Ctrl+C mid-motion. Call this
        before stopping the control thread; _tick() ramps self._weight -> 0
        while holding q fixed (and keeps forcing both hands open) until
        winddown_done()."""
        self._winddown_hold = (self._q_cmd_last if self._q_cmd_last is not None
                                else self.arm_q_target).copy()
        self._winddown_w0 = self._weight
        self._winddown_t0 = self.t
        self._winddown_T = max(T, 1e-6)
        self._winddown = True

    def winddown_done(self) -> bool:
        return self._winddown and self._weight <= 0.0

    def _tick(self):
        if self._winddown:
            # Keep forcing both hands open regardless of how we got here --
            # interrupted mid-run, or Shutdown() called after normal
            # completion already opened them (harmless to repeat).
            for hand in self.hands.values():
                hand.open_hand()
            ratio = float(np.clip((self.t - self._winddown_t0) / self._winddown_T, 0.0, 1.0))
            self._weight = self._winddown_w0 * (1.0 - ratio)
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = self._weight
            for k, j in enumerate(ARM_JOINTS):
                mc = self.low_cmd.motor_cmd[j]
                mc.tau = 0.0
                mc.q = float(self._winddown_hold[k])
                mc.dq = 0.0
                mc.kp = self.arm_kp
                mc.kd = self.arm_kd
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.arm_pub.Write(self.low_cmd)
            return

        if self.done:
            # Choreography already finished (see the end of this method).
            # RecurrentThread.Wait() only sets a quit flag -- it doesn't
            # stop this tick from firing again -- and __main__'s polling
            # loop only checks `done` once a second, so without this the
            # one-shot open_hand() call below would just get overwritten
            # by the next tick's admittance-driven set_angles(). Keep
            # forcing both hands open every tick until the thread actually
            # stops.
            for hand in self.hands.values():
                hand.open_hand()
            return

        self.t += self.control_dt
        seg = self.segments[self.seg_idx]
        seg_t = self.t - self.seg_t0

        raw_L, raw_R, f_L, f_R, err_L, err_R = self._read_hands()
        if raw_L is None or raw_R is None:
            print("[WARN] Modbus angle read failed this tick -- holding last command.")
            return
        self.last_raw_L, self.last_raw_R = raw_L, raw_R
        self.last_f_L, self.last_f_R = f_L, f_R

        if f_L is None or f_R is None:
            print("[WARN] Modbus force read failed this tick -- force guard disabled for this tick.")
            f_hand12_meas_n = np.zeros(12)
        else:
            f_hand12_meas_g = modbus_to_hand12_order(f_L, f_R)
            fault = np.abs(f_hand12_meas_g) > FORCE_FAULT_ABS_G
            if np.any(fault):
                # Zeroed rather than held at the last good value -- same conservative
                # choice the f_L/f_R-is-None branch above already makes: the force
                # guard treats "no trustworthy reading" as "no force", never as if a
                # fault spike were a real grip force (which would otherwise inject a
                # huge spurious tau_contact or falsely freeze q_cmd, see
                # HandAdmittancePH.step's at_limit check).
                faulty_names = [HAND12_NAMES[i] for i in np.flatnonzero(fault)]
                print(f"[WARN] FORCE_ACT fault (|reading| > {FORCE_FAULT_ABS_G:.0f}g) on "
                      f"{faulty_names} -- treating as 0g for this tick's force guard.")
                f_hand12_meas_g = np.where(fault, 0.0, f_hand12_meas_g)
            # FORCE_ACT is natively grams-force -- convert to newtons here, at the
            # hardware boundary, so everything downstream (HandAdmittancePH) stays
            # in the same physical units as tau_ideal.
            f_hand12_meas_n = f_hand12_meas_g * GRAMS_TO_NEWTONS

        self._tick_count += 1
        if self._tick_count % HAND_LOG_EVERY_TICKS == 0:
            # Raw ANGLE_ACT, MODBUS_JOINT_NAMES order, 0-1000 (1000=open, 0=closed).
            print(f"[HAND raw 0-1000, 1000=open/0=closed] "
                  f"L(pinky,ring,middle,index,thumb_bend,thumb_rot)={raw_L} "
                  f"R(pinky,ring,middle,index,thumb_bend,thumb_rot)={raw_R}")
            # Raw FORCE_ACT, MODBUS_JOINT_NAMES order, signed grams-force
            # (matches --modbus_force_limit_g's unit directly; None on a
            # failed force read this tick, distinct from a real 0g reading).
            print(f"[HAND force g] "
                  f"L(pinky,ring,middle,index,thumb_bend,thumb_rot)={f_L} "
                  f"R(pinky,ring,middle,index,thumb_bend,thumb_rot)={f_R}")
        for side, err in (("L", err_L), ("R", err_R)):
            if err is None:
                continue
            for j, byte_val in enumerate(err):
                flags = error_flags(byte_val)
                if flags:
                    print(f"[ABORT] {side} hand joint {j} ({MODBUS_JOINT_NAMES[j]}) "
                          f"error flags {flags} -- stopping.")
                    self.done = True
                    return

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
        for k, j in enumerate(ARM_JOINTS):
            mc = self.low_cmd.motor_cmd[j]
            mc.tau = 0.0
            mc.q = float(q_arm_cmd[k])
            mc.dq = 0.0
            mc.kp = self.arm_kp
            mc.kd = self.arm_kd
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_pub.Write(self.low_cmd)

        # ----- hand admittance step -----
        q_hand12_meas = hand12_raw_to_rad(modbus_to_hand12_order(raw_L, raw_R))
        qd_hand12_raw = (q_hand12_meas - self._q_hand12_prev) / self.control_dt
        self._qd_hand12_filt = (self.qdot_ema_alpha * qd_hand12_raw
                                 + (1.0 - self.qdot_ema_alpha) * self._qd_hand12_filt)
        self._q_hand12_prev = q_hand12_meas.copy()

        arm_q_meas = np.array([self.low_state.motor_state[j].q for j in ARM_JOINTS])[_IK17_TO_ARM_ORDER]
        arm_qd_meas = np.array([self.low_state.motor_state[j].dq for j in ARM_JOINTS])[_IK17_TO_ARM_ORDER]
        q_sim29 = np.concatenate([arm_q_meas, q_hand12_meas])
        qd_sim29 = np.concatenate([arm_qd_meas, self._qd_hand12_filt])

        q_cmd_hand12, tau_ideal = self.hand_ctrl.step(q_sim29, qd_sim29, f_hand12_meas_n, self.control_dt)
        raw_L_cmd, raw_R_cmd = hand12_order_to_modbus(hand12_rad_to_raw(q_cmd_hand12))
        self.hands["L"].set_angles(raw_L_cmd)
        self.hands["R"].set_angles(raw_R_cmd)

        if seg_t >= seg.T:
            if self.seg_idx + 1 < len(self.segments):
                self._load_segment(self.seg_idx + 1)
            else:
                print("[INFO] choreography complete -- forcing both hands fully open "
                      "(direct ANGLE_SET, bypassing the PH admittance loop).")
                for hand in self.hands.values():
                    hand.open_hand()
                self.done = True

    def Shutdown(self):
        """Stop the control thread and release rt/arm_sdk before exiting.

        Call this on every exit path (normal completion, Ctrl+C, exception)
        so a run interrupted mid-motion never leaves a nonzero-weight
        command live on the wire for a later mode switch to act on. See
        safety_config.json.

        First winds the arm_sdk weight down to 0 over shutdown_ramp_T
        seconds via start_winddown() (holding the arm at its last commanded
        pose, not moving it, while forcing both hands open) -- gentler than
        an abrupt weight=1.0 -> 0.0 step, especially if this fires mid-motion
        (see safety_config.json's graceful_winddown). Then, as a final
        guarantee, publishes a neutral (weight=0) sample same as before.
        """
        if hasattr(self, "thread") and hasattr(self, "arm_pub"):
            self.start_winddown(self.shutdown_ramp_T)
            while not self.winddown_done():
                time.sleep(0.05)
        if hasattr(self, "thread"):
            self.thread.Wait(1.0)
        if hasattr(self, "arm_pub"):
            hw_safety.publish_neutral_arm_sdk(self.arm_pub)
        for hand in self.hands.values():
            hand.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G1 real-hardware Inspire-hand PH admittance control + arm positioning."
    )
    parser.add_argument("network_interface", nargs="?", default=None,
                         help="NIC on the robot's LAN (e.g. eth0). Omit to use the default.")
    parser.add_argument("--left_hand_ip", type=str, default="192.168.123.210")
    parser.add_argument("--right_hand_ip", type=str, default="192.168.123.211")
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--arm_kp", type=float, default=60.0)
    parser.add_argument("--arm_kd", type=float, default=1.5)
    parser.add_argument("--admit_mass", type=float, default=4.5,
                         help="Virtual admittance inertia M_a, broadcast to all 12 HAND12 "
                              "DOF. NOT bench-tuned -- placeholder only, same undocumented "
                              "position-servo-bandwidth caveat as the old --admittance_gain.")
    parser.add_argument("--admit_damping", type=float, default=4.5,
                         help="Virtual admittance damping B_a, broadcast to all 12 DOF. "
                              "NOT bench-tuned.")
    parser.add_argument("--admit_ktrack", type=float, default=4.0,
                         help="Virtual stiffness K_track pulling q_cmd back toward the "
                              "hand's measured q_act, broadcast to all 12 DOF -- keeps the "
                              "reference from drifting away from reality if the real servo "
                              "lags. NOT bench-tuned.")
    parser.add_argument("--force_guard_frac", type=float, default=0.85,
                         help="Fraction of --modbus_force_limit_g where the SOFTWARE "
                              "admittance guard starts resisting further closing, strictly "
                              "before the firmware's own FORCE_SET stall ceiling. NOT "
                              "bench-tuned.")
    parser.add_argument("--hand_qdot_max", type=float, default=2.0,
                         help="Max finger position-setpoint rate from admittance [rad/s] -- "
                              "keep well below g1_hw_arm_hand_test.py's default (3.0) until "
                              "admittance_gain is bench-validated.")
    parser.add_argument("--modbus_speed", type=int, default=450, help="Hand internal servo speed, 0-1000.")
    parser.add_argument("--modbus_force_limit_g", type=int, default=600,
                         help="FORCE_SET safety ceiling [g] -- independent of --admittance_gain; "
                              "there is no Nm->g conversion in this codebase, set directly.")
    parser.add_argument("--plot", action="store_true",
                         help="Log hand state (same schema as grasp_ph.py/grasp_absolute.py) to "
                              "grasp_logs/hand_state_phdemo_<ts>.csv at 10 Hz while running, then "
                              "plot right-hand force/speed/direction via plot_hand_dynamics.py "
                              "once the run finishes. Off by default -- no CSV/matplotlib "
                              "overhead unless asked for.")
    args = parser.parse_args()

    print("WARNING: this drives REAL arm + hand hardware via a software admittance "
          "approximation of the PH torqu. The grip_cylinder segment is now "
          "FORCE-mode (GRASP_TAU_D, a per-DOF target grip torque) rather than a fixed "
          "position, so final finger closure will vary with whatever's actually being "
          "grasped -- that's the point, but it also hasn't been validated against a real "
          "object yet. --admit_mass/--admit_damping/--admit_ktrack/--force_guard_frac/"
          "GRASP_TAU_D have not been bench-tuned. Clear the workspace and keep the "
          "e-stop within reach.")
    input("Press Enter to continue...")

    if args.network_interface:
        ChannelFactoryInitialize(0, args.network_interface)
    else:
        ChannelFactoryInitialize(0)

    # Safety gate (safety_config.json): clear any stale rt/arm_sdk command
    # left by a previous run/crash BEFORE doing anything else, then confirm
    # the robot is actually in Locked Standing (fsm 4) before touching real
    # hardware. Do not skip or reorder these -- see the incident notes.
    hw_safety.flush_stale_arm_sdk()
    try:
        hw_safety.require_locked_standing()
    except hw_safety.RobotNotReadyError as e:
        print(f"[ABORT] {e}")
        sys.exit(1)

    custom = Custom(
        default_segments(),
        hand_ips={"L": args.left_hand_ip, "R": args.right_hand_ip},
        control_dt=args.control_dt, arm_kp=args.arm_kp, arm_kd=args.arm_kd,
        admit_mass=args.admit_mass, admit_damping=args.admit_damping,
        admit_ktrack=args.admit_ktrack, force_guard_frac=args.force_guard_frac,
        hand_qdot_max=args.hand_qdot_max,
        modbus_speed=args.modbus_speed, modbus_force_limit_g=args.modbus_force_limit_g,
    )
    custom.Init()
    custom.Start()

    hand_log_f = None
    hand_log_path = None
    if args.plot:
        os.makedirs(LOG_DIR, exist_ok=True)
        hand_log_path = os.path.join(LOG_DIR, f"hand_state_phdemo_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        hand_log_f = open(hand_log_path, "w", newline="")
        hand_log_writer = csv.writer(hand_log_f)
        hand_log_header = ["timestamp", "t_s", "segment"]
        for name in HAND12_NAMES:
            hand_log_header += [f"{name}_pos_rad", f"{name}_force_g", f"{name}_open_rad", f"{name}_close_rad"]
        hand_log_writer.writerow(hand_log_header)
        print(f"[INFO] --plot: logging hand state to {hand_log_path}")

    last_state_log_t = 0.0
    try:
        while True:
            time.sleep(1.0 / STATE_LOG_HZ if hand_log_f is not None else 1.0)
            if hand_log_f is not None:
                now_log = time.time()
                if (now_log - last_state_log_t >= 1.0 / STATE_LOG_HZ
                        and custom.last_raw_L is not None and custom.last_raw_R is not None):
                    last_state_log_t = now_log
                    pos12 = hand12_raw_to_rad(modbus_to_hand12_order(custom.last_raw_L, custom.last_raw_R))
                    if custom.last_f_L is not None and custom.last_f_R is not None:
                        force12 = modbus_to_hand12_order(custom.last_f_L, custom.last_f_R)
                    else:
                        force12 = [float("nan")] * 12
                    row = [time.strftime("%Y-%m-%d %H:%M:%S"), f"{custom.t:.3f}",
                           custom.segments[custom.seg_idx].name]
                    for i in range(12):
                        row += [f"{pos12[i]:.6f}", f"{force12[i]:.1f}",
                                f"{HAND12_LIMITS_LO[i]:.4f}", f"{HAND12_LIMITS_HI[i]:.4f}"]
                    hand_log_writer.writerow(row)
                    hand_log_f.flush()
            if custom.done:
                print("Done!")
                break
    finally:
        if hand_log_f is not None:
            hand_log_f.close()
            print(f"[INFO] hand log saved: {hand_log_path}")
        custom.Shutdown()
        if args.plot:
            try:
                import plot_hand_dynamics
                plot_hand_dynamics.plot(hand_log_path, "R")
            except Exception as e:
                print(f"[WARN] --plot: could not plot {hand_log_path}: {e}")
