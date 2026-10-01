# g1_hw_finger_close_only.py
"""Real-hardware, HANDS-ONLY smoke test: close the Inspire-hand fingers using
the same PH IDA-PBC controller as lift_fingerPH.py / mujoco_lift_finger.py,
then reopen. No arm/waist motion at all -- this is the minimal first step
before running the full arm+hand demo in g1_hw_ph_finger_control.py.

Why hands-only, and why this exists as a separate script
----------------------------------------------------------
g1_hw_ph_finger_control.py already runs the full PH admittance loop on real
hardware, but every one of its segments also drives rt/arm_sdk (even its
"hold" segments keep arm_sdk enabled to hold the arms at their start pose).
That's a reasonable default for a pick demo, but it is NOT the minimal first
test: arm_sdk enable/disable and the arm position-PD loop are a second
untested code path layered on top of the also-untested hand admittance loop.
This script removes that variable entirely -- it never publishes to
rt/arm_sdk or rt/lowcmd, so the arms/legs stay under the robot's own
locomotion FSM the whole time, identical to a hand sitting on the desk. Only
the 12 Inspire-hand finger joints move.

It DOES subscribe (read-only) to rt/lowstate for the current waist/arm
angles, purely so the PH controller's gravity-compensation term
(gravity_torques_hand12 -- depends on the whole kinematic chain up to the
fingers, not just the fingers) uses the robot's actual pose instead of
assuming zero. This is a passive subscription; it cannot actuate anything.
If rt/lowstate doesn't arrive within --lowstate_timeout, the script proceeds
anyway with a zero-pose approximation (printed as a warning) rather than
blocking forever -- gravity-comp error from that approximation only biases
the admittance controller's steady state slightly, it does not change what
the fingers are able to do, since position-mode Kp/Ki still close the loop.

Hand I/O: direct Modbus TCP (same reasoning as g1_hw_ph_finger_control.py)
----------------------------------------------------------------------------
Reuses that script's InspireHandModbus client and HAND12<->Modbus register
mapping verbatim (imported, not reimplemented -- both files live in this
same hardware/ directory and already require pymodbus, so there's no
cross-package dependency being added by sharing them). See that file's
module docstring for why Modbus TCP is used instead of the DDS
rt/inspire/cmd / rt/inspire/state path the dfx_inspire_service middleware
exposes: rt/inspire/state does not actually publish on this robot (confirmed
in ros2_ws/.../inspire_hand_modbus_bridge.py and tools/inspire_hand_modbus_read.py),
so the DDS-based deployment workflow described in
https://dev.classmethod.jp/articles/unitree-g1-inspire-hand-demo-deploy/ (SSH
to the dev PC, start dfx_inspire_service, publish to rt/inspire/cmd) would
hang waiting for hand state that never arrives on this particular robot.
What IS reused from that article here: the network setup (Ethernet to G1's
rear port, dev PC at 192.168.123.164, your PC on the same /24), and its
safety pattern (explicit "clear the area" confirmation before any motion).
Hand IPs (192.168.123.210 / .211) and the register map match
tools/inspire_hand_modbus_control.py, already confirmed reachable on this
robot (ERROR=0, ~35-40C, all DOFs reachable per that script's own module
docstring).

This is still an ADMITTANCE approximation of the PH torque law, not true
torque control -- see g1_hw_ph_finger_control.py's module docstring for the
full explanation of why (the hand firmware only has a position servo).
--admittance_gain has not been bench-tuned; start low.

Run:
    python scripts/g1_control/hardware/g1_hw_finger_close_only.py eth0
    python scripts/g1_control/hardware/g1_hw_finger_close_only.py eth0 --dry_run   # connect + read only, no motion
    python scripts/g1_control/hardware/g1_hw_finger_close_only.py eth0 --close_fraction 0.3 --stay_closed[p]

WARNING: this drives real hand hardware. Clear the workspace (fingers away
from pinch points) and keep the e-stop within reach before running.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.thread import RecurrentThread

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.g1_control.ph import (
    parse_urdf, build_tree, PHSystem, FingertipJacobian, PHContactForceController,
    IK17_JOINT_NAMES, HAND12_NAMES, HAND12_LIMITS_LO, HAND12_LIMITS_HI,
)
from scripts.g1_control.hardware.g1_hw_ph_finger_control import (
    InspireHandModbus, HandAdmittancePH, error_flags,
    hand12_rad_to_raw, hand12_raw_to_rad,
    modbus_to_hand12_order, hand12_order_to_modbus,
    ARM_JOINT_NAMES, ARM_JOINTS, _IK17_TO_ARM_ORDER,
    THUMB_YAW_IDX, THUMB_PITCH_IDX, DEFAULT_URDF, TIP_LINK_NAMES,
    MODBUS_JOINT_NAMES, FORCE_ACT_MAX_G,
)


def print_hand_status(ip: str, angles: List[int], forces: Optional[List[int]],
                       errors: Optional[List[int]]) -> None:
    """Per-finger status line, same layout as tools/inspire_hand_modbus_control.py's
    'status' command: raw Modbus angle (0-1000, 1000=open) + FORCE_ACT (signed
    grams) + decoded error flags, one line per joint, in MODBUS_JOINT_NAMES order."""
    print(f"  [{ip}]")
    for i, joint in enumerate(MODBUS_JOINT_NAMES):
        angle_str = f"{angles[i]:4d}" if angles is not None else "  ??"
        if forces is None:
            force_str = "   ??g"
        else:
            force_str = f"{forces[i]:5d}g"
        flags = error_flags(errors[i]) if errors is not None else []
        flag_str = f"  ERROR: {','.join(flags)}" if flags else ""
        fault_str = ""
        if forces is not None and abs(forces[i]) > FORCE_ACT_MAX_G:
            fault_str = (f"  !!! FORCE_ACT {forces[i]}g exceeds sensor spec "
                         f"(+/-{FORCE_ACT_MAX_G}g) -- SENSOR FAULT SUSPECTED !!!")
        print(f"      {joint:15s} angle={angle_str}  force={force_str}{flag_str}{fault_str}")


# =============================================================================
# Segments -- hand-only subset of g1_hw_ph_finger_control.py's Segment
# (no arm_targets: this script never touches rt/arm_sdk).
# =============================================================================
@dataclass
class Segment:
    name: str
    T: float
    hand_mode: str = "hold"  # "hold" | "position"
    hand_target: Optional[np.ndarray] = None  # (12,) HAND12_NAMES order, radians


def default_segments(close_fraction: float, stay_closed: bool) -> List[Segment]:
    """hold (settle + sanity-check reads) -> close -> hold_closed -> [open ->
    release], unless --stay_closed. Thumb-yaw (HAND12 idx 0,6) is held at 0
    throughout -- it's orientation-only, closing it isn't "closing the
    fingers" and it has the least torque margin (see HAND12_LIMITS_HI[0]=1.3
    over a 1 Nm actuator, same caveat as lift_fingerPH.py's --ph_kp_thumb_yaw)."""
    open_q = HAND12_LIMITS_LO.copy()
    close_q = HAND12_LIMITS_LO + close_fraction * (HAND12_LIMITS_HI - HAND12_LIMITS_LO)
    close_q[[0, 6]] = 0.0  # thumb-yaw: hold, don't close

    segs = [
        Segment("hold", 1.5, hand_mode="hold"),
        Segment("close", 3.0, hand_mode="position", hand_target=close_q),
        Segment("hold_closed", 3.0, hand_mode="hold"),
    ]
    if not stay_closed:
        segs.append(Segment("open", 2.5, hand_mode="position", hand_target=open_q))
        segs.append(Segment("release", 1.5, hand_mode="hold"))
    return segs


# =============================================================================
# Main controller
# =============================================================================
class Custom:
    def __init__(self, segments: List[Segment], hand_ips: Dict[str, str],
                 control_dt: float = 0.1, admittance_gain: float = 0.60,
                 hand_qdot_max: float = 1.0, modbus_speed: int = 250,
                 modbus_force_limit_g: int = 700, qdot_ema_alpha: float = 0.3,
                 lowstate_timeout: float = 3.0, sample_every_s: float = 0.1,
                 plot_window_s: float = 20.0, console_print: bool = True):
        self.segments = segments
        self.control_dt = control_dt
        self.qdot_ema_alpha = qdot_ema_alpha
        self.lowstate_timeout = lowstate_timeout
        self.sample_every_s = sample_every_s
        self.console_print = console_print
        self._last_sample_t = -1e9

        # Live history buffers for run_live_plot() -- fixed-length deques, one
        # sample every `sample_every_s`, covering the trailing `plot_window_s`
        # seconds. Appended from the control thread (_tick), read from the
        # main thread's plot loop; plain deque append/snapshot is safe enough
        # for a status display under the GIL, no lock needed.
        hist_len = max(10, int(plot_window_s / max(sample_every_s, 1e-6)) + 5)
        self.hist_t: deque = deque(maxlen=hist_len)
        self._hist_keys: List[Tuple[str, str]] = [
            (side, joint) for side in ("L", "R") for joint in MODBUS_JOINT_NAMES
        ]
        self.hist_angle: Dict[Tuple[str, str], deque] = {
            k: deque(maxlen=hist_len) for k in self._hist_keys
        }
        self.hist_force: Dict[Tuple[str, str], deque] = {
            k: deque(maxlen=hist_len) for k in self._hist_keys
        }

        self.hands: Dict[str, InspireHandModbus] = {
            side: InspireHandModbus(ip) for side, ip in hand_ips.items()
        }
        self.modbus_speed = modbus_speed
        self.modbus_force_limit_g = modbus_force_limit_g

        urdf_model = parse_urdf(DEFAULT_URDF)
        tree = build_tree(urdf_model)
        ph_sys = PHSystem(tree, control_joint_names=HAND12_NAMES)
        ftip_jac = FingertipJacobian(
            ph_sys=ph_sys, tip_link_names=TIP_LINK_NAMES,
            hand12_tree_idx=np.array(ph_sys.control_indices),
        )
        sim_joint_names = IK17_JOINT_NAMES + HAND12_NAMES
        ph_kp_arr = np.full(12, 4.0)
        ph_kp_arr[list(THUMB_YAW_IDX)] = 1.0
        K_spr_arr = np.full(12, 2.5)
        K_spr_arr[list(THUMB_PITCH_IDX)] = 14.0
        self.ph_ctrl = PHContactForceController(
            ph_sys=ph_sys, ftip_jac=ftip_jac, sim_joint_names=sim_joint_names,
            Kp=ph_kp_arr, Kd=np.full(12, 1.0), Ki=np.full(12, 0.1), i_clamp=0.35,
            Kf=1.0, K_spr=K_spr_arr,  # Kf/K_spr unused in position mode but must be >0
        )
        self.hand_ctrl = HandAdmittancePH(self.ph_ctrl, admittance_gain, qdot_max=hand_qdot_max)

        self.low_state: Optional[LowState_] = None
        self._have_low_state = False

        self.t = 0.0
        self.seg_idx = 0
        self.seg_t0 = 0.0
        self._q_hand12_prev = np.zeros(12)
        self._qd_hand12_filt = np.zeros(12)
        self.done = False

    def Init(self):
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

    def _record_history(self, raw_L: List[int], raw_R: List[int],
                         force_L: Optional[List[int]], force_R: Optional[List[int]]) -> None:
        self.hist_t.append(self.t)
        for i, joint in enumerate(MODBUS_JOINT_NAMES):
            self.hist_angle[("L", joint)].append(raw_L[i])
            self.hist_angle[("R", joint)].append(raw_R[i])
            self.hist_force[("L", joint)].append(force_L[i] if force_L is not None else np.nan)
            self.hist_force[("R", joint)].append(force_R[i] if force_R is not None else np.nan)

    def _read_hands(self):
        raw_L = self.hands["L"].read_angles()
        raw_R = self.hands["R"].read_angles()
        err_L = self.hands["L"].read_errors()
        err_R = self.hands["R"].read_errors()
        return raw_L, raw_R, err_L, err_R

    def read_only_report(self):
        """--dry_run: connect + read once + print, never write a command."""
        raw_L, raw_R, err_L, err_R = self._read_hands()
        if raw_L is None or raw_R is None:
            print("[FAIL] Could not read hand angles over Modbus.")
            return
        force_L = self.hands["L"].read_forces()
        force_R = self.hands["R"].read_forces()
        print("[DRY_RUN] per-finger status:")
        print_hand_status(self.hands["L"].ip, raw_L, force_L, err_L)
        print_hand_status(self.hands["R"].ip, raw_R, force_R, err_R)
        q = hand12_raw_to_rad(modbus_to_hand12_order(raw_L, raw_R))
        print(f"[DRY_RUN] HAND12 (rad) = {np.round(q, 3).tolist()}")

    def Start(self):
        t_wait0 = time.monotonic()
        while not self._have_low_state and (time.monotonic() - t_wait0) < self.lowstate_timeout:
            time.sleep(0.05)
        if self._have_low_state:
            arm_q0 = np.array([self.low_state.motor_state[j].q for j in ARM_JOINTS])[_IK17_TO_ARM_ORDER]
        else:
            print("[WARN] rt/lowstate not received within "
                  f"{self.lowstate_timeout:.1f}s -- proceeding with a zero-pose "
                  "approximation for the finger gravity-compensation term.")
            arm_q0 = np.zeros(len(ARM_JOINT_NAMES))
        self._arm_q_last = arm_q0

        raw_L, raw_R, err_L, err_R = self._read_hands()
        if raw_L is None or raw_R is None:
            raise RuntimeError("Failed to read initial hand angles over Modbus.")
        q_hand12_0 = hand12_raw_to_rad(modbus_to_hand12_order(raw_L, raw_R))
        self.hand_ctrl.reset(q_hand12_0)
        self._q_hand12_prev = q_hand12_0.copy()

        self._load_segment(0)
        self.thread = RecurrentThread(interval=self.control_dt, target=self._tick, name="hw_finger_close_only")
        self.thread.Start()

    def _load_segment(self, i: int):
        seg = self.segments[i]
        self.seg_idx = i
        self.seg_t0 = self.t
        if seg.hand_mode == "position":
            self.ph_ctrl.set_position_mode()
            self.ph_ctrl.q_des_hand12 = seg.hand_target.copy()
            self.ph_ctrl.reset_integral()
        # "hold": leave ph_ctrl target as-is -- admittance settles near its
        # current q_cmd once tau_ideal relaxes toward 0.
        print(f"[SEG] -> {seg.name} (T={seg.T:.1f}s, hand_mode={seg.hand_mode})")

    def _tick(self):
        self.t += self.control_dt
        seg = self.segments[self.seg_idx]
        seg_t = self.t - self.seg_t0

        raw_L, raw_R, err_L, err_R = self._read_hands()
        if raw_L is None or raw_R is None:
            print("[WARN] Modbus read failed this tick -- holding last command.")
            return
        for side, err in (("L", err_L), ("R", err_R)):
            if err is None:
                continue
            for j, byte_val in enumerate(err):
                flags = error_flags(byte_val)
                if flags:
                    print(f"[ABORT] {side} hand joint {j} error flags {flags} -- stopping.")
                    self.done = True
                    return

        if self._have_low_state:
            self._arm_q_last = np.array(
                [self.low_state.motor_state[j].q for j in ARM_JOINTS])[_IK17_TO_ARM_ORDER]

        q_hand12_meas = hand12_raw_to_rad(modbus_to_hand12_order(raw_L, raw_R))
        qd_hand12_raw = (q_hand12_meas - self._q_hand12_prev) / self.control_dt
        self._qd_hand12_filt = (self.qdot_ema_alpha * qd_hand12_raw
                                 + (1.0 - self.qdot_ema_alpha) * self._qd_hand12_filt)
        self._q_hand12_prev = q_hand12_meas.copy()

        q_sim29 = np.concatenate([self._arm_q_last, q_hand12_meas])
        qd_sim29 = np.concatenate([np.zeros(len(ARM_JOINT_NAMES)), self._qd_hand12_filt])

        q_cmd_hand12, tau_ideal = self.hand_ctrl.step(q_sim29, qd_sim29, self.control_dt)
        raw_L_cmd, raw_R_cmd = hand12_order_to_modbus(hand12_rad_to_raw(q_cmd_hand12))
        self.hands["L"].set_angles(raw_L_cmd)
        self.hands["R"].set_angles(raw_R_cmd)

        if self.t - self._last_sample_t >= self.sample_every_s:
            self._last_sample_t = self.t
            force_L = self.hands["L"].read_forces()
            force_R = self.hands["R"].read_forces()
            self._record_history(raw_L, raw_R, force_L, force_R)
            if self.console_print:
                print(f"[STATUS] t={self.t:6.2f}s seg={seg.name}")
                print_hand_status(self.hands["L"].ip, raw_L, force_L, err_L)
                print_hand_status(self.hands["R"].ip, raw_R, force_R, err_R)

        if seg_t >= seg.T:
            if self.seg_idx + 1 < len(self.segments):
                self._load_segment(self.seg_idx + 1)
            else:
                self.done = True


# =============================================================================
# Live plot -- reads Custom.hist_* (filled by _tick() on the RecurrentThread)
# from the main thread and redraws. Needs a display (a local X session, or
# `ssh -X`/`-Y` into the robot's dev PC) -- matplotlib is only imported here,
# so headless runs (just --dry_run or console-only) never need it installed.
# =============================================================================
def run_live_plot(custom: "Custom", refresh_s: float = 0.1) -> None:
    import matplotlib.pyplot as plt

    plt.ion()
    fig, (ax_angle, ax_force) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(custom._hist_keys)))

    lines_angle = {}
    lines_force = {}
    for (side, joint), color in zip(custom._hist_keys, colors):
        label = f"{side}:{joint}"
        (lines_angle[(side, joint)],) = ax_angle.plot([], [], label=label, color=color)
        (lines_force[(side, joint)],) = ax_force.plot([], [], label=label, color=color)

    ax_angle.set_ylabel("angle (0=closed .. 1000=open)")
    ax_angle.set_ylim(-20, 1020)
    ax_angle.legend(loc="upper right", fontsize=7, ncol=2)
    ax_angle.grid(alpha=0.3)
    ax_force.set_ylabel("force (g)")
    ax_force.set_xlabel("t (s)")
    ax_force.grid(alpha=0.3)
    fig.suptitle("g1_hw_finger_close_only.py -- live per-finger angle / force")
    fig.tight_layout()

    print(f"[PLOT] live window open -- refreshing every {refresh_s:.2f}s, "
          "close the window or Ctrl+C to stop watching (the hand motion keeps running).")
    try:
        while not custom.done and plt.fignum_exists(fig.number):
            t = list(custom.hist_t)
            if len(t) >= 2:
                for key in custom._hist_keys:
                    lines_angle[key].set_data(t, list(custom.hist_angle[key]))
                    lines_force[key].set_data(t, list(custom.hist_force[key]))
                t_lo = max(0.0, t[-1] - max(t[-1] - t[0], 1.0))
                ax_angle.set_xlim(t_lo, t[-1])
                ax_force.relim()
                ax_force.autoscale_view(scalex=False)
            fig.canvas.draw_idle()
            plt.pause(refresh_s)
    except KeyboardInterrupt:
        custom.done = True
    finally:
        plt.ioff()
        if plt.fignum_exists(fig.number):
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G1 real-hardware Inspire-hand PH close-only smoke test (hands only, no arm motion)."
    )
    parser.add_argument("network_interface", nargs="?", default=None,
                         help="NIC on the robot's LAN (e.g. eth0). Omit to use the default.")
    parser.add_argument("--left_hand_ip", type=str, default="192.168.123.210")
    parser.add_argument("--right_hand_ip", type=str, default="192.168.123.211")
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--close_fraction", type=float, default=0.4,
                         help="Fraction [0-1] of each finger's full range to close to "
                              "(0=open, 1=HAND12_LIMITS_HI). Conservative default for a "
                              "first hardware run -- raise once bench-validated.")
    parser.add_argument("--stay_closed", action="store_true",
                         help="Skip the automatic reopen at the end -- fingers stay closed "
                              "when the script exits. Default reopens for safety.")
    parser.add_argument("--admittance_gain", type=float, default=0.05,
                         help="rad/(Nm*s). NO analytically-correct default -- start low and "
                              "bench-tune upward. See g1_hw_ph_finger_control.py.")
    parser.add_argument("--hand_qdot_max", type=float, default=1.0,
                         help="Max finger position-setpoint rate from admittance [rad/s].")
    parser.add_argument("--modbus_speed", type=int, default=250, help="Hand internal servo speed, 0-1000.")
    parser.add_argument("--modbus_force_limit_g", type=int, default=300,
                         help="FORCE_SET safety ceiling [g].")
    parser.add_argument("--lowstate_timeout", type=float, default=3.0,
                         help="Seconds to wait for rt/lowstate before falling back to a "
                              "zero-pose gravity-comp approximation.")
    parser.add_argument("--sample_every_s", type=float, default=0.1,
                         help="Seconds between per-finger angle/force/error samples (10 Hz "
                              "default). Drives both the console status print (same layout "
                              "as tools/inspire_hand_modbus_control.py's 'status' command) "
                              "and the --live_plot history buffer. Reads FORCE_ACT an extra "
                              "time per hand only at this cadence, not every control tick.")
    parser.add_argument("--live_plot", action="store_true",
                         help="Open a live matplotlib window plotting all 12 fingers' angle "
                              "and force over time instead of printing status to the "
                              "console. Needs a display (local, or `ssh -X`/`-Y`).")
    parser.add_argument("--plot_window_s", type=float, default=20.0,
                         help="Trailing time window (s) kept/shown by --live_plot.")
    parser.add_argument("--dry_run", action="store_true",
                         help="Connect to the hands, read + print current state and error "
                              "flags once, then exit -- never sends a motion command. Run "
                              "this first on unfamiliar hardware.")
    args = parser.parse_args()

    print("WARNING: this drives real Inspire-hand hardware via a software admittance "
          "approximation of the PH torque controller. --admittance_gain has not been "
          "bench-tuned. Clear the fingers' pinch points and keep the e-stop within reach.")
    input("Press Enter to continue...")

    if args.network_interface:
        ChannelFactoryInitialize(0, args.network_interface)
    else:
        ChannelFactoryInitialize(0)

    custom = Custom(
        default_segments(args.close_fraction, args.stay_closed),
        hand_ips={"L": args.left_hand_ip, "R": args.right_hand_ip},
        control_dt=args.control_dt,
        admittance_gain=args.admittance_gain, hand_qdot_max=args.hand_qdot_max,
        modbus_speed=args.modbus_speed, modbus_force_limit_g=args.modbus_force_limit_g,
        lowstate_timeout=args.lowstate_timeout, sample_every_s=args.sample_every_s,
        plot_window_s=args.plot_window_s, console_print=not args.live_plot,
    )
    custom.Init()

    if args.dry_run:
        custom.read_only_report()
        for hand in custom.hands.values():
            hand.close()
        sys.exit(0)

    custom.Start()
    try:
        if args.live_plot:
            run_live_plot(custom, refresh_s=args.sample_every_s)
            if custom.done:
                print("Done!")
        else:
            while True:
                time.sleep(1)
                if custom.done:
                    print("Done!")
                    break
    finally:
        for hand in custom.hands.values():
            hand.close()
