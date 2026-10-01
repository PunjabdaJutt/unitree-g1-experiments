# g1_hw_handshake.py
"""Real-hardware G1 handshake: camera-triggered, force-limited, right arm only.

Sequence: watch the front camera for a human hand (HandPresenceDetector,
debounced); on a trigger, ramp the right arm to a fixed HANDSHAKE_POSE,
close the right hand with FORCE_SET capped at HAND_FORCE_G grams (the
Inspire hand's own firmware stops early on reaching that force -- see
tools/inspire_hand_modbus_control.py, no new force-limiting logic here),
hold briefly, release, retract, and go back to watching.

Deliberately does NOT estimate the hand's 3D position or aim the arm at it
-- the front camera has no depth channel and no verified calibration (see
hand_detector.py's module docstring). The camera only triggers the
sequence; HANDSHAKE_POSE is fixed and pre-validated against the arm's real
reach envelope at startup (see validate_reach() below).

Left arm, waist, and left hand are never targeted by this script and stay
wherever they were when it started (Custom's existing "omitted = hold"
segment semantics, unchanged from g1_hw_arm_hand_test.py).

Like every other script in this directory, this is gated by hw_safety.py:
flush_stale_arm_sdk() then require_locked_standing() must pass before any
rt/arm_sdk publishing starts. See safety_config.json for the incident that
gate exists to prevent.

Run:
    python scripts/g1_control/hardware/g1_hw_handshake.py eth0

WARNING: this drives real arm + hand hardware autonomously as soon as a
hand is detected in the camera. Clear the workspace and keep the e-stop
within reach before running.
"""
import argparse
import os
import sys
import time

# Must be the first project import -- its module-level reexec_with_clean_env()
# fixes LD_LIBRARY_PATH for the whole process (via os.execve) before anything
# below transitively imports unitree_sdk2py/cv2. See hand_detector.py's
# module docstring.
import hand_detector

from g1_hw_arm_hand_test import (
    FINGER_ORDER, Segment, Custom,
)
import hw_safety
from hand_detector import HandPresenceDetector

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

# ph package (kinematics for reach validation) lives at
# <PH repo root>/scripts/g1_control/ph -- same sys.path pattern as
# g1_hw_ph_finger_control.py in this directory.
_PH_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PH_REPO_ROOT not in sys.path:
    sys.path.insert(0, _PH_REPO_ROOT)
from scripts.g1_control.ph import default_urdf, fk_chain  # noqa: E402


# =============================================================================
# Handshake pose -- right arm only. UNTUNED: bench-validate on hardware
# before relying on this (same caveat as g1_hw_ph_finger_control.py's
# --admittance_gain). Shoulder pitch raised, wrist roll/pitch/yaw neutral
# except a +0.25 rad wrist roll, per spec.
# =============================================================================
HANDSHAKE_POSE = {
    "right_shoulder_pitch": 0.3,
    "right_shoulder_roll": 0.0,
    "right_elbow": 0.25,
    "right_wrist_roll": 0.25,
    "right_wrist_pitch": 0.0,
    "right_wrist_yaw": 0.0,
}
HAND_FORCE_G = 500  # FORCE_SET grip-force cap, grams -- user-specified


def validate_reach(pose: dict, margin_m: float = 0.02) -> None:
    """Print the handshake pose's reach vs. the arm's real max reach
    (found by sweeping right_elbow_joint's full range, NOT by assuming
    elbow=0 is maximally extended -- confirmed by direct FK during
    implementation that elbow=0 on this URDF is actually LESS extended
    than elbow=0.9, i.e. straight-line reach isn't monotonic in the naive
    "elbow angle small = arm straight" sense here). Raises if the pose
    exceeds max reach; this is a startup sanity check, not a runtime clamp.
    """
    import numpy as np

    urdf = default_urdf()
    shoulder = fk_chain(urdf, "pelvis", "right_shoulder_pitch_link")[:3, 3]

    pose_j = {f"{k}_joint": v for k, v in pose.items()}
    wrist = fk_chain(urdf, "pelvis", "right_wrist_yaw_link", pose_j)[:3, 3]
    pose_reach = float(np.linalg.norm(wrist - shoulder))

    elbow = urdf.joints_by_name["right_elbow_joint"]
    best = 0.0
    for e in np.linspace(elbow.lower, elbow.upper, 60):
        w = fk_chain(urdf, "pelvis", "right_wrist_yaw_link", {**pose_j, "right_elbow_joint": float(e)})[:3, 3]
        best = max(best, float(np.linalg.norm(w - shoulder)))

    print(f"  -> reach check: handshake pose = {pose_reach:.3f}m, arm max (elbow swept) = {best:.3f}m")
    if pose_reach > best - margin_m:
        raise ValueError(
            f"HANDSHAKE_POSE reach ({pose_reach:.3f}m) is not comfortably under "
            f"the arm's max reach ({best:.3f}m, {margin_m:.3f}m margin) -- adjust "
            "HANDSHAKE_POSE before running on hardware."
        )


def handshake_segments(hold_close_s: float = 1.5) -> list:
    """Segment list for one handshake cycle: reach -> grip -> hold -> release
    -> retract. Mirrors default_segments()'s structure in
    g1_hw_arm_hand_test.py (first segment ramps arm_sdk weight in, last
    ramps it out) but touches only the right arm/hand -- left side and
    waist are never in arm_targets, so Custom holds them at whatever they
    measured at Start()."""
    open_R = {f: 0.0 for f in FINGER_ORDER}
    close_R = {"pinky": 1.6, "ring": 1.6, "middle": 1.6, "index": 1.6,
               "thumb_bend": 0.6, "thumb_rot": 0.0}
    rest = {k: 0.0 for k in HANDSHAKE_POSE}
    return [
        Segment("hold_zero", 1.0),
        Segment("reach", 3.0, arm_targets=dict(HANDSHAKE_POSE), hand_R=open_R),
        Segment("grip", 2.0, hand_R=close_R),  # hand firmware stops early at FORCE_SET
        Segment("hold_shake", hold_close_s),
        Segment("release", 1.5, hand_R=open_R),
        Segment("retract", 3.0, arm_targets=rest),
        Segment("release_arm_sdk", 2.0),
    ]


def run_one_handshake(args) -> None:
    custom = Custom(
        handshake_segments(), control_dt=args.control_dt,
        arm_kp=args.arm_kp, arm_kd=args.arm_kd,
        hand_speed=args.hand_speed, hand_force=HAND_FORCE_G,
        left_hand_ip=args.left_hand_ip, right_hand_ip=args.right_hand_ip,
    )
    custom.Init()
    custom.Start()
    try:
        while not custom.done:
            time.sleep(0.1)
            print("\r" + custom.status_line() + " " * 8, end="", flush=True)
        print("\n[INFO] handshake complete")
    finally:
        custom.Shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="G1 real-hardware handshake: camera-triggered, force-capped, right arm only."
    )
    parser.add_argument("network_interface", nargs="?", default=None,
                         help="NIC on the robot's LAN (e.g. eth0). Omit to use the default.")
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--arm_kp", type=float, default=60.0)
    parser.add_argument("--arm_kd", type=float, default=1.5)
    parser.add_argument("--hand_speed", type=int, default=250,
                         help="Inspire hand ANGLE_SET ramp speed, 0-1000.")
    parser.add_argument("--left_hand_ip", default="192.168.123.210")
    parser.add_argument("--right_hand_ip", default="192.168.123.211")
    parser.add_argument("--cooldown_s", type=float, default=5.0,
                         help="Seconds to ignore new hand detections after a handshake.")
    args = parser.parse_args()

    print("WARNING: this drives REAL arm + hand hardware AUTONOMOUSLY as soon as a "
          "hand is detected in the camera. Clear the workspace and keep the e-stop "
          "within reach.")
    input("Press Enter to continue...")

    validate_reach(HANDSHAKE_POSE)

    if args.network_interface:
        ChannelFactoryInitialize(0, args.network_interface)
    else:
        ChannelFactoryInitialize(0)

    # Safety gate (safety_config.json): clear any stale rt/arm_sdk command,
    # then confirm the robot is actually in an allowed fsm_id. Same contract
    # as every other script in this directory -- see hw_safety.py.
    hw_safety.flush_stale_arm_sdk()
    try:
        hw_safety.require_locked_standing()
    except hw_safety.RobotNotReadyError as e:
        print(f"[ABORT] {e}")
        sys.exit(1)

    video_client = VideoClient()
    video_client.SetTimeout(3.0)
    video_client.Init()
    detector = HandPresenceDetector(video_client, cooldown_s=args.cooldown_s)

    print("[INFO] watching for a hand -- handshake will trigger automatically. Ctrl+C to stop.")
    try:
        while True:
            detector.poll()
            if detector.hand_present():
                print("\n[TRIGGER] hand detected -- starting handshake")
                detector.arm_cooldown()
                run_one_handshake(args)
                print("[INFO] watching for a hand again...")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] stopped by user")
    finally:
        detector.close()


if __name__ == "__main__":
    main()
