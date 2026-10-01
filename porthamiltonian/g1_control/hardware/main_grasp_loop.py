# main_grasp_loop.py
"""Ties Detection/detection.py -> Detection/localize_3d.py -> trajectory_gen
-> g1_controller into one single-shot grasp: capture a frame, detect the
cube, localize it in 3D, generate a pre-grasp/grasp/post-grasp trajectory,
and execute it on the real G1 right arm + Inspire hand.

This is open-loop per grasp attempt (detect once, then execute the whole
trajectory blind) -- it does NOT re-detect/re-localize mid-approach. TODO:
if the cube can move or the initial estimate is unreliable, re-run detection
+ localization before the final `grasp` waypoint (closed-loop visual
servo) rather than trusting a single detection from the pre-grasp pose.

WARNING: drives real hardware. Clear the workspace, keep the e-stop within
reach, and run Detection/detection.py standalone first to confirm detection
quality before ever letting this move the arm.

Usage:
    python3 main_grasp_loop.py [network_interface]
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "Detection"))

from detection import open_camera, CubeSegmenter, DEFAULT_MODEL_PATH  # noqa: E402
from localize_3d import mask_to_3d_point, T_CAM_TO_BASE  # noqa: E402
from trajectory_gen import generate_grasp_waypoints  # noqa: E402
from g1_controller import G1ArmHandController  # noqa: E402

# TODO: tune per your setup.
DETECT_TIMEOUT_S = 5.0        # give up if no cube detected within this long
MIN_CONFIDENCE = 0.5          # mask_to_3d_point's valid-depth-ratio floor
WAYPOINT_DURATION_S = 3.0     # per-waypoint ramp time


def detect_and_localize(cam, seg: CubeSegmenter, timeout_s: float):
    """Polls the camera until a confident cube detection + 3D localization
    is available, or returns None after timeout_s."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        result = cam.read()
        if result is None:
            continue
        color, depth_m, intrinsics = result
        det = seg.detect(color)
        if det is None:
            continue
        loc = mask_to_3d_point(det.mask, depth_m, intrinsics, T_CAM_TO_BASE)
        if loc is None or loc.confidence < MIN_CONFIDENCE:
            continue
        return loc
    return None


def run_grasp(network_interface: str = None, local_camera: bool = False):
    # Camera: DDS by default (subscribes to depth_relay_onboard.py running
    # on the robot's Jetson, from ~/g1_depth_bridge/ there -- NOT the loose
    # TCP-socket duplicate that was also found at ~/depth_relay_onboard.py
    # on that box; delete/ignore that one). Same network_interface as the
    # arm/hand connection below -- both are DDS channels on this repo's
    # one Unitree SDK ChannelFactory (singleton-safe to init twice with the
    # same params, see channel.py's ChannelFactory.Init). Pass
    # local_camera=True if the D435i is physically plugged in here instead.
    cam = open_camera(network_interface=network_interface, local=local_camera)
    seg = CubeSegmenter(DEFAULT_MODEL_PATH)
    controller = G1ArmHandController(network_interface=network_interface)

    try:
        print("[INFO] connecting to G1...")
        controller.connect()

        print("[INFO] detecting cube...")
        loc = detect_and_localize(cam, seg, DETECT_TIMEOUT_S)
        if loc is None:
            print("[ABORT] no confident cube detection within timeout")
            return
        print(f"[INFO] cube at {loc.point_base} (base frame), confidence={loc.confidence:.2f}")

        waypoints = generate_grasp_waypoints(loc.point_base)

        controller.open_hand()
        for wp in waypoints:
            print(f"[INFO] moving to {wp.name}: {wp.position}")
            converged = controller.move_to_pose(wp.position, wp.R, duration=WAYPOINT_DURATION_S)
            if not converged:
                print(f"[WARN] IK did not converge for {wp.name} -- executed best-effort solution")
            if wp.name == "grasp":
                controller.close_hand()
                time.sleep(1.0)  # let the Inspire firmware finish closing before lifting

        print("[INFO] grasp sequence complete")

    finally:
        cam.close()
        controller.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("network_interface", nargs="?", default=None,
                         help="NIC on the robot's LAN (e.g. eno1)")
    parser.add_argument("--local-camera", action="store_true",
                         help="D435i is physically plugged into this machine "
                              "instead of relayed over DDS from the Jetson")
    args = parser.parse_args()

    print("WARNING: this drives REAL arm + hand hardware. Clear the workspace "
          "and keep the e-stop within reach.")
    input("Press Enter to continue...")
    run_grasp(args.network_interface, local_camera=args.local_camera)
