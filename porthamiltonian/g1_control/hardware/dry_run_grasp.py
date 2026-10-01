# dry_run_grasp.py
"""Dry run of main_grasp_loop.py's run_grasp(): detect the cube, localize it
in 3D, generate pre/grasp/post waypoints, and solve IK for each -- the exact
same steps run_grasp() takes -- but stops there. Deliberately never imports
g1_controller, so rt/arm_sdk is never published to and the Inspire hand's
Modbus connection is never touched -- see safety_config.json's
"not_applicable" entry for this script. Safe to run with no e-stop/workspace
precautions.

Caveat: IK is seeded from q0=zeros(17) (an assumed rest pose), NOT the
robot's actual live joint state (this script never connects to
rt/lowstate) -- "converged" here is a feasibility check against that
assumed start, not a guarantee the real run would converge identically
starting from wherever the arm actually is.

Usage: python3 dry_run_grasp.py [network_interface]           (one-shot)
       python3 dry_run_grasp.py [network_interface] --live    (continuous)

--live: instead of solving IK once and exiting, loops forever -- opens the
same live camera window Detection/test_perception_only.py does (detection
box/mask overlay), and each new confident detection also calls
trajectory_gen.GraspTrajectory.update() with the fresh cube position,
re-solves IK for the pre_grasp waypoint (warm-started from the previous
solve's q, not from rest -- see CLOSED_LOOP_CUBE_TRACKING.md step 7), then
re-runs FK on the solved joints and draws + prints target vs. actual-FK
wrist position every iteration, so you can watch the numbers move instead of
seeing a single final answer. Still just detect+IK+FK -- no hardware
touched. Press 'q' in the window (or Ctrl+C) to stop.
"""
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "Detection"))

from detection import open_camera, CubeSegmenter, DEFAULT_MODEL_PATH  # noqa: E402
from localize_3d import mask_to_3d_point, T_CAM_TO_BASE  # noqa: E402
from trajectory_gen import generate_grasp_waypoints, GraspTrajectory, solve_ik_step  # noqa: E402

_PH_SUBMODULE_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 3)))
if _PH_SUBMODULE_ROOT not in sys.path:
    sys.path.insert(0, _PH_SUBMODULE_ROOT)
from scripts.g1_control.ph import ik_two_hands_dls, fk_chain, default_urdf, IK17_JOINT_NAMES  # noqa: E402

DETECT_TIMEOUT_S = 5.0   # same as main_grasp_loop.py
MIN_CONFIDENCE = 0.5     # same as main_grasp_loop.py


def detect_and_localize(cam, seg, timeout_s):
    """Copy of main_grasp_loop.py's own function, duplicated rather than
    imported so this file never pulls in g1_controller (see module
    docstring)."""
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


def live_track(cam, seg, urdf, tracker: GraspTrajectory, target_name: str = "pre_grasp"):
    """Continuously: detect -> tracker.update(new cube position) ->
    solve_ik_step() (IK warm-started from the previous q + FK verification).
    Draws the same detection box/mask overlay as
    Detection/test_perception_only.py in a live cv2 window, with the
    tracking numbers (target, IK status, FK-verified wrist position) drawn
    on top and printed to the console each accepted detection. Lets you
    watch the IK target and the actual FK-verified wrist position move
    together in real time, rather than a single one-shot final answer. Pure
    detect+IK+FK -- see module docstring, no hardware touched. Press 'q' in
    the window (or Ctrl+C, caught by main()) to stop."""
    q = np.zeros(17)  # first solve only -- every solve after warm-starts from the previous q
    pL_hold = fk_chain(urdf, "pelvis", "left_wrist_yaw_link",
                        dict(zip(IK17_JOINT_NAMES, q)))[:3, 3]
    print(f"[INFO] live tracking '{target_name}' -- 'q' in the window (or Ctrl+C) to stop.")
    n = 0
    while True:
        result = cam.read()
        if result is None:
            continue
        color, depth_m, intrinsics = result
        det = seg.detect(color)
        vis = color.copy()
        status = "no detection"
        track_line = ""

        if det is not None:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            overlay = np.zeros_like(vis)
            overlay[det.mask] = (0, 255, 0)
            vis = cv2.addWeighted(vis, 1.0, overlay, 0.4, 0)
            u, v = det.centroid_px
            cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            loc = mask_to_3d_point(det.mask, depth_m, intrinsics, T_CAM_TO_BASE)
            if loc is not None:
                status = (f"score={det.score:.2f} point_base={loc.point_base.round(3)} "
                          f"conf={loc.confidence:.2f} "
                          f"{'OK' if loc.confidence >= MIN_CONFIDENCE else 'below MIN_CONFIDENCE'}")
                if loc.confidence >= MIN_CONFIDENCE:
                    tracker.update(loc.point_base)
                    wp, q, info, wrist_fk, fk_err = solve_ik_step(tracker, urdf, pL_hold, q, target_name)
                    n += 1
                    ik_status = "OK" if info.converged else "NO_CONVERGE"
                    track_line = (f"[{n:04d}] {target_name} target={wp.position.round(3)}  "
                                  f"IK:{ik_status} ({info.iters} it)  "
                                  f"wrist_FK={wrist_fk.round(3)}  |FK-target|={fk_err:.4f}m")
            else:
                status = f"score={det.score:.2f} but mask_to_3d_point failed (no valid depth)"

        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if track_line:
            cv2.putText(vis, track_line, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            print(track_line)
        else:
            print(status)
        cv2.imshow("dry_run_grasp --live (no hardware)", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    args = sys.argv[1:]
    live = "--live" in args
    if live:
        args.remove("--live")
    iface = args[0] if args else None
    cam = open_camera(network_interface=iface, local=False)
    seg = CubeSegmenter(DEFAULT_MODEL_PATH)

    if live:
        print("[INFO] live dry run: detect -> update tracker -> solve IK -> verify FK, on a loop.")
        print("[INFO] No hardware will move -- g1_controller is never imported here.")
        try:
            print("[INFO] waiting for first confident detection to initialize the tracker...")
            loc0 = detect_and_localize(cam, seg, DETECT_TIMEOUT_S)
            if loc0 is None:
                print("[ABORT] no confident cube detection within timeout")
                return
            print(f"[INFO] tracker initialized at {loc0.point_base.round(3)} (pelvis frame)")
            tracker = GraspTrajectory(loc0.point_base)
            urdf = default_urdf()
            live_track(cam, seg, urdf, tracker)
        except KeyboardInterrupt:
            pass
        finally:
            cam.close()
            cv2.destroyAllWindows()
        return

    print("[INFO] dry run: detect -> localize -> generate waypoints -> solve IK.")
    print("[INFO] No hardware will move -- g1_controller is never imported here.")
    try:
        print("[INFO] detecting cube...")
        loc = detect_and_localize(cam, seg, DETECT_TIMEOUT_S)
        if loc is None:
            print("[ABORT] no confident cube detection within timeout")
            return
        print(f"[INFO] cube at {loc.point_base.round(3)} (pelvis frame), "
              f"confidence={loc.confidence:.2f}")

        waypoints = generate_grasp_waypoints(loc.point_base)

        urdf = default_urdf()
        names17 = IK17_JOINT_NAMES
        q = np.zeros(17)  # ASSUMED rest pose -- see module docstring caveat
        pL_hold = fk_chain(urdf, "pelvis", "left_wrist_yaw_link", dict(zip(names17, q)))[:3, 3]

        print("\n[ASSUMPTION] IK seeded from q0=zeros(17) (rest pose), NOT the robot's "
              "actual live joints -- this script never connects to rt/lowstate.")
        for wp in waypoints:
            q, info = ik_two_hands_dls(pL_target=pL_hold, pR_target=wp.position,
                                        RR_target=wp.R, q0=q, urdf=urdf)
            status = "CONVERGED" if info.converged else "DID NOT CONVERGE"
            print(f"\n[{wp.name}] target={wp.position.round(3)}  IK: {status} ({info.iters} iters)")
            print(f"  final pos error (m): L={info.final_errors.get('eL_p_norm', 0):.5f}  "
                  f"R={info.final_errors.get('eR_p_norm', 0):.5f}")
            print(f"  final rot error: R={info.final_errors.get('eR_w_norm', 0):.5f}")
            right_arm = ", ".join(f"{n.replace('_joint', '')}={v:.3f}"
                                   for n, v in zip(names17[-7:], q[-7:]))
            print(f"  right arm joints (rad): {right_arm}")
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()


if __name__ == "__main__":
    main()
