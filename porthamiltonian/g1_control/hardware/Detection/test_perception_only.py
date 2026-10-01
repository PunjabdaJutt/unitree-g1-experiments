# test_perception_only.py
"""Perception-only smoke test for main_grasp_loop.py's detect_and_localize()
logic -- runs YOLO detection + 3D localization against the live depth feed
and prints/draws the result. Deliberately never imports g1_controller and
never touches rt/arm_sdk or the Inspire hand's Modbus connection, so none of
safety_config.json's arm_sdk hazards (fsm-state gate, graceful winddown)
apply here -- see that file's "not_applicable" entry for this script. Safe
to run with no e-stop/workspace precautions; Ctrl+C just closes the camera
subscriber and the display window (the `finally` block below), same as
detection.py's own standalone viewer.

Also doubles as a capture tool for building a fine-tuning dataset (see
cube-yolo-grey-domain-gap: cube_yolo26_seg.pt was only ever trained on
saturated colored cubes, never the real grey one) -- press 's' to save the
CURRENT RAW color frame (no overlay -- annotation happens later, e.g. in
Roboflow, on a clean image) to captured_frames/. Filenames auto-increment
and never overwrite previous captures, including across separate runs.

Usage: python3 test_perception_only.py [network_interface] [out_dir] [model_path] [--track]

model_path defaults to detection.DEFAULT_MODEL_PATH (cube_yolo26_seg.pt) --
pass a fine-tuned checkpoint's path (e.g. a training run's weights/best.pt)
to A/B it against the live feed without touching the current weights file.

--track: also runs each confident detection through
trajectory_gen.GraspTrajectory.update() + solve_ik_step() (IK warm-started
from the previous solve, then FK-verified) -- same tracking logic
dry_run_grasp.py's --live mode uses -- and draws the target/IK/FK numbers on
this window too, updating every frame. Still perception+IK+FK only, no
hardware touched.
"""
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_HARDWARE_DIR = os.path.abspath(os.path.join(_HERE, os.pardir))
if _HARDWARE_DIR not in sys.path:
    sys.path.insert(0, _HARDWARE_DIR)

import cv2
import numpy as np

from detection import open_camera, CubeSegmenter, DEFAULT_MODEL_PATH  # noqa: E402
from localize_3d import mask_to_3d_point, T_CAM_TO_BASE  # noqa: E402
from trajectory_gen import (GraspTrajectory, solve_ik_step, fk_chain,  # noqa: E402
                             default_urdf, IK17_JOINT_NAMES)

MIN_CONFIDENCE = 0.5  # same floor main_grasp_loop.py uses
DEFAULT_CAPTURE_DIR = os.path.join(_HERE, "captured_frames")
CAPTURE_PATTERN = "cube_{:04d}.jpg"


def _next_capture_index(out_dir: str) -> int:
    """Highest existing cube_NNNN.jpg index in out_dir, +1 -- so re-running
    this script (or running it again after a previous capture session)
    appends instead of overwriting."""
    existing = glob.glob(os.path.join(out_dir, "cube_*.jpg"))
    indices = [int(m.group(1)) for f in existing
               if (m := re.search(r"cube_(\d+)\.jpg$", os.path.basename(f)))]
    return max(indices, default=-1) + 1


def main():
    args = sys.argv[1:]
    track = "--track" in args
    if track:
        args.remove("--track")
    iface = args[0] if len(args) > 0 else None
    out_dir = args[1] if len(args) > 1 else DEFAULT_CAPTURE_DIR
    model_path = args[2] if len(args) > 2 else DEFAULT_MODEL_PATH
    os.makedirs(out_dir, exist_ok=True)
    next_idx = _next_capture_index(out_dir)

    cam = open_camera(network_interface=iface, local=False)
    seg = CubeSegmenter(model_path)

    tracker = None  # GraspTrajectory, created on the first confident detection (--track only)
    if track:
        urdf = default_urdf()
        q_ik = np.zeros(17)  # first IK solve only -- warm-started from the previous q after that
        pL_hold = fk_chain(urdf, "pelvis", "left_wrist_yaw_link",
                            dict(zip(IK17_JOINT_NAMES, q_ik)))[:3, 3]

    print("[INFO] perception-only test running -- no hardware will move.")
    print(f"[INFO] model: {model_path}"
          + (" (default)" if model_path == DEFAULT_MODEL_PATH else " (override)"))
    print(f"[INFO] press 's' to save the current raw frame to {out_dir}/ (next: "
          f"{CAPTURE_PATTERN.format(next_idx)}), 'q' to quit.")
    if track:
        print("[INFO] --track enabled: tracking pre_grasp waypoint IK/FK on top of detection.")
    try:
        while True:
            result = cam.read()
            if result is None:
                continue
            color, depth_m, intrinsics = result
            det = seg.detect(color)
            vis = color.copy()

            status = "no detection"
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
                              f"depth_valid={loc.confidence:.2f} "
                              f"{'OK' if loc.confidence >= MIN_CONFIDENCE else 'below MIN_CONFIDENCE'}")

                    if track and loc.confidence >= MIN_CONFIDENCE:
                        if tracker is None:
                            tracker = GraspTrajectory(loc.point_base)
                            print(f"[TRACK] tracker initialized at {loc.point_base.round(3)}")
                        else:
                            tracker.update(loc.point_base)
                        wp, q_ik, info, wrist_fk, fk_err = solve_ik_step(tracker, urdf, pL_hold, q_ik)
                        ik_status = "OK" if info.converged else "NO_CONVERGE"
                        track_line = (f"{wp.name} target={wp.position.round(3)}  "
                                      f"IK:{ik_status} ({info.iters} it)  "
                                      f"wrist_FK={wrist_fk.round(3)}  |FK-target|={fk_err:.4f}m")
                        cv2.putText(vis, track_line, (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 200, 255), 2)
                        print(f"[TRACK] {track_line}")
                else:
                    status = f"score={det.score:.2f} but mask_to_3d_point failed (no valid depth)"

            cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(vis, f"captured: {next_idx} (s=save, q=quit)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            print(status)
            cv2.imshow("perception-only (no hardware)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Raw `color`, not `vis` -- the mask/box overlay must not end
                # up baked into a training image.
                path = os.path.join(out_dir, CAPTURE_PATTERN.format(next_idx))
                cv2.imwrite(path, color)
                print(f"[SAVED] {path}")
                next_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
