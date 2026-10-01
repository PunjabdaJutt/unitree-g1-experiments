# localize_3d.py
"""2D mask -> 3D cube position in the robot's pelvis frame.

Math:
    Pixel-to-camera unprojection (pinhole model, K = intrinsics):
        Xc = (u - cx) * d / fx
        Yc = (v - cy) * d / fy
        Zc = d
    equivalent to d * K^-1 [u, v, 1]^T with K = [[fx,0,cx],[0,fy,cy],[0,0,1]].

    Camera-to-base transform: pbase = T_cam_to_base . [Xc, Yc, Zc, 1]^T,
    T_cam_to_base a 4x4 homogeneous transform (R | t; 0 0 0 1).

"Base frame" here means the G1 pelvis frame, NOT world/odom -- that's the
frame ph.kinematics.ik_two_hands_dls / fk_chain solve in (root_link=
"pelvis"), so trajectory_gen.py and g1_controller.py downstream expect
points in this same frame. See compute_T_cam_to_base() below for how
T_cam_to_base is derived.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
X_BIAS_CORRECTION_M= 0
Y_BIAS_CORRECTION_M= 0.05
Z_BIAS_CORRECTION_M= 0.04 

@dataclass
class Localization3D:
    point_base: np.ndarray   # (3,) meters, pelvis frame
    confidence: float        # fraction of mask pixels with valid (nonzero) depth


def pixel_to_camera_point(u: float, v: float, depth_m: float, intrinsics: dict) -> np.ndarray:
    """Single-pixel unprojection. depth_m in meters, 0/NaN = invalid (caller
    should check before calling)."""
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    return np.array([x, y, depth_m], dtype=float)


def _transform_points(points_cam: np.ndarray, T_cam_to_base: np.ndarray) -> np.ndarray:
    """points_cam: (N,3) -> (N,3) in base frame."""
    ones = np.ones((points_cam.shape[0], 1))
    homog = np.hstack([points_cam, ones])            # (N,4)
    out = (T_cam_to_base @ homog.T).T                 # (N,4)
    return out[:, :3]


def centroid_to_3d(centroid_px, depth_m_image: np.ndarray, intrinsics: dict,
                    T_cam_to_base: np.ndarray) -> Optional[np.ndarray]:
    """Fast path: unproject just the 2D mask centroid pixel. Cheap but
    sensitive to a single noisy/edge depth reading -- prefer
    mask_to_3d_point() for anything actually driving the arm."""
    u, v = centroid_px
    h, w = depth_m_image.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    d = float(depth_m_image[vi, ui])
    if d <= 0.0:
        return None
    p_cam = pixel_to_camera_point(ui, vi, d, intrinsics)
    return _transform_points(p_cam[None, :], T_cam_to_base)[0]


def mask_to_3d_point(mask: np.ndarray, depth_m_image: np.ndarray, intrinsics: dict,
                      T_cam_to_base: np.ndarray, use_median: bool = True,
                      min_valid_ratio: float = 0.1) -> Optional[Localization3D]:
    """Robust path: unproject every masked pixel with valid depth, then take
    the median (default) or mean 3D point in the base frame.

    Median is preferred over mean: depth at object silhouette edges is prone
    to flying-pixel outliers (RealSense returns spurious near/far readings
    right at a depth discontinuity), and the median is far less sensitive to
    those than the mean. Returns None if fewer than min_valid_ratio of mask
    pixels have valid depth (empty/heavily-occluded mask, or depth dropout).
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    depths = depth_m_image[ys, xs]
    valid = depths > 0.0
    valid_ratio = float(valid.mean())
    if valid_ratio < min_valid_ratio:
        return None

    xs, ys, depths = xs[valid], ys[valid], depths[valid]
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    Xc = (xs - cx) * depths / fx
    Yc = (ys - cy) * depths / fy
    Zc = depths
    points_cam = np.stack([Xc, Yc, Zc], axis=1)       # (N,3)
    points_base = _transform_points(points_cam, T_cam_to_base)

    center = np.median(points_base, axis=0) if use_median else points_base.mean(axis=0)
    center[2] += Z_BIAS_CORRECTION_M
    return Localization3D(point_base=center, confidence=valid_ratio)


# D435i-optical-frame -> pelvis-frame transform, derived from the URDF
# rather than tape-measured: g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf
# defines d435_link as a FIXED child of torso_link --
#   <origin xyz="0.0576235 0.01753 0.42987"
#           rpy="0 0.8307767239493009 0"/>   (~47.6 deg pitch)
# i.e. the D435i is chest/torso-mounted and tilted down ~47.6 degrees,
# consistent with a camera meant to look at a table/object in front of the
# robot for a manipulation task. torso_link itself hangs off pelvis through
# three revolute waist joints (yaw, roll, pitch), so d435_link's pose in the
# pelvis frame is NOT strictly constant -- compute_T_cam_to_base() below
# uses fk_chain() (this repo's own URDF-driven FK, already relied on by
# g1_controller.py's real-hardware IK) to get it exactly, given the current
# waist angles.


def compute_T_cam_to_base(waist_yaw: float = 0.0, waist_roll: float = 0.0,
                           waist_pitch: float = 0.0, urdf=None) -> np.ndarray:
    """D435i optical frame -> pelvis frame (see module comment above for the
    URDF geometry this is built from).

    Two pieces composed together:
      1. fk_chain(urdf, "pelvis", "d435_link", q) -- d435_link's pose in the
         pelvis frame, from the URDF chain pelvis -> waist_yaw_link ->
         waist_roll_link -> torso_link -> d435_link. waist_yaw/roll/pitch
         default to 0.0 (upright, untwisted waist) -- true at the point
         main_grasp_loop.py calls this, right after connect() and before
         any commanded motion. Pass the live waist angles (from LowState_)
         instead of the defaults if a caller ever localizes after the waist
         has moved (ik_two_hands_dls's default use_waist=True means the arm
         IK itself is free to move the waist to reach a target, so this
         matters once detection happens anywhere but the initial pose).
      2. A fixed rotation from d435_link's axes -- x-forward, y-left,
         z-up, the same body-frame convention every other link in this URDF
         uses (confirmed via the waist joints' own rotation axes: yaw=z,
         roll=x, pitch=y) -- to the standard pinhole/optical convention
         pixel_to_camera_point() assumes (+X right, +Y down, +Z forward
         into the scene). This is the same static camera_link ->
         camera_optical_frame correction every ROS camera driver applies
         (quaternion (-0.5, 0.5, -0.5, 0.5)), and is why this isn't just a
         translation.

    Ignores the few-mm stereo-module offset between d435_link's origin and
    the color sensor's actual optical center -- below this pipeline's other
    error sources (mask edge noise, depth quantization).
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    _ph_submodule_root = os.path.abspath(os.path.join(_here, *([os.pardir] * 4)))
    if _ph_submodule_root not in sys.path:
        sys.path.insert(0, _ph_submodule_root)
    from scripts.g1_control.ph import fk_chain, default_urdf as _default_urdf

    urdf = urdf or _default_urdf()
    q_by_name = {
        "waist_yaw_joint": waist_yaw,
        "waist_roll_joint": waist_roll,
        "waist_pitch_joint": waist_pitch,
    }
    T_d435link_to_pelvis = fk_chain(urdf, "pelvis", "d435_link", q_by_name)

    R_d435link_from_optical = np.array([
        [0.0, 0.0, 1.0],    # optical +Z (forward)   -> d435_link +X (forward)
        [-1.0, 0.0, 0.0],   # optical +X (right)     -> d435_link -Y (left is +Y, so right is -Y)
        [0.0, -1.0, 0.0],   # optical +Y (down)      -> d435_link -Z (up is +Z, so down is -Z)
    ])
    T_optical_to_d435link = np.eye(4)
    T_optical_to_d435link[:3, :3] = R_d435link_from_optical

    return T_d435link_to_pelvis @ T_optical_to_d435link


# Evaluated once at import, waist angles defaulted to 0.0 -- see
# compute_T_cam_to_base()'s docstring for when that stops being valid.
T_CAM_TO_BASE = compute_T_cam_to_base()
