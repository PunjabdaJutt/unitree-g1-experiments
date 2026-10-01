# grasp_ph.py
"""Camera-driven cube grasp: detect the cube (D435i + YOLO), localize it in
3D, reach for it with Cartesian arm IK, and grasp it with
g1_hw_ph_finger_control.py's compliant PH-admittance hand controller --
"Stage 4" of hardware/PIPELINE.md's own porting roadmap.

Reuses, unmodified:
  - detection.py                         -- camera -> cube mask
  - trajectory_gen.py                    -- cube position -> pre_grasp/grasp/post_grasp
                                             Cartesian waypoints (position + orientation)
  - g1_hw_ph_finger_control.Custom       -- arm_sdk position-PD + PH-admittance hand
                                             control.

2026-09-16: re-origined onto the right arm ONLY -- 7 DOF (shoulder pitch/
roll/yaw, elbow, wrist roll/pitch/yaw), no waist, no left arm, Cartesian
targets in the "shoulder frame" (pelvis-aligned axes, origin at the right
shoulder pivot -- see pin_kinematics.shoulder_origin_pin's own docstring)
instead of the pelvis frame. localize_3d.compute_T_cam_to_shoulder_pin()
gives detect_and_localize()/detect_and_localize_cloud() shoulder-frame
points directly; pin_kinematics_client.ik_right_arm_dls_remote()/
fk_right_arm_remote() do the IK/FK, built on top of (not replacing)
pin_kinematics.py's existing 17-DOF two-hand solve -- see that module's own
"Right-arm-only" section docstring. bounded_ik_pipeline.py (this script's
own JOINT_BOUNDS sandbox) was updated the same way, to stay representative;
ph.kinematics's separate, more general two-hand solver -- used by the
mujoco simulation harness, main_grasp_loop.py's stack, and PH finger-control
research scripts -- is untouched by this change.

2026-09-11: back to the camera-driven, IK-solved waypoint pipeline (ready +
pre_grasp/grasp/post_grasp -- the "4 poses") after a detour through fully
hand-captured joint poses (APPROACH_POSE/DESCEND_POSE/GRASP_POSE/LIFT_POSE,
see git history) -- that detour was a real, useful diagnostic (it isolated
a genuine DLS redundancy bug: right_wrist_pitch covering a reach that
should have come from right_shoulder_pitch, confirmed from a real run's
arm_positions_*.csv), but it also meant cube detection no longer drove the
arm at all, which defeats the point of a camera-driven grasp for anything
other than the exact vial position those poses were captured at. The IK
solve itself now runs through pin_kinematics_client.py instead of
ph.kinematics.ik_two_hands_dls -- a Pinocchio-backed solver running in a
SEPARATE process (its own terminal, the conda `pinocchio` env, since
.venv's own pip-installed Pinocchio has an unrelated ABI bug -- see
pin_kinematics.py's own docstring), cross-validated against the old
solver's output bit-for-bit before being wired in here. THE PINOCCHIO
SERVER MUST BE RUNNING before this script's preflight IK check will work --
see pin_kinematics_server.py's own docstring for how to start it; this
script raises a clear error (not a hang) if it isn't.

Logs written every run (grasp_logs/, timestamped): arm_positions_*.csv (ALL
29 body joints -- legs+waist+arms, from rt/lowstate; legs/waist are logged
purely so you can confirm they actually stayed put, not because this script
commands them -- see READY_POSE's comment on why it can't touch legs at
all) and hand_state_*.csv (all 12 finger joints' position + force + fixed
open/close angle, from the same Modbus reading Custom._tick() already takes
-- see Custom.last_raw_L/R/last_f_L/R).

WARNING: drives real hardware -- real arm motion AND a real compliant grasp
via g1_hw_ph_finger_control.HandAdmittancePH, whose gains
(--admit_mass/--admit_damping/--admit_ktrack/--force_guard_frac) are still
NOT bench-tuned (see that file's own docstring / PIPELINE.md Stage 2/3).
Clear the workspace, keep the e-stop within reach, and run
Detection/test_perception_only.py and dry_run_grasp.py first to confirm
detection quality and reachability before ever running this for real.

Usage:
    # terminal 1 (conda env pinocchio) -- once, before running this script:
    scripts/g1_control/ph/pin_kinematics_server.py
    # terminal 2 (.venv):
    python3 grasp_ph.py eno1
"""
import csv
from dataclasses import dataclass
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "Detection"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "ph"))

from detection import open_camera, CubeSegmenter, DEFAULT_MODEL_PATH  # noqa: E402
from localize_3d import mask_to_3d_point, mask_to_point_cloud, compute_T_cam_to_shoulder_pin  # noqa: E402
from trajectory_gen import generate_grasp_waypoints, PALM_NORMAL_LOCAL, HAND_OFFSET_LOCAL, Waypoint  # noqa: E402
import grasp_pose_generator  # noqa: E402 -- geometric (PCA/GraspGen) normal estimation,
                              # only used when --geometric_normal is passed (see main())

import g1_hw_ph_finger_control as phctrl  # noqa: E402
import hw_safety  # noqa: E402
import pelvis_stability_check  # noqa: E402
from log_joint_positions import G1_JOINT_NAMES  # noqa: E402

import pin_kinematics_client as pik  # noqa: E402 -- talks to pin_kinematics_server.py over a
                                      # local socket; does NOT import pinocchio itself (see its
                                      # own docstring), safe to import here in .venv.

# The 7-DOF right arm, in Segment.arm_targets's own (no "_joint" suffix)
# naming -- duplicated here rather than imported from ph.kinematics, since
# this script otherwise has no reason to import anything .venv-side from
# ph/ at all now that IK itself is remote (see pin_kinematics.py's
# RIGHT_ARM_JOINT_NAMES for the canonical, "_joint"-suffixed copy this must
# stay in sync with).
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# Assumed waist pose for every shoulder-frame conversion in this script
# (detection AND IK) -- matches localize_3d.compute_T_cam_to_base_pin's own
# default/documented assumption ("true at the point main_grasp_loop.py
# calls this, right after connect() and before any commanded motion").
# Keeping detection's and IK's waist assumption identical is what makes
# "shoulder frame" a single, self-consistent frame across this whole
# script -- see compute_T_cam_to_shoulder_pin's own docstring. If the real
# waist is ever NOT at 0 when this script runs (it never commands the
# waist itself, but the robot could already be sitting somewhere else),
# both detection and IK are equally wrong by the same amount, not
# inconsistent with each other.
WAIST_Q_ASSUMED = (0.0, 0.0, 0.0)

DETECT_TIMEOUT_S = 5.0
MIN_CONFIDENCE = 0.5

#
GRASP_NORMAL = np.array([-0.11741948240259005, -0.9582635602545816, 0.2606599590280831])
GRASP_FORWARD_REF = np.array([0.993030860203254, -0.11062245870727877, 0.04064950569861336])

# Z raised 0.01 -> 0.03 (+0.02m) on 2026-09-15, per user-requested adjustment
# after reviewing real runs with the tuned wrist_weight=300/waist_weight=30
# IK settings -- the detected object's Z was landing a bit low. Empirical,
# not yet re-measured against a tape/ground-truth reference (see
# localize_3d.py's own 2026-09-09 incident comment for why a bare, unmeasured
# Z correction here has bitten this repo before -- re-derive this from a
# measured reference before trusting it much beyond the runs that motivated
# it).
#
# X pushed -0.02 -> -0.06 (-0.04m) on 2026-09-17 to buy IK reach margin: the
# "grasp" waypoint (arm nearly fully extended) was landing within ~9mm of
# the arm's measured max reach (0.4098m from the shoulder pivot, see
# pin_kinematics reach-envelope analysis), a near-singular margin that let
# borderline targets fail to converge. This is a REACH accommodation, not a
# perception fix -- unlike the Z term above, it deliberately moves the IK
# target away from the cube's true position (toward the shoulder), trading
# grasp-point accuracy for a few cm of extra reach margin. Only helps
# borderline-reachable targets (confirmed: pulls a target that was 9mm short
# to +30mm of margin); does nothing for a target that's genuinely far out of
# reach (confirmed: one 12cm beyond max reach is still ~5cm short at this
# offset) -- that case needs the cube physically repositioned closer, not a
# bigger offset. Re-tune (or drop) this term if grasp accuracy on real
# objects looks off center now.
# 2026-09-17 evening: X=-0.09 / Z=+0.07 (set after the 16:37 run to fight a
# hand that landed ~13cm low) REVERTED to X=-0.02 / Z=+0.01. The low hand was
# not a perception/Z problem -- the commanded arm pose passed through the
# torso (see BODY_BOXES_PELVIS's comment) and the real arm stopped against
# the body. Pulling X further back makes that WORSE (wrist under the
# shoulder -> elbow folds into the chest); and with the object's top
# measured only ~1.7cm above the detected median (live point cloud, see
# CubeSegmenter/localize_3d), a +7cm Z offset puts the palm centre ~5cm
# above the object. The IK sweep behind this (object 0.33m ahead of the
# shoulder, pre-flight with body clearance ON) passes all three waypoints
# with X=-0.02, Z=+0.01 once the object sits >=0.20m to the robot's RIGHT
# of the pelvis midline and >=4cm higher than the 16:37 placement
# (shoulder-frame Z >= -0.27); nothing passes on the midline at any height.
GRASP_TARGET_OFFSET = np.array([-0.0, 0.05, 0.13])


PRE_GRASP_Z_CLEARANCE_M = 0.0
#
GEOMETRIC_CLOSING_AXIS_HINT = [0,0,1]

VIDEO_DIR = os.path.join(_HERE, "grasp_videos")
VIDEO_FPS = 15.0  # nominal write rate -- writes are wall-clock-gated to this,
                   # not tied to the camera's own arrival rate (cam.read()
                   # returns the latest cached frame on every call, so writing
                   # unthrottled would duplicate frames instead of recording
                   # real motion at the stated fps)

LOG_DIR = os.path.join(_HERE, "grasp_logs")
STATE_LOG_HZ = 10.0  # arm/hand state log sample rate -- same wall-clock-gating
                      # rationale as VIDEO_FPS above, independent of it


_GRASP_TAU_R = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
GRASP_TORQUE_SCALE = 1.5  # matches default_segments()'s own 1.5x scaling


READY_POSE = {
    "right_shoulder_pitch": -0.29, "right_shoulder_roll": -0.100, "right_shoulder_yaw": -0.273,
    "right_elbow": 0.5, "right_wrist_roll": -0.004, "right_wrist_pitch": -0.359, "right_wrist_yaw": -0.056,
}

LIFT_POSE={"waist_yaw": 0.159, "waist_roll": -0.010, "waist_pitch": 0.008,
    "right_shoulder_pitch": -0.168, "right_shoulder_roll": -0.205, "right_shoulder_yaw": -0.373,
    "right_elbow": 0.600, "right_wrist_roll": -0.214, "right_wrist_pitch": -0.240, "right_wrist_yaw": -0.056,
}

REFERENCE_SEED_POSE = {
    "waist_yaw": 0.326, "waist_roll": 0.016, "waist_pitch": 0.010,
    "right_shoulder_pitch": -0.061, "right_shoulder_roll": 0.015, "right_shoulder_yaw": -0.020,
    "right_elbow": 1.095, "right_wrist_roll": 0.157, "right_wrist_pitch": -0.818, "right_wrist_yaw": 0.417,
}

TRAJ_POSE_3 = {
    "waist_yaw": -0.003 , "waist_roll": +0.000, "waist_pitch":0.005 ,
    "right_shoulder_pitch": -0.5, "right_shoulder_roll": -0.027, "right_shoulder_yaw": 0.123,
    "right_elbow": 0.876, "right_wrist_roll": 0.03, "right_wrist_pitch": -0.516, "right_wrist_yaw": +0.34,
}
WAIST_DLS_WEIGHT = 30.0

WRIST_DLS_WEIGHT =1.0


TOL_P_M = 0.003
TOL_AXIS_DEG = 20.0
# Hand orientation handling (2026-09-18). The pre-flight IK used to pin only
# the palm normal ("aim mode", axis_local/axis_target) and leave the hand's
# roll about it free. FK of the resulting solutions showed the hand's
# pointing direction landing anywhere from 50deg nose-down to 70deg nose-up
# between waypoints/solves, and since the palm centre sits ~10cm out along
# the hand, the PALM ended up 4-9cm below (or above) the planned point while
# the wrist was exactly on target -- the "end effector is ~4cm too low"
# symptom. Full 3-DOF orientation (R_target) fixes that: the palm lands
# within ~3mm of plan. HAND_PITCH_UP_DEG is the planned nose-UP pitch of the
# hand (positive = fingers above the wrist, per operator request 2026-09-18;
# the first version tilted it nose-down). Trade-off measured on the
# 2026-09-18 target: nose-up lowers the wrist and costs reach at the grasp
# (+10deg: ~0.4cm short at that placement, fine 8cm further right; +20deg:
# ~2.5cm short), nose-down buys reach. --aim_mode restores the old
# behaviour, --hand_pitch_up_deg overrides the angle (negative = nose-down).
HAND_PITCH_UP_DEG = 2.0
TOL_W_RAD = 0.15          # orientation tolerance for full-orientation IK (~8.6deg)
ORIENTATION_MODE = "full"  # "full" | "aim" -- set from --aim_mode in main()


def _tilted_forward_ref(deg=None):
    """GRASP_FORWARD_REF pitched nose-UP by `deg` (positive = fingers above
    the wrist) about the pelvis Y axis."""
    d = HAND_PITCH_UP_DEG if deg is None else deg
    c, s_ = np.cos(np.radians(-d)), np.sin(np.radians(-d))   # Ry(+a) tips +X toward -Z, so negate
    Ry = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
    return Ry @ GRASP_FORWARD_REF


def solve_waypoint_ik(p_target, R_target, q7_seed, max_iters=2000):
    """The one IK call both build_via_waypoints() and preflight_ik_check()
    use, so a via point judged feasible is feasible under the same
    constraints the pre-flight enforces. Returns (q7, info)."""
    if ORIENTATION_MODE == "aim":
        q7, info, _ = pik.ik_right_arm_dls_remote(
            p_target=p_target, waist_q=WAIST_Q_ASSUMED, q0_7=q7_seed,
            axis_local=PALM_NORMAL_LOCAL, axis_target=R_target @ PALM_NORMAL_LOCAL,
            joint_bounds=dict(WRIST_JOINT_BOUNDS), waist_weight=WAIST_DLS_WEIGHT,
            wrist_weight=WRIST_DLS_WEIGHT, tol_p=TOL_P_M, aim_length=AIM_LENGTH_M,
            tol_axis_deg=TOL_AXIS_DEG, max_iters=max_iters)
    else:
        q7, info, _ = pik.ik_right_arm_dls_remote(
            p_target=p_target, R_target=R_target, waist_q=WAIST_Q_ASSUMED, q0_7=q7_seed,
            joint_bounds=dict(WRIST_JOINT_BOUNDS), waist_weight=WAIST_DLS_WEIGHT,
            wrist_weight=WRIST_DLS_WEIGHT, tol_p=TOL_P_M, tol_w=TOL_W_RAD, w_rot=1.0,
            max_iters=max_iters)
    return q7, info
AIM_LENGTH_M = 0.1  # lever arm for the aim-point trick -- arbitrary but nonzero;
                     # only scales how strongly axis error influences the DLS
                     # step relative to position error, not a physical dimension

WRIST_JOINT_BOUNDS = {}


# ---------------------------------------------------------------------------
# Support surface (the stool/table the object stands on) -- added 2026-09-18.
#
# The object is ~4.3cm wide but sits on a stool ~36cm across, and the robot
# stands right against it. Two consequences the old sequence ignored:
#   * The old first pose (READY_POSE) swung the hand forward, OVER the stool,
#     at roughly the stool's height, from an arm that starts hanging beside
#     the hip BELOW the stool top -- i.e. straight through the stool's rim.
#   * pre_grasp (12cm to the side of the object along GRASP_NORMAL) is still
#     above the stool top, so the hand can't rise into it from below either.
# Fix: detect the support surface from the depth cloud (a horizontal plane
# just below the object), then route the hand OUTSIDE its rim on the
# approach side and ABOVE it before coming in:
#   side_up      -- arm lifted out to the robot's right, beside the torso,
#                   palm VIA_HEIGHT_ABOVE_SUPPORT_M above the support and
#                   SUPPORT_RIM_MARGIN_M beyond the rim
#   approach_out -- same height/rim clearance, slid forward toward the
#                   object's X as far as reach allows
#   pre_grasp / grasp / post_grasp -- as before
# and the same via points in reverse on the way out. Every waypoint's palm
# centre is also checked against the detected footprint in the pre-flight
# (HAND_HALF_HEIGHT_M below the palm centre must clear the surface).
# If the plane can't be found (too few points, e.g. a table that fills the
# whole view), a SUPPORT_RADIUS_DEFAULT_M disk centred on the object is
# assumed and the run says so.
SUPPORT_RADIUS_DEFAULT_M = 0.18   # 36cm stool, assumed centred on the object when not seen
SUPPORT_RIM_MARGIN_M = 0.06       # via points sit this far beyond the detected rim
SUPPORT_SLAB_HALF_M = 0.015       # +-1.5cm band around the plane counts as "surface"
SUPPORT_SEARCH_RADIUS_M = 0.45    # only look for the plane this close (horizontally) to the object
SUPPORT_MIN_POINTS = 150
HAND_HALF_HEIGHT_M = 0.05         # lowest part of the open hand hangs ~5cm below the palm centre
VIA_HEIGHT_ABOVE_SUPPORT_M = 0.20 # palm centre height above the support at side_up/approach_out
SIDE_UP_X_M = 0.05                # palm X (ahead of the shoulder) for side_up
VIA_MAX_WRIST_REACH_M = 0.39      # slide a via point back toward the shoulder until its wrist is within this
SCENE_CLOUD_STRIDE = 3            # subsample the full depth image for the support search


@dataclass
class SupportSurface:
    top_z: float               # surface height, shoulder frame
    footprint_xy: np.ndarray   # (N,2) horizontal samples of the surface, shoulder frame
    rim_dist_along: float      # how far the surface extends from the object along the approach direction
    source: str


def _scene_cloud(depth_m, intrinsics, T_cam_to_shoulder):
    m = np.zeros(depth_m.shape, dtype=bool)
    m[::SCENE_CLOUD_STRIDE, ::SCENE_CLOUD_STRIDE] = True
    return mask_to_point_cloud(m, depth_m, intrinsics, T_cam_to_shoulder, min_valid_ratio=0.0)


def _default_support(obj_point, object_points, approach_dir_xy):
    # lowest VISIBLE object point minus 2cm -- the camera looks down on the
    # object, so its true base is somewhat below what it sees
    bottom = (float(np.percentile(object_points[:, 2], 5)) - 0.02 if object_points is not None and len(object_points)
              else float(obj_point[2]) - 0.06)
    g_ = np.arange(-SUPPORT_RADIUS_DEFAULT_M, SUPPORT_RADIUS_DEFAULT_M + 1e-9, 0.02)
    gx, gy = np.meshgrid(g_, g_)
    disk = np.stack([gx.ravel(), gy.ravel()], 1)
    disk = disk[np.linalg.norm(disk, axis=1) <= SUPPORT_RADIUS_DEFAULT_M] + obj_point[:2]
    return SupportSurface(bottom, disk, SUPPORT_RADIUS_DEFAULT_M,
                          f"DEFAULT disk r={SUPPORT_RADIUS_DEFAULT_M:.2f}m centred on the object (plane not found)")


def estimate_support_surface(scene_points, object_points, obj_point, approach_dir_xy) -> SupportSurface:
    """Horizontal plane the object stands on, from the (subsampled) full depth
    cloud in the shoulder frame: take points within SUPPORT_SEARCH_RADIUS_M
    horizontally of the object, between 2cm and 25cm below the object's
    detected centre (the floor is ~0.7m down and is excluded by that), drop
    the object's own footprint, and pick the most populated 1cm height bin.
    Points within SUPPORT_SLAB_HALF_M of it are the surface; its extent
    along approach_dir_xy (from the object) is the rim distance the via
    points must clear. Falls back to _default_support() with a loud
    source string if that fails."""
    d = np.asarray(approach_dir_xy, dtype=float); d = d / (np.linalg.norm(d) + 1e-12)
    if scene_points is None or len(scene_points) < SUPPORT_MIN_POINTS:
        return _default_support(obj_point, object_points, d)
    P = scene_points
    horiz = np.linalg.norm(P[:, :2] - obj_point[:2], axis=1)
    sel = P[(horiz < SUPPORT_SEARCH_RADIUS_M) & (horiz > 0.04)
            & (P[:, 2] < obj_point[2] - 0.02) & (P[:, 2] > obj_point[2] - 0.25)]
    if len(sel) < SUPPORT_MIN_POINTS:
        return _default_support(obj_point, object_points, d)
    edges = np.arange(sel[:, 2].min(), sel[:, 2].max() + 0.01, 0.01)
    hist, _ = np.histogram(sel[:, 2], edges)
    k = int(np.argmax(hist))
    z0 = 0.5 * (edges[k] + edges[k + 1])
    plane = sel[np.abs(sel[:, 2] - z0) < SUPPORT_SLAB_HALF_M]
    if len(plane) < SUPPORT_MIN_POINTS:
        return _default_support(obj_point, object_points, d)
    top_z = float(np.median(plane[:, 2]))
    rel = plane[:, :2] - obj_point[:2]
    rim = float(np.max(rel @ d))
    if len(plane) > 4000:
        plane = plane[np.random.default_rng(0).choice(len(plane), 4000, replace=False)]
    return SupportSurface(top_z, plane[:, :2], max(rim, 0.0),
                          f"vision ({len(plane)} pts, extent seen: fwd {rel[:,0].max():+.2f}/{rel[:,0].min():+.2f}m, "
                          f"lateral {rel[:,1].max():+.2f}/{rel[:,1].min():+.2f}m from the object)")


def build_via_waypoints(waypoints, obj_point, support: SupportSurface):
    """side_up + approach_out Waypoints (wrist targets, same R as pre_grasp)
    to prepend to [pre_grasp, grasp, post_grasp] -- see the module comment
    above. Slides each via point back toward the shoulder in X until its
    wrist target is within VIA_MAX_WRIST_REACH_M, so a wide stool doesn't
    put the via point out of reach."""
    pre = next(w for w in waypoints if w.name == "pre_grasp")
    grasp = next(w for w in waypoints if w.name == "grasp")
    R = pre.R
    d = (pre.position - grasp.position)[:2]
    d = d / (np.linalg.norm(d) + 1e-12)                  # approach side, horizontal
    hand_off = R @ HAND_OFFSET_LOCAL                      # palm centre = wrist + hand_off
    pre_palm_z = float((pre.position + hand_off)[2])
    z_via = max(support.top_z + VIA_HEIGHT_ABOVE_SUPPORT_M, pre_palm_z + 0.02)
    out_xy = obj_point[:2] + d * (support.rim_dist_along + SUPPORT_RIM_MARGIN_M)

    def feasible(wrist):
        """Same IK + body-clearance test the pre-flight applies, silently."""
        q7_seed = np.array([READY_POSE[n] for n in RIGHT_ARM_JOINT_NAMES])
        q7, info = solve_waypoint_ik(wrist, R, q7_seed, max_iters=600)
        return info.converged and not body_clearance_violations(q7)

    def via(name, palm_xy):
        # Keep the lateral (rim) clearance fixed; if the point is out of
        # reach, first go HIGHER (also more clearance above the stool), then
        # slide back toward the shoulder in X. Feasibility is the real IK +
        # body-clearance test, not a reach-sphere guess.
        palm = np.array([palm_xy[0], palm_xy[1], z_via])
        raised, slid = 0.0, 0.0
        z_max = support.top_z + VIA_HEIGHT_ABOVE_SUPPORT_M + 0.25
        while not feasible(palm - hand_off):
            if palm[2] + 0.02 <= z_max:
                palm[2] += 0.02; raised += 0.02
            elif palm[0] - 0.02 >= SIDE_UP_X_M:
                palm[0] -= 0.02; slid += 0.02
            else:
                print(f"[WARN] {name}: no reachable via point found beyond the rim -- pre-flight will "
                      f"report it; the object/stool is probably too far to the robot's right")
                break
        if raised or slid:
            print(f"[INFO] {name}: raised {raised*100:.0f}cm / slid {slid*100:.0f}cm back toward the "
                  f"shoulder to find a reachable, body-clear pose")
        return Waypoint(name=name, position=palm - hand_off, R=R.copy())

    approach_out = via("approach_out", out_xy)
    side_up = via("side_up", np.array([SIDE_UP_X_M, out_xy[1]]))
    print(f"[INFO] support surface: top z={support.top_z:+.3f} (object centre z={obj_point[2]:+.3f}), "
          f"rim {support.rim_dist_along*100:.0f}cm from the object on the approach side -- {support.source}")
    print(f"[INFO] via palm height z={z_via:+.3f} ({(z_via-support.top_z)*100:.0f}cm above the support); "
          f"approach_out palm xy={(approach_out.position+hand_off)[:2].round(3)}, "
          f"side_up palm xy={(side_up.position+hand_off)[:2].round(3)}")
    return [side_up, approach_out]


def support_clearance(q7, support: SupportSurface, waist_q=WAIST_Q_ASSUMED):
    """(over_surface, clearance_m): is the palm centre horizontally over the
    detected footprint (within 4cm of a sample), and how far the lowest part
    of the hand (palm centre - HAND_HALF_HEIGHT_M) is above the surface."""
    T_w = pik.fk_right_arm_remote(q7, waist_q=waist_q)
    palm = T_w[:3, 3] + T_w[:3, :3] @ HAND_OFFSET_LOCAL
    dmin = float(np.min(np.linalg.norm(support.footprint_xy - palm[:2], axis=1)))
    return dmin <= 0.04, float(palm[2] - HAND_HALF_HEIGHT_M - support.top_z)


# Sanity gate on the DETECTED position, before any IK/pre-flight: the
# arm's measured max reach is ~0.41 m from the shoulder pivot (see
# GRASP_TARGET_OFFSET's comment). A "cube" reported well beyond that isn't
# a borderline-reach problem for IK to grind on for 3x2000 iterations, it's
# a wrong detection (2026-09-17: background clutter 3 m away out-scored the
# real object -- CubeSegmenter.detect() now depth-gates candidates, this is
# the belt to that suspender). Generous on purpose: only catches the
# obviously-not-the-object case, real borderline targets still go through
# preflight_ik_check() as before.
MAX_CUBE_DIST_FROM_SHOULDER_M = 0.9


def _cube_within_reach_envelope(p_shoulder: np.ndarray) -> bool:
    dist = float(np.linalg.norm(p_shoulder))
    if dist <= MAX_CUBE_DIST_FROM_SHOULDER_M:
        return True
    print(f"[ABORT] detected object is {dist:.2f}m from the shoulder pivot (limit "
          f"{MAX_CUBE_DIST_FROM_SHOULDER_M:.2f}m, arm reach ~0.41m) -- that is not a graspable "
          f"target, almost certainly a false detection on background clutter (look at what "
          f"else is in the camera's view: test_perception_only.py shows every candidate). "
          f"Not running pre-flight IK, not touching hardware.")
    return False


def detect_and_localize(cam, seg: CubeSegmenter, timeout_s: float):
    """Duplicated from main_grasp_loop.py (not imported) so this script
    never pulls in g1_controller -- a different, unused controller stack
    here -- same rationale dry_run_grasp.py already used.

    Uses compute_T_cam_to_shoulder_pin() (Pinocchio, via the same remote
    server this script's IK already requires -- see module docstring) so
    perception and IK share one kinematic engine/live model AND the same
    "shoulder frame" (see WAIST_Q_ASSUMED's own comment on why both use the
    same waist assumption). Returns Localization3D.point_base in that
    shoulder frame despite the field's generic ("pelvis frame" elsewhere in
    this repo) name -- it's frame-agnostic, entirely determined by whatever
    T_cam_to_X is passed to mask_to_3d_point() below."""
    T_cam_to_shoulder = compute_T_cam_to_shoulder_pin(*WAIST_Q_ASSUMED)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        result = cam.read()
        if result is None:
            continue
        color, depth_m, intrinsics = result
        det = seg.detect(color, depth_m)
        if det is None:
            continue
        loc = mask_to_3d_point(det.mask, depth_m, intrinsics, T_cam_to_shoulder)
        if loc is None or loc.confidence < MIN_CONFIDENCE:
            continue
        loc.yolo_score = det.score            # for the [INFO] line in main() -- Localization3D's
        loc.median_depth_m = det.median_depth_m  # own .confidence is the valid-DEPTH ratio, not YOLO's
        loc.object_points = mask_to_point_cloud(det.mask, depth_m, intrinsics, T_cam_to_shoulder)
        loc.scene_points = _scene_cloud(depth_m, intrinsics, T_cam_to_shoulder)  # for the support surface
        return loc
    return None


def detect_and_localize_cloud(cam, seg: CubeSegmenter, timeout_s: float):
    
    T_cam_to_shoulder = compute_T_cam_to_shoulder_pin(*WAIST_Q_ASSUMED)
    camera_position_shoulder = T_cam_to_shoulder[:3, 3]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        result = cam.read()
        if result is None:
            continue
        color, depth_m, intrinsics = result
        det = seg.detect(color, depth_m)
        if det is None:
            continue
        loc = mask_to_3d_point(det.mask, depth_m, intrinsics, T_cam_to_shoulder)
        if loc is None or loc.confidence < MIN_CONFIDENCE:
            continue
        points_shoulder = mask_to_point_cloud(det.mask, depth_m, intrinsics, T_cam_to_shoulder)
        if points_shoulder is None:
            continue
        loc.yolo_score = det.score
        loc.median_depth_m = det.median_depth_m
        loc.object_points = points_shoulder
        loc.scene_points = _scene_cloud(depth_m, intrinsics, T_cam_to_shoulder)  # for the support surface
        return loc, points_shoulder, camera_position_shoulder
    return None


# Body-clearance check for pre-flight IK solutions (added 2026-09-17,
# switched from bounding boxes to mesh distance 2026-09-18).
#
# Incident: grasp_ph.py run 2026-09-17 16:37 -- all three waypoints
# "CONVERGED" in pre-flight, yet on hardware the hand ended up ~13cm below
# and ~15cm behind the grasp point, hanging beside the right hip. FK of the
# commanded joint solution (grasp_logs/arm_positions_20260917_163729.csv vs
# the IK solution for that run's target) put the ELBOW at pelvis-frame
# (X=0.00, Y=-0.04, Z=0.135) and the wrist-roll link at (0.06, 0.00, 0.07):
# inside the torso shell. Pinocchio IK has no self-collision model, so a
# target in front of the belly/midline "converges" to an arm folded through
# the chest; the real arm stops against the torso, the shoulder roll never
# adducts, and the hand drops down the side of the body. From the outside
# that looked exactly like "the Z is calculated too low".
#
# Test: sample points along the right arm's axis (shoulder_yaw -> elbow ->
# wrist_roll -> wrist_yaw, plus the palm centre) via the same remote
# Pinocchio FK the IK runs on, and require each to be at least
# BODY_MIN_AXIS_CLEARANCE_M from the torso + pelvis mesh vertices
# (unitree_g1_inspire_assets/meshes, expressed in the pelvis frame at
# waist=0; torso_link origin = pelvis + (-0.004, 0, 0.044)). Calibration
# (2026-09-18, axis-to-mesh distance of the closest sampled point):
#   poses realized on hardware: READY 5.8cm, LIFT 7.2cm, TRAJ_POSE_3 5.0cm,
#                               REFERENCE_SEED_POSE 4.9cm
#   16:37 run (arm jammed):     pre_grasp 0.3cm, grasp 0.3cm, post 2.1cm
#   09-18 run (object 12.6cm right of midline): grasp 2.8cm, post 1.9cm
# The upper arm/forearm are ~3.5cm in radius, so 4.5cm keeps ~1cm of real
# air and still passes every known-good pose. An axis-aligned box version
# of this check (kept below as BODY_BOXES_PELVIS, used only if the meshes
# can't be found) over-rejected near the torso's rounded front edge.
BODY_MIN_AXIS_CLEARANCE_M = 0.045
BODY_MESHES = ["torso_link_rev_1_0.STL", "pelvis.STL", "pelvis_contour_link.STL"]
_TORSO_LINK_IN_PELVIS = np.array([-0.0039635, 0.0, 0.044])  # waist joint origins at q=0
_MESH_DIR = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 3), "unitree_g1_inspire_assets", "meshes"))
# Fallback only (see above): (xmin, xmax, ymin, ymax, zmin, zmax), pelvis frame.
BODY_BOXES_PELVIS = {
    "torso":  (-0.071, 0.080, -0.108, 0.108, 0.035, 0.356),
    "pelvis": (-0.061, 0.061, -0.068, 0.068, -0.152, 0.035),
}
BODY_CLEARANCE_MARGIN_M = 0.03  # fallback box inflation
_ARM_CLEARANCE_FRAMES = ["right_shoulder_yaw_link", "right_elbow_link",
                         "right_wrist_roll_link", "right_wrist_yaw_link"]
_body_vertices_cache = None


def _load_stl_vertices(path):
    import struct
    data = open(path, "rb").read()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        return np.array([list(map(float, l.split()[1:4])) for l in data.decode(errors="ignore").splitlines()
                         if l.strip().startswith("vertex")])
    n = struct.unpack("<I", data[80:84])[0]
    arr = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return arr["v"].reshape(-1, 3).astype(float)


def body_vertices_pelvis():
    """(N,3) torso+pelvis mesh vertices in the pelvis frame (1mm-deduplicated,
    cached). Returns None if the mesh files aren't available."""
    global _body_vertices_cache
    if _body_vertices_cache is None:
        parts = []
        for name in BODY_MESHES:
            path = os.path.join(_MESH_DIR, name)
            if not os.path.exists(path):
                print(f"[WARN] body mesh {path} not found -- body clearance falls back to boxes")
                _body_vertices_cache = False
                return None
            v = np.unique(np.round(_load_stl_vertices(path), 3), axis=0)
            if name.startswith("torso"):
                v = v + _TORSO_LINK_IN_PELVIS
            parts.append(v)
        _body_vertices_cache = np.vstack(parts)
    return None if _body_vertices_cache is False else _body_vertices_cache


def _arm_sample_points(q7, waist_q):
    t_shoulder = np.asarray(pik.shoulder_origin_remote(waist_q))
    origins = [(f, pik.fk_right_arm_remote(q7, waist_q=waist_q, frame_name=f)[:3, 3] + t_shoulder)
               for f in _ARM_CLEARANCE_FRAMES]
    pts = []
    for (fa, pa), (fb, pb) in zip(origins[:-1], origins[1:]):
        pts.append((fa, pa))
        for k in (0.25, 0.5, 0.75):  # along each arm segment's axis
            pts.append((f"{fa}->{fb} @{k:.2f}", (1 - k) * pa + k * pb))
    pts.append(origins[-1])
    T_w = pik.fk_right_arm_remote(q7, waist_q=waist_q)
    pts.append(("palm_centre", T_w[:3, 3] + T_w[:3, :3] @ HAND_OFFSET_LOCAL + t_shoulder))
    return pts


def body_clearance_violations(q7, waist_q=WAIST_Q_ASSUMED, min_clearance=BODY_MIN_AXIS_CLEARANCE_M,
                              margin=BODY_CLEARANCE_MARGIN_M):
    """Returns a list of (point_name, pelvis_xyz, distance_m) for every sampled
    arm-axis point closer than min_clearance to the torso/pelvis mesh --
    empty list = clear. Falls back to the inflated-box test (distance
    reported as nan) if the meshes aren't found."""
    pts = _arm_sample_points(q7, waist_q)
    verts = body_vertices_pelvis()
    bad = []
    if verts is not None:
        for name, p in pts:
            d = float(np.min(np.linalg.norm(verts - p, axis=1)))
            if d < min_clearance:
                bad.append((name, p, d))
        return bad
    for name, p in pts:
        for box, (x0, x1, y0, y1, z0, z1) in BODY_BOXES_PELVIS.items():
            if (x0 - margin <= p[0] <= x1 + margin and y0 - margin <= p[1] <= y1 + margin
                    and z0 - margin <= p[2] <= z1 + margin):
                bad.append((name, p, float("nan")))
                break
    return bad


def body_min_clearance(q7, waist_q=WAIST_Q_ASSUMED):
    """Smallest axis-to-body-mesh distance over the sampled arm points (m),
    or None without meshes -- for the [PREFLIGHT] summary line."""
    verts = body_vertices_pelvis()
    if verts is None:
        return None
    return min(float(np.min(np.linalg.norm(verts - p, axis=1))) for _, p in _arm_sample_points(q7, waist_q))


def preflight_ik_check(waypoints, seed_pose=None, check_body_clearance=True, support=None,
                       enforce_clearance=True):
    """Solves every waypoint with the remote Pinocchio IK and, for each
    converged solve, also rejects it if body_clearance_violations() says the
    arm would pass through the torso/pelvis (see BODY_BOXES_PELVIS's comment
    for the incident that motivated this). enforce_clearance=False
    (--skip_clearance_check) still runs and PRINTS the body/support clearance
    numbers but never fails a waypoint on them -- only a non-converged IK
    solve fails then. Returns (all_ok, q7_solutions)."""
    skip_note = "" if enforce_clearance else " (ignored: --skip_clearance_check)"
    if not pik.ping():
        raise pik.KinematicsServerNotRunning()

    seed = seed_pose if seed_pose is not None else READY_POSE
    q7_seed = np.array([seed[name] for name in RIGHT_ARM_JOINT_NAMES])

    ik_all_converged = True
    q7_solutions = []
    for wp in waypoints:
        q7, info = solve_waypoint_ik(wp.position, wp.R, q7_seed)
        status = "CONVERGED" if info.converged else "DID NOT CONVERGE"
        if ORIENTATION_MODE == "aim":
            rot_txt = f"axis_err={info.final_errors.get('axis_err_deg', 0):.1f}deg"
        else:
            rot_txt = f"rot_err={np.degrees(info.final_errors.get('eR_w_norm', 0)):.1f}deg"
        # where the PALM centre actually lands for this solution vs the plan
        T_fk = pik.fk_right_arm_remote(q7, waist_q=WAIST_Q_ASSUMED)
        palm_fk = T_fk[:3, 3] + T_fk[:3, :3] @ HAND_OFFSET_LOCAL
        palm_plan = wp.position + wp.R @ HAND_OFFSET_LOCAL
        hand_tilt = np.degrees(np.arcsin(np.clip(T_fk[2, 0], -1, 1)))   # + = fingers above the wrist
        print(f"[PREFLIGHT] {wp.name}: target={wp.position.round(3)} IK: {status} "
              f"({info.iters} iters, pos_err(R)={info.final_errors.get('eR_p_norm', 0):.4f}m, {rot_txt}); "
              f"palm centre off plan by {np.linalg.norm(palm_fk - palm_plan)*100:.1f}cm "
              f"(dZ {(palm_fk - palm_plan)[2]*100:+.1f}cm), hand {hand_tilt:+.0f}deg nose-up")
        wp_ok = info.converged
        if info.converged and check_body_clearance:
            hits = body_clearance_violations(q7)
            clearance = body_min_clearance(q7)
            if clearance is not None:
                print(f"[PREFLIGHT] {wp.name}: arm-to-body clearance {clearance*100:.1f}cm "
                      f"(needs >= {BODY_MIN_AXIS_CLEARANCE_M*100:.1f}cm)")
            if hits:
                wp_ok = wp_ok and not enforce_clearance
                worst = min(hits, key=lambda h: h[2])
                print(f"[PREFLIGHT] {wp.name}: {'REJECTED' if enforce_clearance else 'CLEARANCE FAIL' + skip_note}"
                      f" -- arm would press into the torso/pelvis: "
                      f"closest point {worst[0]} at pelvis {worst[1].round(3)}, {worst[2]*100:.1f}cm from "
                      f"the body surface. On hardware the arm stops against the body and the hand "
                      f"hangs low beside the hip. Move the object further to the robot's right "
                      f"(outboard of the right shoulder); roughly 1cm of object shift buys ~0.3cm "
                      f"of clearance here.")
        if info.converged and support is not None:
            over, clr = support_clearance(q7, support)
            if over:
                print(f"[PREFLIGHT] {wp.name}: hand over the support surface, lowest point "
                      f"{clr*100:+.1f}cm above it")
                if clr < 0.0 and wp.name == "grasp":
                    # the grasp is at object mid-height by design, so the
                    # fingertips end up near the surface for a short object
                    # -- warn, don't abort (HAND_HALF_HEIGHT_M is a bound)
                    print(f"[PREFLIGHT] {wp.name}: WARNING -- fingertips may touch the stool/table top "
                          f"({-clr*100:.1f}cm by the HAND_HALF_HEIGHT_M bound); fine for a light touch, "
                          f"raise GRASP_TARGET_OFFSET[2] if the fingers drag on the surface")
                elif clr < 0.0:
                    wp_ok = wp_ok and not enforce_clearance
                    print(f"[PREFLIGHT] {wp.name}: {'REJECTED' if enforce_clearance else 'CLEARANCE FAIL' + skip_note}"
                          f" -- the hand would hit the stool/table top ({-clr*100:.1f}cm into it). "
                          f"Raise the object or VIA_HEIGHT_ABOVE_SUPPORT_M.")
        ik_all_converged = ik_all_converged and wp_ok
        q7_solutions.append(q7.copy())
        # Only advance the seed on a converged solve -- a non-converged solve
        # can leave the arm in a contorted, near-singular pose (see
        # 2026-09-17 incident: a borderline-unreachable "grasp" waypoint's
        # failed solution, fed forward as "post_grasp"'s seed, made an
        # otherwise-easy post_grasp target fail too). Keep retrying every
        # subsequent waypoint from the last KNOWN-GOOD seed instead.
        if info.converged:
            q7_seed = q7
    return ik_all_converged, q7_solutions


def build_segments(q7_solutions):
    open_q = phctrl.HAND12_LIMITS_LO.copy()

    close_q = open_q.copy()
    close_q[6:] = phctrl.pose_to_hand12_rad(phctrl.INSPIRE_POSES["grip_cylinder"])[6:]
    grasp_tau = np.zeros(12)
    grasp_tau[6:] = GRASP_TORQUE_SCALE * _GRASP_TAU_R

    q7_side, q7_out, q7_pre, q7_grasp, q7_post = q7_solutions
    arm = lambda q: dict(zip(RIGHT_ARM_JOINT_NAMES, q))

    return [
        phctrl.Segment("hold", 2.0, hand_mode="hold"),
        # First pose (2026-09-18): NOT READY_POSE any more -- that swung the
        # hand forward over the stool at stool height from an arm hanging
        # below it. side_up lifts the arm out to the right, beside the
        # torso, above the support and beyond its rim (build_via_waypoints).
        phctrl.Segment("side_up", 3.0, arm_targets=arm(q7_side), hand_mode="position", hand_target=open_q),
        phctrl.Segment("approach_out", 3.0, arm_targets=arm(q7_out), hand_mode="position", hand_target=open_q),
        phctrl.Segment("approach", 3.0, arm_targets=arm(q7_pre), hand_mode="position", hand_target=open_q),
        phctrl.Segment("descend", 2.0, arm_targets=arm(q7_grasp), hand_mode="position", hand_target=open_q),
        phctrl.Segment("grasp", 3.0, arm_targets=dict(zip(RIGHT_ARM_JOINT_NAMES, q7_grasp)),
                       hand_mode="position", hand_target=close_q),
        phctrl.Segment("grasp_hold", 1.5, hand_mode="force", hand_target=grasp_tau),
        # Pick-and-place-back (2026-09-18): the task is ONLY "lift the object
        # and put it back where it was". Lift is the pre-flight-checked
        # post_grasp waypoint (straight up above the grasp point, same X/Y),
        # not the old fixed LIFT_POSE swing-out; then lower back to the very
        # same grasp joint solution, open, and retreat along the approach
        # path. hand_mode="hold" keeps the grasp_hold force mode active
        # through lift/lower so the grip doesn't relax mid-air.
        phctrl.Segment("lift", 3.0, arm_targets=dict(zip(RIGHT_ARM_JOINT_NAMES, q7_post)),
                       hand_mode="hold"),
        phctrl.Segment("lift_hold", 1.5, hand_mode="hold"),
        phctrl.Segment("lower", 3.0, arm_targets=dict(zip(RIGHT_ARM_JOINT_NAMES, q7_grasp)),
                       hand_mode="hold"),
        phctrl.Segment("release", 2.5, hand_mode="position", hand_target=open_q),
        phctrl.Segment("retreat", 3.0, arm_targets=arm(q7_pre), hand_mode="position", hand_target=open_q),
        phctrl.Segment("retreat_out", 3.0, arm_targets=arm(q7_out), hand_mode="position", hand_target=open_q),
        # Last segment is where Custom ramps the arm_sdk weight back to 0
        # (disable_ramp_T): side_up is beside the torso and outside the
        # stool's rim, so the whole-body controller pulls the arm straight
        # down past the stool, not through it.
        phctrl.Segment("side_down", 3.0, arm_targets=arm(q7_side), hand_mode="hold"),
    ]


def main():
    global ORIENTATION_MODE, HAND_PITCH_UP_DEG
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network_interface", nargs="?", default=None,
                         help="NIC on the robot's LAN (e.g. eno1/eth0) -- used for both the "
                              "depth camera DDS relay and the G1 SDK.")
    parser.add_argument("--left_hand_ip", type=str, default="192.168.123.210")
    parser.add_argument("--right_hand_ip", type=str, default="192.168.123.211")
    parser.add_argument("--admit_mass", type=float, default=7.0)
    parser.add_argument("--admit_damping", type=float, default=12.0)
    parser.add_argument("--admit_ktrack", type=float, default=2.0)
    parser.add_argument("--force_guard_frac", type=float, default=0.75)
    parser.add_argument("--hand_qdot_max", type=float, default=1.5)
    parser.add_argument("--modbus_speed", type=int, default=450)
    parser.add_argument("--modbus_force_limit_g", type=int, default=300)
    parser.add_argument("--seed_pose", action="store_true",
                         help="Seed preflight_ik_check()'s IK solve from REFERENCE_SEED_POSE's "
                              "right-arm entries instead of READY_POSE's. Only changes which of "
                              "IK's multiple valid solutions the 7-DOF solve converges to -- the "
                              "real robot's \"ready\" hold segment still targets READY_POSE either "
                              "way (see preflight_ik_check()'s own docstring). Off by default -- "
                              "today's READY_POSE-seeded behavior is unchanged unless this is "
                              "passed.")
    parser.add_argument("--save_video", action="store_true",
                         help="Record the live view to grasp_videos/ (off by default).")
    parser.add_argument("--geometric_normal", action="store_true",
                         help="Estimate the approach normal/closing axis from the object's real "
                              "point-cloud geometry (grasp_pose_generator.generate_geometric_grasp) "
                              "instead of the hardcoded GRASP_NORMAL/GRASP_FORWARD_REF constants "
                              "above. Off by default -- today's hardcoded-constant behavior is "
                              "unchanged unless this is passed.")
    parser.add_argument("--estimator", choices=["graspgen", "pca"], default="graspgen",
                         help="Only used with --geometric_normal. 'graspgen' tries the NVlabs/"
                              "GraspGen server first and falls back to the no-GPU PCA estimator "
                              "on any failure (unreachable server, low confidence); 'pca' skips "
                              "GraspGen entirely, e.g. for a bench test without the Docker server "
                              "running.")
    parser.add_argument("--aim_mode", action="store_true",
                         help="Use the old palm-normal-only ('aim') IK constraint instead of the full "
                              "3-DOF hand orientation. Leaves the hand's roll free, which is what let "
                              "the palm land 4-9cm off the planned height (see HAND_PITCH_UP_DEG's "
                              "comment). Off by default.")
    parser.add_argument("--hand_pitch_up_deg", type=float, default=HAND_PITCH_UP_DEG,
                         help="Planned nose-UP pitch of the hand in degrees (full-orientation mode only; "
                              "positive = fingertips above the wrist, negative = nose-down). Default "
                              f"{HAND_PITCH_UP_DEG:.0f}. More nose-up costs reach at the grasp (see "
                              "HAND_PITCH_UP_DEG's comment).")
    parser.add_argument("--skip_clearance_check", action="store_true",
                         help="Make the pre-flight's body-clearance (arm vs torso/pelvis mesh) and "
                              "support-clearance (hand vs detected stool/table top) checks advisory: "
                              "they are still computed and printed per waypoint, but a failing "
                              "clearance no longer aborts. A waypoint whose IK does NOT converge "
                              "still aborts (use --skip_ik_check for that). Off by default -- pass "
                              "this only when you've looked at the printed clearance numbers and "
                              "judged the margin acceptable (e.g. the upper arm brushing the torso).")
    parser.add_argument("--skip_ik_check", action="store_true",
                         help="Proceed to real hardware even if preflight_ik_check() reports a "
                              "waypoint did NOT converge, using whatever (possibly bounds-clipped/"
                              "unconverged) joint solution it last produced. Off by default -- "
                              "the preflight abort exists because an unconverged solve has "
                              "actually meant an unreachable/incorrect target in past runs (see "
                              "preflight_ik_check()'s own docstring). Only pass this when you've "
                              "inspected the printed per-waypoint pos_err/axis_err yourself and "
                              "judged it safe to proceed anyway.")
    args = parser.parse_args()

    ORIENTATION_MODE = "aim" if args.aim_mode else "full"
    HAND_PITCH_UP_DEG = args.hand_pitch_up_deg
    print(f"[INFO] hand orientation: {ORIENTATION_MODE} mode"
          + ("" if args.aim_mode else f", planned {HAND_PITCH_UP_DEG:.0f}deg nose-up"))

    if not pik.ping():
        print(f"[ABORT] pin_kinematics_server.py isn't running (checked {pik.SOCK_PATH!r}). "
              "Start it first, in its own terminal, under the conda `pinocchio` env:\n"
              "    scripts/g1_control/ph/pin_kinematics_server.py")
        return

    cam = open_camera(network_interface=args.network_interface, local=False)
    seg = CubeSegmenter(DEFAULT_MODEL_PATH)

    try:
        print("[INFO] detecting cube...")
        if args.geometric_normal:
            det_result = detect_and_localize_cloud(cam, seg, DETECT_TIMEOUT_S)
            if det_result is None:
                print("[ABORT] no confident cube detection within timeout")
                return
            loc, points_shoulder, camera_position_shoulder = det_result
            print(f"[INFO] cube at {loc.point_base.round(3)} (shoulder frame), "
                  f"valid-depth ratio={loc.confidence:.2f}, yolo score={loc.yolo_score:.2f}, "
                  f"camera depth={loc.median_depth_m:.2f}m, cloud size={points_shoulder.shape[0]} points")
            if not _cube_within_reach_envelope(loc.point_base):
                return

            # grasp_pose_generator.py's own field names still say "_pelvis"
            # (that module is untouched -- see this script's own module
            # docstring on scope) but are frame-agnostic in practice, same
            # as Localization3D.point_base: whatever frame the points going
            # in are expressed in is the frame the outputs come back in.
            # points_shoulder/camera_position_shoulder above make that
            # SHOULDER frame here.
            candidate = grasp_pose_generator.generate_geometric_grasp(
                points_shoulder, camera_position_shoulder, prefer=args.estimator,
                closing_axis_hint=GEOMETRIC_CLOSING_AXIS_HINT)
            if not candidate.valid:
                print(f"[ABORT] geometric grasp-pose generation rejected this detection: "
                      f"{candidate.reason}")
                return
            print(f"[INFO] geometric grasp pose (estimator={candidate.estimator_used}, "
                  f"quality={candidate.quality:.2f}): surface_point={candidate.surface_point_pelvis.round(3)} "
                  f"normal={candidate.surface_normal_pelvis.round(3)} (shoulder frame)")
            waypoints = candidate.waypoints
            # Same pre_grasp hover-clearance bump as the plain (non-
            # geometric) path below -- generate_geometric_grasp() calls the
            # SAME trajectory_gen.generate_grasp_waypoints() internally, so
            # pre_grasp's Z falls out at (or very near) grasp height here
            # too whenever the estimated normal's Z component is small,
            # same root cause as the plain path's own comment on this.
            for _wp in waypoints:
                if _wp.name == "pre_grasp":
                    _wp.position = _wp.position + np.array([0.0, 0.0, PRE_GRASP_Z_CLEARANCE_M])
        else:
            loc = detect_and_localize(cam, seg, DETECT_TIMEOUT_S)
            if loc is None:
                print("[ABORT] no confident cube detection within timeout")
                return
            print(f"[INFO] cube at {loc.point_base.round(3)} (shoulder frame), "
                  f"valid-depth ratio={loc.confidence:.2f}, yolo score={loc.yolo_score:.2f}, "
                  f"camera depth={loc.median_depth_m:.2f}m")
            if not _cube_within_reach_envelope(loc.point_base):
                return

            Target = loc.point_base + GRASP_TARGET_OFFSET
            waypoints = generate_grasp_waypoints(Target, normal=GRASP_NORMAL, forward_ref=_tilted_forward_ref())
            # GRASP_NORMAL's Z component is small but nonzero (see its own
            # comment -- it's ~15 degrees off pure -Y, not pure -Y anymore
            # since the 2026-09-14 recalibration), so pre_grasp's Z is close
            # to but not exactly grasp's Z from approach_dist alone -- not
            # enough real hover on its own. Bump pre_grasp's Z up by a real
            # clearance margin here so "approach" hovers above grasp height
            # and "descend" (still targets the unchanged grasp waypoint) is
            # a genuine downward motion -- grasp/post_grasp positions (and
            # therefore GRASP_TARGET_OFFSET above) are untouched.
            for _wp in waypoints:
                if _wp.name == "pre_grasp":
                    _wp.position = _wp.position + np.array([0.0, 0.0, PRE_GRASP_Z_CLEARANCE_M])

        # Stool/table the object stands on -> via points outside its rim and
        # above it (see the SupportSurface comment block). Both detection
        # paths above leave loc.scene_points / loc.object_points behind.
        _pre = next(w for w in waypoints if w.name == "pre_grasp")
        _gr = next(w for w in waypoints if w.name == "grasp")
        support = estimate_support_surface(getattr(loc, "scene_points", None),
                                           getattr(loc, "object_points", None),
                                           loc.point_base, (_pre.position - _gr.position)[:2])
        waypoints = build_via_waypoints(waypoints, loc.point_base, support) + list(waypoints)

        print("[INFO] running pre-flight IK check via the Pinocchio server (no hardware touched yet)...")
        if args.skip_clearance_check:
            print("[WARNING] --skip_clearance_check: body/support clearance failures below will NOT abort.")
        ik_ok, q7_solutions = preflight_ik_check(
            waypoints, seed_pose=REFERENCE_SEED_POSE if args.seed_pose else None, support=support,
            enforce_clearance=not args.skip_clearance_check)
        if not ik_ok and not args.skip_ik_check:
            print("[ABORT] at least one waypoint failed the pre-flight check (did not converge, "
                  "or converged to a pose that passes through the torso -- see the per-waypoint "
                  "lines above). Not proceeding to real hardware. Reposition the object, or pass "
                  "--skip_ik_check to proceed anyway (see its --help for what that risks).")
            return
        if not ik_ok:
            print("[WARNING] --skip_ik_check passed -- proceeding to REAL hardware despite at "
                  "least one waypoint NOT converging in the pre-flight check. The joint solution "
                  "being used may be unreachable/incorrect (see the per-waypoint pos_err/axis_err "
                  "printed above). Keep the e-stop within reach.")
        else:
            print("[INFO] pre-flight IK check passed for all waypoints -- reusing these solved "
                  "joint angles directly for the real run, IK is not solved again for this detection.")

        print("WARNING: this drives REAL arm + hand hardware -- a real Cartesian reach "
              "followed by a real compliant grasp via g1_hw_ph_finger_control's PH "
              "admittance controller. --admit_mass/--admit_damping/--admit_ktrack/"
              "--force_guard_frac have not been bench-tuned. Clear the workspace and "
              "keep the e-stop within reach.")
        input("Press Enter to continue...")

        if args.network_interface:
            phctrl.ChannelFactoryInitialize(0, args.network_interface)
        else:
            phctrl.ChannelFactoryInitialize(0)

        hw_safety.flush_stale_arm_sdk()
        try:
            hw_safety.require_locked_standing()
        except hw_safety.RobotNotReadyError as e:
            print(f"[ABORT] {e}")
            return

        custom = phctrl.Custom(
            build_segments(q7_solutions),
            hand_ips={"L": args.left_hand_ip, "R": args.right_hand_ip},
            admit_mass=args.admit_mass, admit_damping=args.admit_damping,
            admit_ktrack=args.admit_ktrack, force_guard_frac=args.force_guard_frac,
            hand_qdot_max=args.hand_qdot_max,
            modbus_speed=args.modbus_speed, modbus_force_limit_g=args.modbus_force_limit_g,
            # use_waist here only matters for Custom's own arm_pR_target
            # IK-resolve fallback (_load_segment()) -- no segment
            # build_segments() makes ever sets arm_pR_target (the solved
            # joint angles are baked into arm_targets instead), so this is
            # inert in practice. False (not a CLI flag anymore -- the
            # right-arm-only IK has no waist DOF to move at all, stronger
            # than the old --use_waist=False default) passed through for
            # consistency anyway.
            use_waist=False,
            # Disabled (None). The wrist is NOT hard-locked anywhere in this
            # pipeline -- preflight_ik_check() already solves it freely
            # (subject only to WRIST_JOINT_BOUNDS), so this Custom-level
            # mechanism (which would unconditionally override wrist_roll on
            # EVERY segment, including "ready"/"Lift"'s own separately-
            # captured wrist values) has no role to play here.
            lock_right_wrist_roll=None,
        )
        custom.Init()

        # Pelvis-stability sanity check (diagnostic, NOT a hard abort -- see
        # pelvis_stability_check.py's own docstring) right before committing
        # to the pre-solved fixed-joint segments below: they were all solved
        # relative to a single pelvis-frame snapshot, which implicitly
        # assumes both feet are flat/fully planted. Confirms the IMU's real
        # measured orientation agrees with what leg-FK predicts under that
        # assumption -- catches a foot having come up on its edge, which
        # (2026-09-15 bench testing found) changes NEITHER leg NOR waist
        # joint encoder readings at all, since a rigid-body rock about a
        # foot's contact edge isn't produced by any actuated joint.
        _t0 = time.monotonic()
        while custom.low_state is None and time.monotonic() - _t0 < 3.0:
            time.sleep(0.05)
        if custom.low_state is not None:
            q_now = {name: custom.low_state.motor_state[i].q for i, name in enumerate(G1_JOINT_NAMES)}
            stable_ok, stable_msg = pelvis_stability_check.check_feet_flat(
                q_now, custom.low_state.imu_state.rpy)
            print(f"[INFO] {stable_msg}" if stable_ok else f"[WARNING] {stable_msg}")
        else:
            print("[WARNING] no rt/lowstate within 3.0s -- skipping feet-flat sanity check.")

        custom.Start()

        video_writer = None
        video_path = None
        last_video_write_t = 0.0
        if args.save_video:
            os.makedirs(VIDEO_DIR, exist_ok=True)
            video_path = os.path.join(VIDEO_DIR, f"grasp_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

        # Arm-position log (ALL 29 body joints -- legs+waist+arms, from
        # rt/lowstate, same G1_JOINT_NAMES/motor_state[i] convention as
        # log_joint_positions.py -- legs are included even though this
        # script never commands them, purely so you can directly confirm
        # from the log that the legs/waist actually stayed put during a
        # real run) and finger position/force log (12 HAND12 joints, from
        # the SAME Modbus reading Custom._tick() already takes every
        # control_dt for its own admittance step -- see
        # Custom.last_raw_L/R/last_f_L/R, read here rather than re-reading
        # Modbus a second time from this thread). Each finger row also
        # carries its fixed open/close angle (HAND12_LIMITS_LO/HI) so a
        # position can be read as "how open is this finger" without
        # cross-referencing the source code afterward.
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(LOG_DIR, exist_ok=True)
        arm_log_path = os.path.join(LOG_DIR, f"arm_positions_{run_ts}.csv")
        hand_log_path = os.path.join(LOG_DIR, f"hand_state_{run_ts}.csv")
        arm_log_f = open(arm_log_path, "w", newline="")
        arm_log_writer = csv.writer(arm_log_f)
        # q_des/q_err columns only exist for the 17 rt/arm_sdk-controlled joints
        # (phctrl.ARM_JOINT_NAMES == G1_JOINT_NAMES[12:29], verified index-for-
        # index equal) -- legs are never commanded by this script (see
        # READY_POSE's own comment), so they get no q_des of their own.
        arm_log_writer.writerow(
            ["timestamp", "t_s", "segment"] + G1_JOINT_NAMES
            + [f"{n}_qdes_rad" for n in phctrl.ARM_JOINT_NAMES]
            + [f"{n}_qerr_rad" for n in phctrl.ARM_JOINT_NAMES])
        hand_log_f = open(hand_log_path, "w", newline="")
        hand_log_writer = csv.writer(hand_log_f)
        hand_log_header = ["timestamp", "t_s", "segment"]
        for name in phctrl.HAND12_NAMES:
            # qdes_rad: the admittance-driven position reference actually being
            # sent to the hand this tick (Custom.hand_ctrl.q_cmd_hand12) --
            # NOT the fixed open/close pose the segment asked for, since the
            # admittance law ramps toward that with its own dynamics (see
            # HandAdmittancePH.step()'s own docstring). Valid in BOTH position
            # and force hand_mode -- q_cmd_hand12 is what step() always outputs
            # and is always what's written to the hand over Modbus.
            # tau_des_Nm: the force-mode spring target (PHContactForceController
            # tau_d) -- only meaningful while hand_mode="force" for this segment
            # (see build_segments()); 0 the rest of the time, same convention as
            # GRASP_TAU_D itself.
            hand_log_header += [f"{name}_pos_rad", f"{name}_force_g",
                                 f"{name}_open_rad", f"{name}_close_rad",
                                 f"{name}_qdes_rad", f"{name}_qerr_rad", f"{name}_tau_des_Nm"]
        hand_log_writer.writerow(hand_log_header)
        last_state_log_t = 0.0
        print(f"[INFO] logging arm positions to {arm_log_path}")
        print(f"[INFO] logging finger positions/forces to {hand_log_path}")

        print("[INFO] executing -- live camera view open. 'q' in the video window (or "
              "Ctrl+C in this terminal) stops early via the same graceful winddown as "
              "any other exit path -- it does not cut the arm immediately.")
        try:
            while True:
                seg_name = custom.segments[custom.seg_idx].name
                now_log = time.time()
                if now_log - last_state_log_t >= 1.0 / STATE_LOG_HZ:
                    last_state_log_t = now_log
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    if custom.low_state is not None:
                        q_body = [custom.low_state.motor_state[i].q for i in range(len(G1_JOINT_NAMES))]
                        # _q_cmd_last: the actual ratio-interpolated q_des this
                        # tick's rt/arm_sdk write used (Custom._tick(), ARM_JOINT_NAMES
                        # order) -- not just the segment's final target, so this
                        # traces the commanded ramp itself, not a step function.
                        # None only before the very first tick; falls back to the
                        # segment's (not-yet-interpolated) target for that one row.
                        q_des17 = (custom._q_cmd_last if custom._q_cmd_last is not None
                                   else custom.arm_q_target)
                        q_err17 = [q_body[12 + k] - float(q_des17[k]) for k in range(17)]
                        arm_log_writer.writerow(
                            [ts, f"{custom.t:.3f}", seg_name] + [f"{v:.6f}" for v in q_body]
                            + [f"{v:.6f}" for v in q_des17] + [f"{v:.6f}" for v in q_err17])
                        arm_log_f.flush()
                    if custom.last_raw_L is not None and custom.last_raw_R is not None:
                        pos12 = phctrl.hand12_raw_to_rad(
                            phctrl.modbus_to_hand12_order(custom.last_raw_L, custom.last_raw_R))
                        if custom.last_f_L is not None and custom.last_f_R is not None:
                            force12 = phctrl.modbus_to_hand12_order(custom.last_f_L, custom.last_f_R)
                        else:
                            force12 = [float("nan")] * 12
                        # q_cmd_hand12: the admittance reference actually sent to the
                        # hand this tick (None only before the first _tick() -- falls
                        # back to the measured pose, i.e. zero error, for that row).
                        # _tau_d: force-mode spring target, HAND12_NAMES order,
                        # PHContactForceController's own internal state -- see header
                        # comment above for why it's 0 outside force-mode segments.
                        q_des12 = (custom.hand_ctrl.q_cmd_hand12
                                   if custom.hand_ctrl.q_cmd_hand12 is not None else pos12)
                        tau_des12 = custom.ph_ctrl._tau_d
                        row = [ts, f"{custom.t:.3f}", seg_name]
                        for i in range(12):
                            row += [f"{pos12[i]:.6f}", f"{force12[i]:.1f}",
                                    f"{phctrl.HAND12_LIMITS_LO[i]:.4f}", f"{phctrl.HAND12_LIMITS_HI[i]:.4f}",
                                    f"{q_des12[i]:.6f}", f"{pos12[i] - q_des12[i]:.6f}",
                                    f"{tau_des12[i]:.4f}"]
                        hand_log_writer.writerow(row)
                        hand_log_f.flush()
                result = cam.read()
                if result is not None:
                    color, _depth_m, _intr = result
                    vis = color.copy()
                    cur_seg = custom.segments[custom.seg_idx]
                    seg_t = custom.t - custom.seg_t0
                    cv2.putText(vis, f"segment: {cur_seg.name} ({seg_t:4.1f}/{cur_seg.T:4.1f}s)",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(vis, f"target (shoulder frame): {loc.point_base.round(3)}",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                    cv2.imshow("grasp_ph -- live view (q = stop early)", vis)

                    if video_path is not None:
                        if video_writer is None:
                            h, w = vis.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (w, h))
                            print(f"[INFO] recording video to {video_path}")
                        now = time.time()
                        if now - last_video_write_t >= 1.0 / VIDEO_FPS:
                            video_writer.write(vis)
                            last_video_write_t = now
                if custom.done:
                    print("Done!")
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] 'q' pressed -- stopping early.")
                    break
        finally:
            cv2.destroyAllWindows()
            if video_writer is not None:
                video_writer.release()
                print(f"[INFO] video saved: {video_path}")
            arm_log_f.close()
            hand_log_f.close()
            print(f"[INFO] arm log saved: {arm_log_path}")
            print(f"[INFO] hand log saved: {hand_log_path}")
            custom.Shutdown()
    finally:
        cam.close()


if __name__ == "__main__":
    main()
