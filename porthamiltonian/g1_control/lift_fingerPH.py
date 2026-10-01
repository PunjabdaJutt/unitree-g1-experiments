# lift_fingerPH.py
"""
PH-only finger control variant of lift_fingerTorque.py.

- Waist + arms: position PID (unchanged).
- HAND12 (12 finger proximal/thumb-base joints): PH IDA-PBC gravity-comp PD is
  the SOLE actuator. No finger position PID, no tau-PI, no thumb-yaw hold.
- Segments still drive q_des for the fingers via joint_pos_overrides; PH tracks it.

  conda activate env_isaaclab
  python scripts/g1_control/lift_fingerPH.py --num_envs 1 --headless --enable_cameras \\
      --use_waist --ph_kp 4 --ph_kd 1.4 --ph_ki 0 --ph_i_clamp 0.35

  --use_waist matches the reference lift_fingerTorque.py's own documented usage
  ("Run with arguments: --use_waist --enable_cameras") and is NOT optional in
  practice: without it, ik_two_hands_dls (kinematics.py) zeroes the waist columns
  of the IK Jacobian, solving arms-only. That shrinks the reachable workspace
  (especially vertical reach, since waist pitch/roll is what extends how high/low
  the hand can go) and shows up as the hand falling short of its Z target in
  lift/reach segments -- a workspace-reachability limit, not a controller bug.

-------------------------------------------------------------------------
PH-THEORY MAPPING (van der Schaft, DISC lecture notes "Port-Hamiltonian
Systems Theory: An Introductory Overview" -- Ch.1 lumped PH systems, Ch.2
control of PH systems / IDA-PBC)
-------------------------------------------------------------------------
The HAND12 law implemented by ph.ph_impedance.PHContactForceController.compute()
is DISC Ch.2's IDA-PBC applied per control-mode segment:

  position mode (S1,S2,S6,S7,S8,S9,S10):
      u = g_c(q) - Kp*e - Kd*qdot_c - Ki*z,   e = q_c - q_des
    shapes the potential V_d(q_c) = 1/2 Kp e^2 (+ an integral term), moving
    the energy minimum of H_d = H + V_d from wherever gravity puts it to q_des.

  force mode (S3,S4,S5, IDA-PBC force regulation):
      u = g_c(q) - grad_Vd(q_c) - Kd*qdot_c,  grad_Vd = Kf*(tau_spring - tau_d)
    shapes a virtual-spring potential whose minimum is a desired CONTACT
    TORQUE rather than a joint angle -- Ch.2's control-by-interconnection
    with damping injection (-Kd*qdot_c), needed because stabilizing a
    non-zero-effort equilibrium runs into the "dissipation obstacle": pure
    energy/Casimir shaping alone cannot do it once R != 0 (or, as here, once
    damping is being actively injected through the port).

Both modes share the same damping-injection term -Kd*qdot_c, i.e. the same
Ch.1 R (dissipation) being added at the actuated port to make H_d a genuine
Lyapunov function (dH_d/dt = -qdot_c^T Kd qdot_c <= 0). This is exactly what
`scripts/g1_control/analyze_controller.py` checks numerically against the
logged data post-hoc -- see its own PH-THEORY MAPPING docstring for the
per-figure breakdown, and [[finger-torque-bangbang-saturation]] for why that
check currently fails (torque-rail saturation means this law is often NOT
the input actually being delivered).
"""

import argparse
import time
import math
import os
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import numpy as np

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="G1 two-arms + waist + fingers: PH-only finger control, arm/waist PID."
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_fps", type=float, default=120.0)
parser.add_argument("--print_every", type=int, default=60)

# Video recording
parser.add_argument("--video", action="store_true", default=False,
                    help="Record a viewport video of the simulation.")
parser.add_argument("--video_length", type=int, default=0,
                    help="Max number of steps to record (0 = record until end).")
parser.add_argument("--video_interval", type=int, default=1,
                    help="Capture one frame every N simulation steps.")
parser.add_argument("--video_folder", type=str, default="videos",
                    help="Output directory for the recorded video.")
parser.add_argument("--video_fps", type=float, default=30.0,
                    help="Frames per second for the output video.")
parser.add_argument("--video_width", type=int, default=1280)
parser.add_argument("--video_height", type=int, default=720)

NUM_SEGS = 10
for k in range(1, NUM_SEGS + 1):
    parser.add_argument(f"--t_seg_{k}", type=float, default=1.5)

# PhysX solver params
# min_velocity_iteration_count was 1 (the minimum) -- under-resolves joint
# velocity for the near-zero-inertia finger links, producing a per-step
# +/- oscillation in qdot that gets amplified by Kd into a full-scale torque
# chatter (position stays smooth; only the velocity readback -- and hence
# the commanded torque -- oscillates). Bumped default to 8 to test whether
# more velocity-solver iterations resolve it at the source.
parser.add_argument("--min_vel_iters", type=int, default=8,
                    help="PhysX min_velocity_iteration_count. Higher = more solver "
                         "effort resolving joint velocities per step (was hardcoded "
                         "to 1); raise this if finger torque chatters every step.")

# IK params
parser.add_argument("--use_waist", action="store_true")
parser.add_argument("--ik_max_iters", type=int, default=2000)
parser.add_argument("--ik_lambda", type=float, default=0.2)
parser.add_argument("--ik_alpha", type=float, default=0.5)
parser.add_argument("--ik_tol_p", type=float, default=1e-4)
parser.add_argument("--ik_tol_w", type=float, default=1e-4)
parser.add_argument("--ik_dq_max_deg", type=float, default=1.0)

# PID gains (waist + arms only)
# Waist+arm position-PID gains. Matched to the reference lift_fingerTorque.py
# (kp=4, kd=1, no arm integral). kp was 2.0 here, which is HALF the reference's arm
# stiffness -- too soft to hold the wrist against the grasp contact reaction, so the
# hand visibly drifts UP when contact begins at S3. ki was 1.0; the reference uses no
# arm integral because integral action during the sustained S3-S5 contact winds up and
# adds drift/oscillation (gravity comp already handles static hold).
parser.add_argument("--kp", type=float, default=4.0)
parser.add_argument("--kd", type=float, default=1.0)
parser.add_argument("--ki", type=float, default=0.0)
parser.add_argument("--i_clamp", type=float, default=0.35)

# PH finger controller
parser.add_argument("--ph_urdf_path", type=str,
                    default="unitree_g1_inspire_assets/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")
parser.add_argument("--ph_kp", type=float, default=4)
parser.add_argument("--ph_kd", type=float, default=1)
parser.add_argument("--ph_ki", type=float, default=0.1)
parser.add_argument("--ph_i_clamp", type=float, default=0.35)
parser.add_argument("--ph_kp_thumb_yaw", type=float, default=1.0,
                    help="Position-mode Kp override for L/R thumb-yaw joints only (HAND12 idx "
                         "0,6). The URDF caps these joints at effort=1 Nm over a ~1.4 rad range "
                         "(-0.1..1.3), but the segment-transition IK jump in q_des can be a large "
                         "fraction of that range in one step. At the shared --ph_kp=4 the ideal "
                         "(pre-clip) torque Kp*e can hit ~5-6 Nm -- 5-6x the actuator's real "
                         "torque -- before --finger_tau_clamp saturates it away. Thumb yaw only "
                         "holds orientation (no grip force), so it doesn't need grip-joint "
                         "stiffness; lowering just this Kp keeps the ideal law inside a sane range.")

# Force-impedance gains -- matched to skill spec gain schedule table
parser.add_argument("--Kt_free",   type=float, default=5.0,
                    help="Task-space stiffness: free motion S1,S2,S9,S10 (N/m)")
parser.add_argument("--Bt_free",   type=float, default=0.5,
                    help="Task-space damping: free motion S1,S2,S9,S10 (N.s/m)")
parser.add_argument("--Kt_close",  type=float, default=150.0,
                    help="Task-space stiffness: grasp closure S3 (N/m)")
parser.add_argument("--Bt_close",  type=float, default=8.0,
                    help="Task-space damping: grasp closure S3 (N.s/m)")
parser.add_argument("--Kt_grasp",  type=float, default=250.0,
                    help="Task-space stiffness: lift in contact S4,S5 (N/m)")
parser.add_argument("--Bt_grasp",  type=float, default=12.0,
                    help="Task-space damping: lift in contact S4,S5 (N.s/m)")
parser.add_argument("--contact_threshold", type=float, default=0.3,
                    help="Torque residual norm (Nm) above which contact is active")

# True force-impedance (IDA-PBC, DISC Ch.2 Sec.4.2)
# In contact phases (S3-S5), the finger Hamiltonian is shaped so its minimum
# is at tau_spring(q*) = tau_des, NOT at a joint angle.  Contact forces enter
# the PH interaction port; the robot naturally regulates to the desired grip force.
parser.add_argument("--Kf", type=float, default=1.0,
                    help="Force-loop gain Kf for IDA-PBC force regulation (Nm/Nm)")
parser.add_argument("--grasp_torque", type=float, default=5.0,
                    help="Desired contact torque for thumb-pitch joints in S3-S5 (Nm). "
                         "Finger joints receive 01.25x this value.")
parser.add_argument("--K_spr", type=float, default=2.5,
                    help="Virtual joint-spring stiffness (Nm/rad) for force-mode IDA-PBC. "
                         "CRITICAL: the force-mode equilibrium is q* = q0 + tau_d/K_spr, "
                         "NOT 'apply tau_d until contact' -- the finger stops moving once it "
                         "reaches this angle, contact or not. With the old hardcoded default "
                         "(50.0, never exposed here), grasp_torque=15 only closed the 4 main "
                         "fingers to 3.75/50=0.075 rad (out of 1.7 rad range) -- they looked "
                         "'stuck' near the object because they'd already hit their equilibrium, "
                         "not because of contact. Default 2.5 -> at grasp_torque=15 the 4 main "
                         "fingers equilibrate near 3.75/2.5=1.5 rad (~88%% closed); thumb-pitch "
                         "(15/2.5=6 rad target, but only a 0.6 rad range) saturates against its "
                         "own joint limit and holds there at ~max torque, i.e. a firm pinch. "
                         "Retune via q* = tau_d/K_spr for your own grasp_torque.")
parser.add_argument("--K_spr_thumb_pitch", type=float, default=14.0,
                    help="K_spr override for L/R thumb-pitch joints only (HAND12 idx 1,7). "
                         "Thumb-pitch's real range is only ~0.6 rad (-0.1..0.6, effort=1 Nm), "
                         "much smaller than the 4 main fingers' ~1.7 rad, so a single shared "
                         "K_spr can't fit both: at the default K_spr=2.5 and grasp_torque=7, "
                         "q*=tau_d/K_spr=2.8 rad is 4x the joint's real travel, so thumb-pitch "
                         "just rails against its mechanical limit and the ideal (pre-clip) law "
                         "sits pinned near Kf*grasp_torque (~7 Nm, 7x the actuator's 1 Nm) for "
                         "the whole grasp instead of relaxing toward 0. Raising K_spr for this "
                         "joint alone moves q* back inside its real range (14.0 -> 7/14=0.5 rad) "
                         "so the virtual spring actually reaches equilibrium and the ideal torque "
                         "decays instead of staying pinned at max. Retune via "
                         "K_spr_thumb_pitch = grasp_torque / (desired equilibrium rad, <~0.55).")

# Torque limiting
parser.add_argument("--tau_clamp", type=float, default=0.0)
parser.add_argument("--effort_limit_scale", type=float, default=1.0)
parser.add_argument("--arm_tau_clamp", type=float, default=0.0,
                    help="Extra torque clamp applied only to waist+arm (17 DOF) joints [Nm]. "
                         "0 = no extra clamp beyond --tau_clamp / URDF effort limits.")
parser.add_argument("--finger_tau_clamp", type=float, default=1.0,
                    help="Torque clamp applied only to HAND12 (12 DOF) finger joints [Nm]. "
                         "0 = no extra clamp.")

# Live step-response test (bypasses the camera/grasp task entirely)
parser.add_argument("--step_response", action="store_true",
                    help="Run a closed-loop step-response test in the live Isaac Sim (real "
                         "dynamics/contacts/armature, not the offline ODE approximation in "
                         "analyze_step_response.py) on --step_response_joints, save a diagnostic "
                         "figure + report, then exit -- skips the camera/IK/grasp task entirely. "
                         "Toggle on with just this flag; tune via the other --step_response_* args.")
parser.add_argument("--step_response_control_mode", type=str, default="position",
                    choices=["position", "force"],
                    help="Which PHContactForceController mode to step: 'position' steps "
                         "q_des_hand12 by --step_response_amplitude rad (uses --ph_kp/--ph_kd, "
                         "or --ph_kp_thumb_yaw for thumb-yaw joints); 'force' steps tau_d by "
                         "--step_response_amplitude Nm via set_force_mode (uses --Kf/--K_spr, "
                         "or --K_spr_thumb_pitch for thumb-pitch joints) -- the actual S3-S5 law.")
parser.add_argument("--step_response_joints", type=str, nargs="+",
                    default=["R_index_proximal_joint", "R_thumb_proximal_pitch_joint"],
                    help="HAND12 joint names to step (must be in HAND12_NAMES). Every other "
                         "HAND12 joint is held at its value when the test starts.")
parser.add_argument("--step_response_amplitude", type=float, default=0.7,
                    help="Step size above the resting level: rad (position mode) or Nm (force "
                         "mode).")
parser.add_argument("--step_response_pre_hold", type=float, default=0.3,
                    help="Seconds to hold at the resting level before stepping.")
parser.add_argument("--step_response_hold", type=float, default=2.9,
                    help="Seconds to hold at the stepped level. Must comfortably exceed the "
                         "closed loop's real settling time or the settle-time metric gets "
                         "right-censored by the window edge (see analyze_step_response.py's "
                         "same gotcha).")
parser.add_argument("--step_response_post_hold", type=float, default=1.0,
                    help="Seconds to hold back at the resting level after the step, to also "
                         "see the release transient.")
parser.add_argument("--step_response_out", type=str, default=None,
                    help="Output figure path. Default: scripts/g1_control/diag_step_response_live.png")

# CSV logging
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--log_every", type=int, default=1)
parser.add_argument("--log_prefix", type=str, default="robot_ph")

# Gain optimizer
parser.add_argument("--optimize_gains", action="store_true",
                    help="Run derivative-free Kp/Kd optimization instead of live sim.")
parser.add_argument("--opt_kmax",      type=float, default=300.0)
parser.add_argument("--opt_alpha1",    type=float, default=1.0)
parser.add_argument("--opt_alpha2",    type=float, default=1e-3)
parser.add_argument("--opt_alpha3",    type=float, default=0.0)
parser.add_argument("--opt_max_iters", type=int,   default=60)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Video recording requires cameras to be enabled
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Suppress the carb performance warning: gpu.foundation.plugin acquires
# IMemoryBudgetManagerFactory on every frame instead of caching it.
# This is internal to NVIDIA's GPU plugin -- users cannot fix the acquisition
# site, but we can silence the warning channel.
try:
    import omni.log as _omni_log
    _log = _omni_log.get_log()
    _ch = _log.find_channel("gpu.foundation.plugin")
    if _ch != _omni_log.CHANNEL_ID_INVALID:
        _log.set_channel_level(_ch, _omni_log.Level.ERROR,
                               _omni_log.SettingBehavior.OVERRIDE)
except Exception:
    pass


import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from scripts.g1_control.robot.unitree import UNITREE_G1_29DOF_CFG_HANDS_GRAVITY as ROBOT_CFG_GRAVITY
import isaaclab.utils.math as math_utils
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg

import matplotlib
matplotlib.use("Agg")  # off-screen backend -- TkAgg conflicts with Isaac Sim's carb event loop
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scripts.g1_control.ph.kinematics import ik_two_hands_dls
from scripts.g1_control.ph.ik_fingers_analytical import ik_fingers_analytical, ik_fingers_from_curl
from scripts.g1_control.ph.fk_fingers import fk_fingers as _fk_fingers
from scripts.g1_control.ph.dh_table import compute_all_dh_chains as _compute_dh_chains
from scripts.g1_control.plot_style import (
    DPI as _PLOT_DPI,
    ACTUAL_STYLE as _ACTUAL_STYLE,
    REF_STYLE as _REF_STYLE,
    ERROR_STYLE as _ERROR_STYLE,
    DELIVERED_STYLE as _DELIVERED_STYLE,
    IDEAL_STYLE as _IDEAL_STYLE,
    IMPEDANCE_STYLE as _IMPEDANCE_STYLE,
    FORCE_COLOR as _FORCE_COLOR,
    POSITION_COLOR as _POSITION_COLOR,
    FORCE_MODE_SEGMENTS as _FORCE_MODE_SEGMENTS,
    shade_mode as _shade_mode,
    mode_legend_patches as _mode_legend_patches,
    torque_rails as _torque_rails_base,
    saturation_mask as _saturation_mask,
    saturation_duty as _saturation_duty_base,
)

# Reuse the (pure, side-effect-free) metric formulas from the offline step-response
# diagnostic so the live --step_response test reports numbers directly comparable to
# `analyze_step_response.py` -- NOT a modification of that script, just importing its math.
from scripts.g1_control.analyze_step_response import (
    step_metrics_empirical as _step_metrics_empirical,
    analytic_metrics as _analytic_metrics,
    effective_inertia as _effective_inertia,
)


# -----------------------------------------------------------------------------
# Scene constants
# -----------------------------------------------------------------------------
TABLE_Z = 0.80
BOX_LEN_X = 0.06
BOX_LEN_Y = 0.03
BOX_LEN_Z = 0.10
BOX_HALF_Z = BOX_LEN_Z * 0.5
CUBE_MASS = 0.05

# NOTE: the converted USD (see scripts/g1_control/robot/unitree.py) has a FLAT prim
# hierarchy -- all links are direct children of the articulation root, so d435_link
# is a sibling of torso_link, not nested under it.
RS_ROOT = "{ENV_REGEX_NS}/Robot/d435_link/realsense_d455/RSD455"

# Sim indices for the 6 proximal (actuated) joints in each hand.
# Matches HAND12_SIM_INDICES in robotics/robot_indices.py.
HAND12_L_SIM = [29, 30, 33, 35, 37, 39]   # L: thumb_yaw, thumb_pitch, idx, mid, rng, pnk
HAND12_R_SIM = [41, 42, 45, 47, 49, 51]   # R: same order

# Cylinder grasp radius: matches the visual cylinder in SceneCfg
CYL_RADIUS = BOX_LEN_Y * 0.84              # â‰ˆ 0.025 m


# -----------------------------------------------------------------------------
# Camera helpers (unchanged from lift_fingerTorque.py)
# -----------------------------------------------------------------------------
def _find_semantic_id_for_class(sem_info, class_name):
    if sem_info is None:
        return None
    id_to_labels = sem_info.get("idToLabels") or sem_info.get("id_to_labels")
    if isinstance(id_to_labels, dict):
        for k, v in id_to_labels.items():
            if isinstance(v, dict) and v.get("class") == class_name:
                try: return int(k)
                except Exception: pass
            if isinstance(v, str) and v == class_name:
                try: return int(k)
                except Exception: pass
    labels = sem_info.get("labels")
    if isinstance(labels, list):
        for item in labels:
            if isinstance(item, dict) and item.get("class") == class_name:
                try: return int(item.get("id"))
                except Exception: pass
    return None


def cube_centroid_in_camera_frame(cam, class_name="cube", env_id=0, max_samples=10000):
    out = cam.data.output
    if ("semantic_segmentation" not in out) or (("depth" not in out) and ("distance_to_image_plane" not in out)):
        return None
    depth_key = "depth" if "depth" in out else "distance_to_image_plane"
    sem = out["semantic_segmentation"]
    dep = out[depth_key]
    sem_e = sem[env_id, ..., 0].to(torch.int32)
    dep_e = dep[env_id, ..., 0]
    sem_info = None
    if getattr(cam.data, "info", None) is not None:
        try:
            sem_info = cam.data.info[env_id].get("semantic_segmentation", None)
        except Exception:
            sem_info = None
    sem_id = _find_semantic_id_for_class(sem_info, class_name)
    if sem_id is None:
        return None
    mask = sem_e == sem_id
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    if ys.numel() > max_samples:
        sel = torch.randperm(ys.numel(), device=ys.device)[:max_samples]
        ys, xs = ys[sel], xs[sel]
    z = dep_e[ys, xs]
    valid = torch.isfinite(z) & (z > 0.0)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if ys.numel() == 0:
        return None
    K = cam.data.intrinsic_matrices[env_id]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = xs.to(z.dtype); v = ys.to(z.dtype)
    X = (u - cx) / fx * z; Y = (v - cy) / fy * z; Z = z
    return torch.stack([X, Y, Z], dim=-1).mean(dim=0)


def cube_cam_to_pelvis_from_torso(robot, p_cam_optical, env_id=0, torso_name="torso_link",
                                   d435_xyz=(0.0576235, 0.01753, 0.42987),
                                   d435_rpy=(0.0, 0.8307767239493009, 0.0),
                                   cam_offset_x=0.01):
    device = p_cam_optical.device
    dtype = p_cam_optical.dtype
    body_names = list(getattr(robot.data, "body_names", [])) or list(getattr(robot.data, "link_names", []))
    if torso_name not in body_names:
        raise RuntimeError(f"Body/link '{torso_name}' not found.")
    torso_idx = body_names.index(torso_name)
    torso_pos_w = robot.data.body_pos_w[env_id, torso_idx, :]
    torso_quat_w = robot.data.body_quat_w[env_id, torso_idx, :]
    pelvis_pos_w = robot.data.root_link_pos_w[env_id]
    pelvis_quat_w = robot.data.root_link_quat_w[env_id]
    t_t_d = torch.tensor(d435_xyz, device=device, dtype=dtype)
    rpy = torch.tensor(d435_rpy, device=device, dtype=dtype)
    q_t_d = math_utils.quat_from_euler_xyz(rpy[0], rpy[1], rpy[2])
    t_d_c = torch.tensor([cam_offset_x, 0.0, 0.0], device=device, dtype=dtype)
    p_cam_in_d435 = torch.stack([p_cam_optical[2], -p_cam_optical[0], -p_cam_optical[1]], dim=0)
    p_d = t_d_c + p_cam_in_d435
    p_t = t_t_d + math_utils.quat_apply(q_t_d.unsqueeze(0), p_d.unsqueeze(0))[0]
    p_w = torso_pos_w + math_utils.quat_apply(torso_quat_w.unsqueeze(0), p_t.unsqueeze(0))[0]
    dp_w = (p_w - pelvis_pos_w).unsqueeze(0)
    return math_utils.quat_apply_inverse(pelvis_quat_w.unsqueeze(0), dp_w)[0]


def get_effort_abs_limits(robot, device, dtype, num_envs, num_joints):
    lim = getattr(robot.data, "joint_effort_limits", None)
    if lim is None:
        return None
    lim_t = torch.as_tensor(lim, device=device, dtype=dtype)
    if lim_t.ndim == 2 and lim_t.shape[-1] == 2:
        absmax = torch.maximum(lim_t[:, 0].abs(), lim_t[:, 1].abs())
        return absmax.unsqueeze(0).expand(num_envs, -1)
    if lim_t.ndim == 3 and lim_t.shape[-1] == 2:
        return torch.maximum(lim_t[..., 0].abs(), lim_t[..., 1].abs())
    if lim_t.ndim == 1 and lim_t.shape[0] == num_joints:
        return lim_t.abs().unsqueeze(0).expand(num_envs, -1)
    if lim_t.ndim == 2 and lim_t.shape == (num_envs, num_joints):
        return lim_t.abs()
    return None


# -----------------------------------------------------------------------------
# Scene config
# -----------------------------------------------------------------------------
@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100, 100)))
    light = AssetBaseCfg(prim_path="/World/DomeLight",
                         spawn=sim_utils.DomeLightCfg(color=(0.95, 0.95, 0.95), intensity=3000.0))
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=[2, 0.8, 0.05],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0025, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, TABLE_Z - 0.025), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CylinderCfg(
            radius=BOX_LEN_Y * 0.84,  # make the cylinder wide enough to be a box in top-down view
            height=BOX_LEN_Z+ 0.02,  # add more height so it is easier to grasp
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            physics_material=RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=2.5, restitution=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
            semantic_tags=[("class", "cube")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.45, TABLE_Z + BOX_HALF_Z + 0.07),
                                                   rot=(1.0, 0.0, 0.0, 0.0)),
    )
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            copy_from_source=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.45, 0.85), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    robot = ROBOT_CFG_GRAVITY.replace(prim_path="{ENV_REGEX_NS}/Robot")
    realsense_d455 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link/realsense_d455",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Sensors/Intel/RealSense/rsd455.usd",
            # No rigid_props: the RSD455 USD has no RigidBodyAPI on its mesh;
            # passing any RigidBodyPropertiesCfg causes a physics-registration
            # conflict that silently breaks the kinematic attachment to torso_link.
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.01, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    rs_depth = CameraCfg(
        prim_path=f"{RS_ROOT}/Camera_Pseudo_Depth",
        spawn=None, update_period=0.0, width=320, height=240,
        data_types=["depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
    )


# -----------------------------------------------------------------------------
# Joint name mapping
# -----------------------------------------------------------------------------
IK17_JOINT_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
IK29_JOINT_NAMES = IK17_JOINT_NAMES + [
    "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint",
    "L_index_proximal_joint", "L_middle_proximal_joint",
    "L_ring_proximal_joint", "L_pinky_proximal_joint",
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint", "R_middle_proximal_joint",
    "R_ring_proximal_joint", "R_pinky_proximal_joint",
]
HAND12_NAMES = IK29_JOINT_NAMES[17:]


# -----------------------------------------------------------------------------
# Segment spec
# -----------------------------------------------------------------------------
@dataclass
class SegmentSpec:
    name: str
    T: float
    hands_pL: Optional[np.ndarray] = None
    hands_pR: Optional[np.ndarray] = None
    # Finger control â€” mutually exclusive:
    #   use_grasp_ik=True : compute FK-based cylinder contact targets analytically
    #   finger_curl_*     : direct curl fraction [0=open, 1=closed] via ik_fingers_from_curl
    #   both None/False   : hold previous finger angles unchanged
    use_grasp_ik: bool = False
    finger_curl_L: Optional[float] = 0.0
    finger_curl_R: Optional[float] = 0.0
    joint_pos_overrides: Optional[Dict[str, float]] = None  # non-finger overrides only
    RL_target: Optional[np.ndarray] = None
    RR_target: Optional[np.ndarray] = None


# -----------------------------------------------------------------------------
# Analytical finger IK helpers
# -----------------------------------------------------------------------------

def _build_theta29(q17: np.ndarray, q_sim_cpu: np.ndarray) -> np.ndarray:
    """
    Build the 29-DOF vector [waist(3), arm(14), HAND12_proximal(12)] expected
    by fk_fingers / ik_fingers_analytical from current arm IK solution + sim state.
    """
    th = np.zeros(29, dtype=float)
    th[:17] = q17
    th[17:23] = q_sim_cpu[HAND12_L_SIM]
    th[23:29] = q_sim_cpu[HAND12_R_SIM]
    return th


def _compute_grasp_targets(q17: np.ndarray, q_sim_cpu: np.ndarray,
                            cx: float, cy: float, cz: float) -> dict:
    """
    FK-based cylinder contact targets in PELVIS frame.

    1. Runs FK at the current arm pose + open-ish finger angles.
    2. Projects each fingertip radially onto the cylinder surface
       (vertical axis at (cx, cy), radius CYL_RADIUS).
    3. Returns a targets dict suitable for ik_fingers_analytical().

    Fingers that are already on the cylinder axis (dist < 1 mm) are skipped
    to avoid degenerate projections.
    """
    th29 = _build_theta29(q17, q_sim_cpu)
    fk   = _fk_fingers(th29)

    targets: dict = {'L': {}, 'R': {}}
    for side, fk_tips in (('L', fk.T_L), ('R', fk.T_R)):
        for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            p   = fk_tips[finger][:3, 3]
            dx  = float(p[0]) - cx
            dy  = float(p[1]) - cy
            d   = float(np.hypot(dx, dy))
            if d < 1e-3:
                continue
            targets[side][finger] = np.array([
                cx + dx * CYL_RADIUS / d,
                cy + dy * CYL_RADIUS / d,
                float(p[2]),
            ])
    return targets


def _live_object_pos_pelvis(scene, robot, env_id=0, obj_key="cube"):
    """Live (cx, cy, cz) of the grasped rigid object, in pelvis/tree frame --
    the same frame `_ftip_jac.tip_positions()` reports fingertips in. Unlike
    `cube_pos_pelvis_cam` (a one-shot camera estimate taken before the grasp),
    this reads the object's true simulated pose every call, so it tracks the
    object as the hand lifts/transports it."""
    obj_pos_w = scene[obj_key].data.root_pos_w[env_id]
    pel_pos_w = robot.data.root_link_pos_w[env_id]
    pel_quat_w = robot.data.root_link_quat_w[env_id]
    dp = (obj_pos_w - pel_pos_w).unsqueeze(0)
    return math_utils.quat_apply_inverse(pel_quat_w.unsqueeze(0), dp)[0].detach().cpu().numpy()


def _object_surface_ref(tip_xyz: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Per-fingertip reference = radial projection of each fingertip onto the
    live cylinder surface (vertical axis at (cx, cy), radius CYL_RADIUS),
    keeping each tip's own Z -- i.e. 'where this finger should be touching
    the object right now'. Mirrors `_compute_grasp_targets`'s projection but
    driven by the object's live pose instead of a pre-grasp camera snapshot,
    and vectorized over all 12 HAND12 tips at once.

    Fingers already on the cylinder axis (dist < 1 mm) fall back to their own
    (untouched) position to avoid a degenerate divide-by-zero projection.
    """
    tip_xyz = np.asarray(tip_xyz, dtype=float)
    ref = tip_xyz.copy()
    dx = tip_xyz[:, 0] - cx
    dy = tip_xyz[:, 1] - cy
    d = np.hypot(dx, dy)
    ok = d > 1e-3
    ref[ok, 0] = cx + dx[ok] * CYL_RADIUS / d[ok]
    ref[ok, 1] = cy + dy[ok] * CYL_RADIUS / d[ok]
    return ref


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(
        device=args.device,
        gravity=(0.0, 0.0, -9.8),
        physx=PhysxCfg(enable_external_forces_every_iteration=True,
                       min_velocity_iteration_count=int(args.min_vel_iters)),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([0, 2.2, 1.5], [0.2, 0.4, 0.9])

    # ------------------------------------------------------------------
    # Video recorder setup (omni.replicator viewport capture)
    # Writer is opened when S1_pregrasp begins; closed when S8_idle ends.
    # ------------------------------------------------------------------
    _video_writer = None
    _video_frames_captured = 0
    _video_render_product = None
    _video_annotator = None
    _video_path = None
    _video_recording = False   # True while recording S1_pregrasp -> S8_idle
    if args.video:
        import omni.replicator.core as rep
        import cv2

        os.makedirs(args.video_folder, exist_ok=True)
        _video_render_product = rep.create.render_product(
            "/OmniverseKit_Persp", (args.video_width, args.video_height)
        )
        _video_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        _video_annotator.attach([_video_render_product])
        print(f"[VIDEO] Render product ready -- will record S1_pregrasp..S8_idle -> {args.video_folder}/")

    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=3.0))
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    rs_depth_cam = scene["rs_depth"]

    dt_boot = sim.get_physics_dt()
    for _ in range(30):
        scene.write_data_to_sim(); sim.step(); scene.update(dt_boot)

    # Mutable container so the per-step camera loop can update it in-place
    # without needing 'nonlocal' inside the closures.
    cube_pos_pelvis_cam = [None]   # cube_pos_pelvis_cam[0] = numpy array (3,) in pelvis frame
    for _ in range(300):
        p_cam = cube_centroid_in_camera_frame(rs_depth_cam, class_name="cube", env_id=0)
        if p_cam is not None:
            p_pelvis = cube_cam_to_pelvis_from_torso(robot, p_cam, 0, "torso_link")
            cube_pos_pelvis_cam[0] = p_pelvis.detach().cpu().numpy()
            break
        scene.write_data_to_sim(); sim.step(); scene.update(dt_boot)

    if cube_pos_pelvis_cam[0] is None:
        # Fallback: convert the cylinder's known world-frame spawn position to
        # pelvis frame using the robot's current root pose.
        # World spawn: (0.05, 0.45, TABLE_Z + BOX_HALF_Z + 0.07).
        # The robot spawns at world (0, 0, 0.793) rotated 90Â° about Z, so the
        # pelvis-frame X (forward) â‰ˆ world Y and pelvis Y (left) â‰ˆ -world X.
        _cyl_world = torch.tensor(
            [0.05, 0.45, TABLE_Z + BOX_HALF_Z + 0.07],
            device=robot.data.root_link_pos_w.device,
            dtype=robot.data.root_link_pos_w.dtype,
        )
        _pel_pos = robot.data.root_link_pos_w[0]
        _pel_quat = robot.data.root_link_quat_w[0]
        from isaaclab.utils import math as _math
        _dp = (_cyl_world - _pel_pos).unsqueeze(0)
        _fallback = _math.quat_apply_inverse(_pel_quat.unsqueeze(0), _dp)[0].cpu().numpy()
        print(f"[WARNING] Cube not detected by camera (--enable_cameras not set?). "
              f"Using pelvis-frame fallback {_fallback.tolist()}.")
        cube_pos_pelvis_cam[0] = _fallback

    print(f"[INFO] cube center (pelvis frame) = [{cube_pos_pelvis_cam[0][0]:+.3f}, "
          f"{cube_pos_pelvis_cam[0][1]:+.3f}, {cube_pos_pelvis_cam[0][2]:+.3f}] m  "
          f"(expected â‰ˆ [+0.45, -0.05, +0.17] for default scene)")

    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    joint_names = list(robot.data.joint_names)

    def idx_of_joint(name):
        if name not in joint_names:
            raise RuntimeError(f"Joint '{name}' not found.")
        return joint_names.index(name)

    ik17_indices = [idx_of_joint(n) for n in IK17_JOINT_NAMES]
    ik29_indices = [idx_of_joint(n) for n in IK29_JOINT_NAMES]
    hand12_indices = [idx_of_joint(n) for n in HAND12_NAMES] #artifulation_order_indices for the 12 controlled finger joints

    # ----- PH force-impedance finger controller (true IDA-PBC, DISC Ch.2 Sec.4.2) -----
    from scripts.g1_control.ph import (
        parse_urdf, build_tree, PHSystem,
        FingertipJacobian, PHContactForceController,
    )

    # Fingertip (most distal) link names from the DFQ URDF, in HAND12 order.
    # Thumb shares a distal link for both yaw and pitch joints.
    # Non-thumb fingers: *_intermediate is the distal-most actuated link.
    TIP_LINK_NAMES = [
        "L_thumb_distal",       # L-thumb-proximal-yaw
        "L_thumb_distal",       # L-thumb-proximal-pitch
        "L_index_intermediate", # L-index-proximal
        "L_middle_intermediate",# L-middle-proximal
        "L_ring_intermediate",  # L-ring-proximal
        "L_pinky_intermediate", # L-pinky-proximal
        "R_thumb_distal",       # R-thumb-proximal-yaw
        "R_thumb_distal",       # R-thumb-proximal-pitch
        "R_index_intermediate", # R-index-proximal
        "R_middle_intermediate",# R-middle-proximal
        "R_ring_intermediate",  # R-ring-proximal
        "R_pinky_intermediate", # R-pinky-proximal
    ]

    _ph_urdf_path = args.ph_urdf_path
    if not os.path.isabs(_ph_urdf_path):
        _ph_urdf_path = os.path.join(REPO_ROOT, _ph_urdf_path)
    _ph_urdf = parse_urdf(_ph_urdf_path)
    _ph_tree = build_tree(_ph_urdf)
    _ph_sys  = PHSystem(_ph_tree, control_joint_names=HAND12_NAMES)


    _ftip_jac = FingertipJacobian(
        ph_sys=_ph_sys,
        tip_link_names=TIP_LINK_NAMES,
        hand12_tree_idx=np.array(_ph_sys.control_indices),
    )

    # HAND12 order: L_thumb_yaw(0), L_thumb_pitch(1), L_idx..L_pnk(2-5),
    #               R_thumb_yaw(6), R_thumb_pitch(7), R_idx..R_pnk(8-11).
    # Thumb-yaw and thumb-pitch each get their own gain (see --ph_kp_thumb_yaw /
    # --K_spr_thumb_pitch help text): a single shared Kp/K_spr for all 12 joints
    # drives the pre-clip "ideal" torque 5-7x past these joints' real 1 Nm/small-
    # range limits, since they have far less usable travel than the 4 main fingers.
    _THUMB_YAW_IDX = (0, 6)
    _THUMB_PITCH_IDX = (1, 7)
    _ph_kp_arr = np.full(12, float(args.ph_kp))
    _ph_kp_arr[list(_THUMB_YAW_IDX)] = float(args.ph_kp_thumb_yaw)
    _K_spr_arr = np.full(12, float(args.K_spr))
    _K_spr_arr[list(_THUMB_PITCH_IDX)] = float(args.K_spr_thumb_pitch)

    ph_ctrl = PHContactForceController(
        ph_sys=_ph_sys,
        ftip_jac=_ftip_jac,
        sim_joint_names=joint_names,
        Kp=_ph_kp_arr,
        Kd=np.full(12, float(args.ph_kd)),
        Ki=np.full(12, float(args.ph_ki)),
        i_clamp=float(args.ph_i_clamp),
        Kt_free=float(args.Kt_free),
        Kt_grasp=float(args.Kt_grasp),
        Bt_free=float(args.Bt_free),
        Bt_grasp=float(args.Bt_grasp),
        contact_threshold=float(args.contact_threshold),
        Kf=float(args.Kf),
        K_spr=_K_spr_arr,
    )
    ph_hand12_sim_idx = torch.tensor(ph_ctrl.hand12_sim_indices, device=robot.device, dtype=torch.long)
    print(f"[PH] Contact-force IDA-PBC controller active: NB={_ph_tree.nb}, "
          f"HAND12 sim indices {ph_ctrl.hand12_sim_indices}, "
          f"{len(ph_ctrl.missing_in_sim)} tree DOFs not in sim (default 0).")

    # Pre-compute DH kinematic tables for all 10 finger chains (used for FK validation).
    _dh_chains = _compute_dh_chains()
    print(f"[IK] DH chains ready: {len(_dh_chains)} finger chains "
          f"({', '.join(sorted(_dh_chains.keys()))})")

    # Desired contact torques for force-regulation phases (S3, S4, S5).
    # HAND12 order: L_thumb_yaw, L_thumb_pitch, L_idx, L_mid, L_rng, L_pnk,
    #               R_thumb_yaw, R_thumb_pitch, R_idx, R_mid, R_rng, R_pnk
    # Yaw joints are kept at 0 (orientation control only).
    # Thumb-pitch joints receive full grasp_torque; finger joints receive 0.25x.
    # Matches the tau_fingers proportions in forceImpedanceResults.py.
    _gt = float(args.grasp_torque)
    tau_des_grasp = np.array([
        0.0,       # L thumb yaw   -- orientation, no grip force
        _gt,       # L thumb pitch -- primary pinch contact
        _gt * 0.25, _gt * 0.25, _gt * 0.25, _gt * 0.25,  # L index/mid/rng/pnk
        0.0,       # R thumb yaw
        _gt,       # R thumb pitch
        _gt * 0.25, _gt * 0.25, _gt * 0.25, _gt * 0.25,  # R index/mid/rng/pnk
    ])

    q_init = robot.data.joint_pos.clone()
    num_envs, num_joints = q_init.shape

    def get_gravity_comp_torque():
        view = getattr(robot, "root_physx_view", None) or getattr(robot, "_root_physx_view", None)
        if view is None or not hasattr(view, "get_gravity_compensation_forces"):
            return torch.zeros((num_envs, num_joints), device=device, dtype=dtype)
        g = torch.as_tensor(view.get_gravity_compensation_forces(), device=device, dtype=dtype).view(num_envs, -1)
        if g.shape[1] == num_joints + 6:
            g = g[:, 6:]
        elif g.shape[1] != num_joints:
            g = g[:, -num_joints:] if g.shape[1] > num_joints else torch.zeros_like(q_init)
        return g

    # -------------------------------------------------------------------------
    # Gain optimizer episode runner
    # -------------------------------------------------------------------------
    def run_episode(Kp: np.ndarray, Kd: np.ndarray):
        """Reset scene, run one full segment pass, return (J1, J2, J3)."""
        ph_ctrl.Kp = np.clip(Kp, 0.0, None)
        ph_ctrl.Kd = np.clip(Kd, 0.0, None)

        # reset robot and cube to defaults
        scene["robot"].write_joint_state_to_sim(
            scene["robot"].data.default_joint_pos.clone(),
            torch.zeros_like(scene["robot"].data.default_joint_pos))
        scene["cube"].write_root_state_to_sim(
            scene["cube"].data.default_root_state.clone())
        scene.reset()
        _ep_dt = sim.get_physics_dt()
        for _ in range(30):
            scene.write_data_to_sim(); sim.step(); scene.update(_ep_dt)

        # re-detect cube from camera
        for _ in range(60):
            _pc = cube_centroid_in_camera_frame(rs_depth_cam, class_name="cube", env_id=0)
            if _pc is not None:
                cube_pos_pelvis_cam[0] = cube_cam_to_pelvis_from_torso(
                    robot, _pc, 0, "torso_link").detach().cpu().numpy()
                break
            scene.write_data_to_sim(); sim.step(); scene.update(_ep_dt)

        segs = make_segments()
        J1 = J2 = J3 = 0.0
        _q17_loc   = np.zeros(17)
        _q_h12_loc = np.zeros(12)
        _dq_max = float(args.ik_dq_max_deg) * math.pi / 180.0

        for seg in segs:
            _q_des = robot.data.joint_pos.clone()
            if seg.hands_pL is not None:
                _RL = seg.RL_target if seg.RL_target is not None else np.eye(3)
                _RR = seg.RR_target if seg.RR_target is not None else np.eye(3)
                _q17_loc, _ = ik_two_hands_dls(
                    pL_target=seg.hands_pL, pR_target=seg.hands_pR,
                    RL_target=_RL, RR_target=_RR,
                    use_waist=bool(args.use_waist), q0=_q17_loc,
                    max_iters=int(args.ik_max_iters), lamb=float(args.ik_lambda),
                    alpha=float(args.ik_alpha), tol_p=float(args.ik_tol_p),
                    tol_w=float(args.ik_tol_w), dq_max=_dq_max,
                )
                _q_des = apply_q17(_q_des, _q17_loc)

            # Analytical finger IK (mirrors main-loop logic)
            _q_sim_cpu = robot.data.joint_pos[0].detach().cpu().numpy()
            if seg.use_grasp_ik:
                _cx, _cy, _cz = cube_pos_pelvis_cam[0].tolist()
                _gtgts = _compute_grasp_targets(_q17_loc, _q_sim_cpu, _cx, _cy, _cz)
                _th29  = _build_theta29(_q17_loc, _q_sim_cpu)
                _res   = ik_fingers_analytical(_gtgts, _th29, validate=False)
                _q_h12_loc[:6] = _res.q_hand6_L
                _q_h12_loc[6:] = _res.q_hand6_R
            elif seg.finger_curl_L is not None:
                _th29  = _build_theta29(_q17_loc, _q_sim_cpu)
                _qcurl = ik_fingers_from_curl(
                    curl_L=seg.finger_curl_L,
                    curl_R=seg.finger_curl_R if seg.finger_curl_R is not None else seg.finger_curl_L,
                    theta29_0=_th29,
                )
                _q_h12_loc[:6] = _qcurl[17:23]
                _q_h12_loc[6:] = _qcurl[23:29]
            _q29a = np.zeros(29)
            _q29a[:17]   = _q17_loc
            _q29a[17:23] = _q_h12_loc[:6]
            _q29a[23:29] = _q_h12_loc[6:]
            _q_des = apply_q29(_q_des, _q29a)
            _q_des = apply_pos_overrides(_q_des, seg.joint_pos_overrides)

            ph_ctrl.q_des_hand12 = _q_des[0, ph_hand12_sim_idx].detach().cpu().numpy()
            ph_ctrl.reset_integral()

            n_steps = max(1, int(seg.T / _ep_dt))
            for _ in range(n_steps):
                q_np  = robot.data.joint_pos[0].detach().cpu().numpy()
                qd_np = robot.data.joint_vel[0].detach().cpu().numpy()
                q_c   = q_np[ph_ctrl.hand12_sim_indices]
                tau_h12 = ph_ctrl.compute(q_np, qd_np, dt=_ep_dt)

                J1 += float(np.sum((q_c - ph_ctrl.q_des_hand12) ** 2)) * _ep_dt
                J2 += float(np.sum(tau_h12 ** 2)) * _ep_dt

                _q_t = robot.data.joint_pos
                _qd_t = robot.data.joint_vel
                _tau = kp * (_q_des - _q_t) + kd * (-_qd_t) + get_gravity_comp_torque()
                for _env in range(num_envs):
                    _tau[_env, ph_hand12_sim_idx] = torch.as_tensor(
                        tau_h12, dtype=dtype, device=device)
                if effort_abs_lim is not None:
                    _tau = torch.clamp(_tau, -effort_abs_lim, effort_abs_lim)
                robot.set_joint_effort_target(_tau)
                scene.write_data_to_sim(); sim.step(); scene.update(_ep_dt)

        return J1, J2, J3

    # -------------------------------------------------------------------------
    # Live step-response test (toggle: --step_response)
    # -------------------------------------------------------------------------
    def run_step_response():
        """Step a subset of HAND12 joints through rest->stepped->rest in the real
        Isaac Sim closed loop (actual gravity/contacts/PhysX armature -- not the
        offline 1-DOF ODE approximation in analyze_step_response.py), then save a
        diagnostic figure + report using that script's own metric formulas so the
        numbers are directly comparable. Bypasses the camera/IK/grasp task
        entirely: waist+arms hold their starting pose via the existing --kp/--kd
        PID, and HAND12 joints not named in --step_response_joints hold their
        starting angle too."""
        step_names = list(args.step_response_joints)
        for n in step_names:
            if n not in HAND12_NAMES:
                raise ValueError(f"--step_response_joints: '{n}' not in HAND12_NAMES {HAND12_NAMES}")
        local_idx = {n: HAND12_NAMES.index(n) for n in step_names}

        mode = args.step_response_control_mode
        amp = float(args.step_response_amplitude)
        t_lo = float(args.step_response_pre_hold)
        t_hi = t_lo + float(args.step_response_hold)
        t_end = t_hi + float(args.step_response_post_hold)
        n_steps = int(round(t_end / dt))

        q_hold_full = robot.data.joint_pos.clone()
        q0_hand12 = q_hold_full[0, ph_hand12_sim_idx].detach().cpu().numpy().copy()

        # Prime _last_q_c so set_force_mode()'s q0_spr snapshot (and position mode's
        # error term) start from the real current pose, not the controller's
        # construction-time default (see ph_impedance.py set_force_mode docstring).
        ph_ctrl._last_q_c = q0_hand12.copy()

        if mode == "position":
            ph_ctrl.set_position_mode()
            ph_ctrl.q_des_hand12 = q0_hand12.copy()
        else:
            ph_ctrl.set_force_mode(np.zeros(12))  # captures q0_spr=current pose, tau_d=0
        ph_ctrl.reset_integral()

        hist = {n: dict(t=[], q=[], ref=[], tau_delivered=[], tau_ideal=[]) for n in step_names}

        print(f"\n[STEP] Live step-response test: mode={mode} joints={step_names} "
              f"amplitude={amp} pre_hold={t_lo}s hold={args.step_response_hold}s "
              f"post_hold={args.step_response_post_hold}s ({n_steps} steps)")

        for k in range(n_steps + 1):
            tk = k * dt
            level = 1.0 if (t_lo <= tk < t_hi) else 0.0

            if mode == "position":
                q_target = q0_hand12.copy()
                for n in step_names:
                    q_target[local_idx[n]] = q0_hand12[local_idx[n]] + level * amp
                ph_ctrl.q_des_hand12 = q_target
            else:
                tau_d = np.zeros(12)
                for n in step_names:
                    tau_d[local_idx[n]] = level * amp
                ph_ctrl._tau_d = tau_d  # step tau_d only; keep the q0_spr anchor fixed

            q_np = robot.data.joint_pos[0].detach().cpu().numpy()
            qd_np = robot.data.joint_vel[0].detach().cpu().numpy()
            tau_ideal_12 = ph_ctrl.compute(q_np, qd_np, dt=dt)
            tau_delivered_12 = tau_ideal_12.copy()
            if _finger_lim is not None:
                tau_delivered_12 = np.clip(tau_delivered_12, -_finger_lim, _finger_lim)

            q_t = robot.data.joint_pos
            qd_t = robot.data.joint_vel
            tau = kp * (q_hold_full - q_t) + kd * (-qd_t) + get_gravity_comp_torque()
            for env in range(num_envs):
                tau[env, ph_hand12_sim_idx] = torch.as_tensor(
                    tau_delivered_12, dtype=dtype, device=device)
            if effort_abs_lim is not None:
                tau = torch.clamp(tau, -effort_abs_lim, effort_abs_lim)
            robot.set_joint_effort_target(tau)
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

            for n in step_names:
                li = local_idx[n]
                if mode == "position":
                    ref = q0_hand12[li] + level * amp
                else:
                    ref = ph_ctrl._q0_spr[li] + (level * amp) / ph_ctrl.K_spr[li]
                hist[n]["t"].append(tk)
                hist[n]["q"].append(float(q_np[ph_hand12_sim_idx[li]]))
                hist[n]["ref"].append(float(ref))
                hist[n]["tau_delivered"].append(float(tau_delivered_12[li]))
                hist[n]["tau_ideal"].append(float(tau_ideal_12[li]))

            if k % 60 == 0:
                print(f"[STEP] t={tk:5.2f}s level={level:.0f}  " + "  ".join(
                    f"{n}: q={hist[n]['q'][-1]:+.3f} ref={hist[n]['ref'][-1]:+.3f} "
                    f"tau={hist[n]['tau_delivered'][-1]:+.2f}(ideal {hist[n]['tau_ideal'][-1]:+.2f})"
                    for n in step_names))

        # -- metrics (reusing analyze_step_response.py's formulas) + figure --
        out_path = args.step_response_out or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "diag_step_response_live.png")
        report_path = os.path.splitext(out_path)[0] + "_report.txt"

        fig, axes = plt.subplots(2 * len(step_names), 1, figsize=(9, 3.0 * 2 * len(step_names)),
                                  squeeze=False)
        report_lines = [
            "lift_fingerPH.py --step_response -- LIVE Isaac Sim diagnostic report",
            f"control_mode={mode}  amplitude={amp}  pre_hold={t_lo}s  hold={args.step_response_hold}s  "
            f"post_hold={args.step_response_post_hold}s  finger_tau_clamp={args.finger_tau_clamp}",
            "",
        ]
        for row, n in enumerate(step_names):
            li = local_idx[n]
            t_arr = np.asarray(hist[n]["t"]); q_arr = np.asarray(hist[n]["q"])
            ref_arr = np.asarray(hist[n]["ref"])
            taud_arr = np.asarray(hist[n]["tau_delivered"]); taui_arr = np.asarray(hist[n]["tau_ideal"])

            tree_idx = ph_ctrl.hand12_tree_indices[li]
            M_eff = _effective_inertia(_ph_sys, tree_idx, 0.01, np.zeros(_ph_sys.tree.nb))
            if mode == "position":
                K = float(ph_ctrl.Kp[li])
                ref_before, ref_after = float(q0_hand12[li]), float(q0_hand12[li] + amp)
            else:
                K = float(ph_ctrl.Kf[li] * ph_ctrl.K_spr[li])
                ref_before = float(ph_ctrl._q0_spr[li])
                ref_after = float(ph_ctrl._q0_spr[li] + amp / ph_ctrl.K_spr[li])
            Kd = float(ph_ctrl.Kd[li])
            ana = _analytic_metrics(K, Kd, M_eff)
            emp = _step_metrics_empirical(t_arr, q_arr, t_lo, ref_before, ref_after, t_step_end=t_hi)

            ax_q = axes[2 * row][0]
            ax_q.plot(t_arr, ref_arr, "k--", lw=1.4, label="reference")
            ax_q.plot(t_arr, q_arr, color="#d9534f", lw=1.8, label="measured q (live Isaac Sim)")
            ax_q.set_ylabel("q [rad]")
            ax_q.set_title(f"{n}  (mode={mode})", fontsize=10)
            ax_q.legend(fontsize=8, loc="lower right")
            ax_q.grid(alpha=0.3)
            txt = (
                f"K={K:.2f} Kd={Kd:.2f} M_eff={M_eff:.5f}  ζ_analytic={ana['zeta']:.2f}  "
                f"no-overshoot? {ana['no_overshoot']}\n"
                f"measured: overshoot={emp['overshoot_pct']:.1f}% (analytic {ana['overshoot_pct']:.1f}%)  "
                f"rise={emp['rise_time']*1e3:.0f}ms  settle2%={emp['settle_time']*1e3:.0f}ms "
                f"(analytic {ana['settle_time_2pct']*1e3:.0f}ms)"
            )
            ax_q.text(0.01, 0.02, txt, transform=ax_q.transAxes, fontsize=7.5, va="bottom", ha="left",
                       family="monospace", bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

            ax_tau = axes[2 * row + 1][0]
            ax_tau.plot(t_arr, taui_arr, color="tab:orange", lw=1.3, ls="--", label="PH law (pre-clip, ideal)")
            ax_tau.plot(t_arr, taud_arr, color="tab:blue", lw=1.4, label="delivered (post-clip)")
            if _finger_lim is not None:
                ax_tau.axhline(_finger_lim, color="0.4", lw=0.8, ls=":")
                ax_tau.axhline(-_finger_lim, color="0.4", lw=0.8, ls=":")
            ax_tau.set_ylabel("tau [Nm]")
            ax_tau.legend(fontsize=8, loc="lower right")
            ax_tau.grid(alpha=0.3)
            if row == len(step_names) - 1:
                ax_tau.set_xlabel("time [s]")

            report_lines += [
                f"== {n} (local idx {li}) ==",
                f"  M_eff={M_eff:.5f} kg*m^2  K={K:.3f}  Kd={Kd:.3f}",
                f"  analytic:  zeta={ana['zeta']:.3f}  wn={ana['wn']:.2f} rad/s  "
                f"overshoot={ana['overshoot_pct']:.1f}%  settle2%={ana['settle_time_2pct']*1e3:.0f}ms  "
                f"no-overshoot condition (Kd^2>=4*K*M)? {ana['no_overshoot']}",
                f"  measured:  overshoot={emp['overshoot_pct']:.1f}%  "
                f"rise(10-90%)={emp['rise_time']*1e3:.0f}ms  settle2%={emp['settle_time']*1e3:.0f}ms  "
                f"peak|tau_ideal|={np.max(np.abs(taui_arr)):.2f} Nm  peak|tau_delivered|={np.max(np.abs(taud_arr)):.2f} Nm",
                "",
            ]

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        report_text = "\n".join(report_lines)
        with open(report_path, "w") as f:
            f.write(report_text + "\n")
        print(report_text)
        print(f"[STEP] saved {out_path}")
        print(f"[STEP] saved {report_path}")

    # -------------------------------------------------------------------------
    # Segments
    # -------------------------------------------------------------------------
    seg_T = [float(getattr(args, f"t_seg_{k}")) for k in range(1, NUM_SEGS + 1)]

    def make_segments():
        """
        Bimanual pick-place segments adapted from Unitree dex1 redblock task geometry.

        Unitree scene: 6x6x6 cm cube, table at 0.84 m, symmetric +/-Y palm approach.
        Scaled here to BOX_LEN_X/Y/Z = 0.05/0.03/0.10 m.

        Coordinate frame: pelvis-centred.
          cx, cy, cz  -- cube centre
          +Y           -- left side of robot (left palm approaches from here)
          -Y           -- right side of robot (right palm approaches from here)
          +Z           -- up

        Palm geometry follows Unitree dex1:
          - Bimanual symmetric: left at cy+(hy+y_near), right at cy-(hy+y_near)
          - palmX offsets the wrist in the robot-forward (X) direction
          - RR_45 rotates the right palm 45deg around Z for angled cube approach
          - Identity rotation during vertical lift / transport (Unitree style)
        """
        cx, cy, cz = cube_pos_pelvis_cam[0].tolist()   # live camera estimate
        hy = BOX_LEN_Y * 0.5    # 0.015 m  (half of 3 cm Y-width)

        # ---- geometry constants (Unitree dex1-scaled to our cube) ----
        palmX      = 0.04   # wrist forward offset (all other waypoints)
        palmX_R_s2 = 0.1    # S2/S3 right hand: at cube X (no offset)
        y_wide  = 0.06   # S1 pregrasp: wide hover, 60 mm from cube face
        y_near  = 0.045  # S2/S3 approach: standoff from cube Y-face (tighter -- was 0.070)
        z_abv        = -0.10 # left palm Z: 8 cm below cube centre (thumb reaches under)
        z_grasp_bias = -0.10   # extra drop for S3 to compensate contact-force upward drift

        # z_below: depth of the rubber_hand IK end-effector below the cylinder centre.
        # From URDF geometry: the rubber_hand frame origin sits between index (+3.2 cm)
        # and pinky (-2.5 cm) MCP joints in the hand's local Z axis, so the MCP
        # midpoint is only ~3 mm above the rubber_hand origin -- essentially coincident.
        # Setting z_below = 0.05 places the palm centre ~5 cm below cz (lower third of
        # the 12 cm tall cylinder), giving a bottom-biased stable wrap grasp.
        # Decrease toward 0.0 to raise the hand to the cylinder equator.
        z_below = 0.05
        z_lift  = 0.20   # S4: raise 15 cm above cube centre
        y_box   = 0.30   # S5: transport -- move 30 cm in -Y toward drop box

        # ---- orientations ----
        RL_id = np.eye(3)
        RR_id = np.eye(3)
        # S2/S3: small valid rotation around Z (10Â°) to angle right palm toward cube.
        # Previously used a non-orthogonal matrix (column norm 1.1) which corrupted
        # the IK orientation error; replaced with proper SO(3).
        _c, _s = math.cos(math.radians(10)), math.sin(math.radians(10))
        RR_face = np.array([
            [ _c, -_s,  0.0],
            [ _s*1.1,  _c,  0.0],
            [0.0,  0.0,  1.0],
        ])

        # ---- waypoints ----
        # S1: wide pregrasp hover -- right hand already positioned below cube centre
        pL_pre  = np.array([cx - palmX,  cy + (hy + y_wide+0.05),  cz + z_abv])
        pR_pre  = np.array([cx - palmX+0.5,  cy - (hy + (y_wide-0.05)),  cz - z_below])

        # S2: close approach -- right palm below cube centre, facing cube
        pL_cls  = np.array([cx - palmX+0.01,  cy + (hy + y_near+0.3),  cz + z_abv])
        pR_cls  = np.array([cx -palmX_R_s2,  cy - (hy + y_near),  cz + z_abv])
        pR_clss = np.array([cx -palmX_R_s2,  cy - (hy+  y_near),  cz + z_abv ]) 
        # S3: grasp IK target drops further to pre-compensate contact-force upward drift
        pL_grs  = np.array([cx - palmX,  cy + (hy + y_near),  cz + z_abv + z_grasp_bias])
        pR_grs  = np.array([cx+0.05,          cy - (hy + y_near),  cz + z_abv + z_grasp_bias*1.5])

        # S4: vertical lift -- keep palm contact geometry, raise Z
        pL_lft  = np.array([cx - palmX,  cy + (hy + y_near),  cz + z_lift])
        pR_lft  = np.array([cx - palmX,  cy - (hy + y_near),  cz + z_lift])

        # S5-S7: transport in -Y toward drop box, maintain lift height
        pL_box  = np.array([cx - palmX,  cy + (hy + y_near+0.3),  cz + z_lift])
        pR_box  = np.array([cx - palmX,  cy - (hy + y_box),   cz + z_lift])

        # S8-S10: safe idle retreat -- back to wide hover at cube height
        pL_idle = np.array([cx,     cy + (hy + y_wide),  cz])
        pR_idle = np.array([cx - palmX,  cy - (hy + y_near),  cz + z_lift])
        partial_curl=np.array([1, 0, 0, 0, 0, 0]) #(for setting the thumbs yaw/pitch)
        # Finger control legend:
        #   use_grasp_ik=True          â†’ FK-project each fingertip onto cylinder surface,
        #                                solve analytically (ik_fingers_analytical)
        #   finger_curl_L/R=0.0        â†’ open hand (ik_fingers_from_curl, curl=0)
        #   finger_curl_L/R=None       â†’ hold current angles (no IK call)
        #
        # Gain schedule (applied in transition hook):
        #   S1, S2, S9, S10 : Kt_free / Bt_free   -- free motion, contact OFF
        #   S3-grasp-PH     : Kt_close / Bt_close  -- grasp closure, auto-detect
        #   S4-lift-PH,
        #   S5-moveair-PH   : Kt_grasp / Bt_grasp  -- lift in contact, forced ON
        #   S6-release,
        #   S7-drop, S8-idle: Kt_free / Bt_free    -- release, contact OFF
        return [
            #            name                  dur       pL       pR       grasp   curlL curlR    ovr   RL      RR
            SegmentSpec("S1-pregrasp",         seg_T[0], pL_pre,  pR_cls,  False,  partial_curl,  partial_curl,  None, RL_id,  RR_id),
            SegmentSpec("S2-close-approach",   seg_T[1], pL_cls,  pR_cls,  False,  partial_curl,  partial_curl,  None, RL_id,  RR_face),
            SegmentSpec("S3-grasp-PH",         seg_T[2], pL_cls,  pR_clss,  True,  None, None, None, RL_id,  RR_face),
            SegmentSpec("S4-lift-PH",          seg_T[3], pL_cls,  pR_lft,  False,  None, None, None, RL_id,  RL_id),
            SegmentSpec("S5-moveair-PH",       seg_T[4], pL_cls,  pR_box,  False,  None, None, None, RL_id,  RL_id),
            SegmentSpec("S6-release",          seg_T[5], pL_cls,  pR_box,  False,  0.0,  0.0,  None, RL_id,  RL_id),
            SegmentSpec("S7-drop",             seg_T[6], pL_cls,  pR_box,  False,  0.0,  0.0,  None, RL_id,  RL_id),
            SegmentSpec("S8-idle",             seg_T[7], pL_pre,  pL_pre,  False,  0.0,  0.0,  None, RL_id,  RL_id),
            SegmentSpec("S9-idle",             seg_T[8], pL_pre,  pR_pre,  False,  0.0,  0.0,  None, RL_id,  RR_id),
            SegmentSpec("S10-idle",            seg_T[9], pL_pre,  pR_pre,  False,  0.0,  0.0,  None, RL_id,  RR_id),
        ]

    segments = make_segments()
    total_T = sum(s.T for s in segments)
    if total_T <= 0.0:
        raise RuntimeError("Total segment time <= 0.")

    def get_segment_index(t_in):
        cum = 0.0
        for i, s in enumerate(segments):
            if t_in < cum + s.T:
                return i, s.name
            cum += s.T
        return len(segments) - 1, segments[-1].name

    # -------------------------------------------------------------------------
    # q_des builders
    # -------------------------------------------------------------------------
    def apply_q17(q_full, q17):
        q = q_full.clone()
        q[:, torch.tensor(ik17_indices, device=device, dtype=torch.long)] = \
            torch.tensor(np.asarray(q17, dtype=float).reshape(17), device=device, dtype=q.dtype)
        return q

    def apply_q29(q_full, q29):
        q = q_full.clone()
        q[:, torch.tensor(ik29_indices, device=device, dtype=torch.long)] = \
            torch.tensor(np.asarray(q29, dtype=float).reshape(29), device=device, dtype=q.dtype)
        return q

    def apply_pos_overrides(q_full, overrides):
        if not overrides:
            return q_full
        q = q_full.clone()
        for jn, v in overrides.items():
            q[:, idx_of_joint(jn)] = float(v)
        return q

    # -------------------------------------------------------------------------
    # PID on waist + arms only (17 DOFs)
    # -------------------------------------------------------------------------
    waist_arm_mask = torch.zeros((num_joints,), device=device, dtype=dtype)
    waist_arm_mask[torch.tensor(ik17_indices, device=device)] = 1.0
    waist_arm_mask = waist_arm_mask.unsqueeze(0).expand(num_envs, -1)

    kp = torch.full((num_envs, num_joints), float(args.kp), device=device, dtype=dtype) * waist_arm_mask
    kd = torch.full((num_envs, num_joints), float(args.kd), device=device, dtype=dtype) * waist_arm_mask
    ki = torch.full((num_envs, num_joints), float(args.ki), device=device, dtype=dtype) * waist_arm_mask

    e_int = torch.zeros((num_envs, num_joints), device=device, dtype=dtype)
    i_clamp = float(args.i_clamp)

    effort_abs_lim = get_effort_abs_limits(robot, device, dtype, num_envs, num_joints)
    if effort_abs_lim is not None:
        effort_abs_lim = effort_abs_lim * float(args.effort_limit_scale)
    if float(args.tau_clamp) > 0.0:
        manual_lim = torch.full((num_envs, num_joints), float(args.tau_clamp), device=device, dtype=dtype)
        effort_abs_lim = manual_lim if effort_abs_lim is None else torch.minimum(effort_abs_lim, manual_lim)

    # Per-region torque limits (independent of the global effort_abs_lim).
    # _arm_idx_t  : long tensor of the 17 waist+arm sim indices, for slice clamping.
    # _arm_lim    : scalar float or None
    # _finger_lim : scalar float or None
    _arm_idx_t  = torch.tensor(ik17_indices, device=device, dtype=torch.long)
    _arm_lim    = float(args.arm_tau_clamp)    if float(args.arm_tau_clamp)    > 0.0 else None
    _finger_lim = float(args.finger_tau_clamp) if float(args.finger_tau_clamp) > 0.0 else None

    # Right-wrist body index (world-frame Z is logged each status print so the
    # "hand drifts up at S3" behaviour is measurable, not just eyeballed).
    _body_names = list(robot.data.body_names)
    _rwrist_bidx = next((i for i, n in enumerate(_body_names)
                         if n in ("right_wrist_yaw_link", "right_rubber_hand", "R_hand_base_link")), None)

    print("[INFO] PH-only script ready. Waist+arms: PID. HAND12: PH IDA-PBC gravity-comp PD.")

    
    # CSV logging
    # -------------------------------------------------------------------------
    log_indices = [idx_of_joint(n) for n in IK29_JOINT_NAMES]
    os.makedirs(args.log_dir, exist_ok=True)
    # sequential run naming: next number not already in the log dir
    _lead, _tail = f"{args.log_prefix}_run", "_q_tau_29.csv"
    _nums = [int(fn[len(_lead):-len(_tail)])
             for fn in os.listdir(args.log_dir)
             if fn.startswith(_lead) and fn.endswith(_tail)
             and fn[len(_lead):-len(_tail)].isdigit()]
    stamp = f"run{max(_nums) + 1 if _nums else 1}"
    csv_path = os.path.join(args.log_dir, f"{args.log_prefix}_{stamp}_q_tau_29.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["time_s", "seg_name"]
        + [n + "_pos"  for n in IK29_JOINT_NAMES]
        + [n + "_qdes" for n in IK29_JOINT_NAMES]
        + [n + "_tau"  for n in IK29_JOINT_NAMES]
        + [n + "_tau_imp" for n in HAND12_NAMES]
        + [n + "_tau_ideal" for n in HAND12_NAMES]
        + [n + "_qdot" for n in HAND12_NAMES]
        + [n + "_gradVd" for n in HAND12_NAMES]
        + [n + "_tipX" for n in HAND12_NAMES]
        + [n + "_tipY" for n in HAND12_NAMES]
        + [n + "_tipZ" for n in HAND12_NAMES]
        + [n + "_refX" for n in HAND12_NAMES]
        + [n + "_refY" for n in HAND12_NAMES]
        + [n + "_refZ" for n in HAND12_NAMES]
        + ["Kt_active", "Bt_active", "contact_active"]
        + ["cube_cam_x", "cube_cam_y", "cube_cam_z"]
        + ["cube_gt_x", "cube_gt_y", "cube_gt_z"]
    )
    csv_file.flush()
    print(f"[INFO] Logging to: {csv_path}")

    # -------------------------------------------------------------------------
    # Full-run diagnostic figures -- accumulate EVERY sampled step across ALL
    # phases (S1..S10). Redrawn from the full history each save (Agg backend;
    # cheap at print_every cadence). Segments are shaded by CONTROL MODE
    # (matching the IDA-PBC docstring): S3/S4/S5 = force mode (contact
    # regulation, orange); S1/S2/S6..S10 = position mode (free motion, gray).
    # For HAND12 these happen to coincide with "object grasped", but the
    # shading is driven by segment membership, NOT ph_ctrl.contact_active --
    # with torque chattering at the rail, that residual test reads "contact"
    # almost the whole run and is not a reliable grasp indicator.
    #
    # Consistent line styles everywhere: solid blue = actual/delivered value,
    # dashed orange = reference/target/ideal value, and (where a 3rd series
    # is a component breakdown rather than a target) dotted green.
    # -------------------------------------------------------------------------
    # Style constants, mode-shading helpers, and segment taxonomy now live in
    # plot_style.py (shared with analyze_controller.py) -- imported above as
    # _ACTUAL_STYLE, _shade_mode, _FORCE_MODE_SEGMENTS, etc.
    _short_names = [n.replace("_proximal", "").replace("_joint", "") for n in HAND12_NAMES]
    _HAND_COLS = {"L": list(range(0, 6)), "R": list(range(6, 12))}

    # growing per-sample history (python lists; stacked to arrays at draw time)
    _hist_t          = []   # (n,)
    _hist_seg        = []   # (n,)     seg_name string
    _hist_q          = []   # (n,12)   actual finger angle
    _hist_qdes       = []   # (n,12)   reference finger angle
    _hist_tautot     = []   # (n,12)   total applied torque (post-clip, delivered)
    _hist_tauimp     = []   # (n,12)   impedance-only torque component
    _hist_tauideal   = []   # (n,12)   PH law + task, BEFORE the finger-tau clamp
    _hist_force_mode = []   # (n,)     bool: force-mode segment (S3-S5)

    def _p(name):
        return os.path.join(args.log_dir, f"{args.log_prefix}_{stamp}_{name}.png")
    _track_paths     = {"L": _p("finger_tracking_L"),      "R": _p("finger_tracking_R")}
    _tau_paths       = {"L": _p("torque_impedance_L"),     "R": _p("torque_impedance_R")}
    _authority_path  = _p("control_authority")

    # _shade_mode/_mode_legend_patches come from plot_style (imported above).
    # _torque_rails/_saturation_duty stay as thin local
    # wrappers since they close over --finger_tau_clamp / mask-then-reduce,
    # which plot_style's torque_rails/saturation_mask/saturation_duty take as
    # explicit arguments instead of a closure.
    def _torque_rails(tautot):
        """Per-joint torque rail: --finger_tau_clamp if configured, else the
        running max|tau| observed so far (auto-detect fallback)."""
        return _torque_rails_base(tautot, float(args.finger_tau_clamp))

    def _saturation_duty(tautot, rails, sel=None):
        return _saturation_duty_base(_saturation_mask(tautot, rails), sel=sel)

    def _save_hand_figure(path, hand, suptitle, ylabel, series, t_arr, mode_arr, hlines=()):
        """series: list of (data(n,12), label, kwargs). Draws a per-hand 2x3
        figure (6 fingers), mode-shaded, with a shared legend."""
        cols = _HAND_COLS[hand]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
        axes = axes.flatten()
        for i, c in enumerate(cols):
            ax = axes[i]
            _shade_mode(ax, t_arr, mode_arr)
            for data, label, kw in series:
                ax.plot(t_arr, data[:, c], label=label, **kw)
            for hv in hlines:
                ax.axhline(hv, color="k", lw=0.6, ls=":")
            ax.set_title(_short_names[c], fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            if i >= 3:
                ax.set_xlabel("time [s]", fontsize=8)
        handles = [Line2D([0], [0], color=kw.get("color", "k"), ls=kw.get("ls", "-"),
                          lw=kw.get("lw", 1.3), label=label) for (_, label, kw) in series]
        handles += _mode_legend_patches()
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
                   fontsize=8, frameon=True)
        fig.suptitle(f"{suptitle}  --  {hand} hand (all phases S1..S10)", fontsize=11)
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))
        fig.savefig(path, dpi=_PLOT_DPI)
        plt.close(fig)

    def _save_joint_tracking_figure(path, hand, t_arr, mode_arr, q, qdes):
        """Per-hand 4x3 grid: for each finger, a value row (actual vs
        reference) paired with an error row e_q(t) = actual - reference."""
        cols = _HAND_COLS[hand]
        err = q - qdes
        fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
        for i, c in enumerate(cols):
            row = (i // 3) * 2
            col = i % 3
            ax_val, ax_err = axes[row, col], axes[row + 1, col]
            _shade_mode(ax_val, t_arr, mode_arr)
            _shade_mode(ax_err, t_arr, mode_arr)
            ax_val.plot(t_arr, q[:, c],    label="actual",    **_ACTUAL_STYLE)
            ax_val.plot(t_arr, qdes[:, c], label="reference", **_REF_STYLE)
            ax_val.set_title(_short_names[c], fontsize=9)
            ax_val.set_ylabel("rad", fontsize=8)
            ax_val.tick_params(labelsize=7)
            ax_err.axhline(0, color="k", lw=0.5)
            ax_err.plot(t_arr, err[:, c], label="error", **_ERROR_STYLE)
            ax_err.set_ylabel("e_q [rad]", fontsize=8)
            ax_err.tick_params(labelsize=7)
            if row + 1 == 3:
                ax_err.set_xlabel("time [s]", fontsize=8)
        handles = [
            Line2D([0], [0], **_ACTUAL_STYLE, label="actual"),
            Line2D([0], [0], **_REF_STYLE, label="reference"),
            Line2D([0], [0], **_ERROR_STYLE, label="error e_q = actual - reference"),
        ] + _mode_legend_patches()
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7.5, frameon=True)
        fig.suptitle(f"Finger joint tracking: actual vs reference, with error  --  "
                     f"{hand} hand (all phases S1..S10)", fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        fig.savefig(path, dpi=_PLOT_DPI)
        plt.close(fig)

    def _save_control_authority_figure(path, mode_arr, tautot):
        """Combined (not per-hand) bar chart: saturation duty cycle per joint,
        split by control mode -- 'where do PH gains matter vs where does the
        rail dominate?'. Kept as ONE figure spanning all 12 joints since the
        L-vs-R comparison is the point of this view."""
        rails = _torque_rails(tautot)
        duty_all   = _saturation_duty(tautot, rails)
        duty_pos   = _saturation_duty(tautot, rails, sel=~mode_arr)
        duty_force = _saturation_duty(tautot, rails, sel=mode_arr)
        x = np.arange(12)
        fig, ax = plt.subplots(figsize=(14, 5))
        w = 0.27
        ax.bar(x - w, 100 * duty_pos,   width=w, color="tab:blue",   label="position mode")
        ax.bar(x,     100 * duty_force, width=w, color=_FORCE_COLOR, label="force mode")
        ax.bar(x + w, 100 * duty_all,   width=w, color="tab:gray",   label="whole run")
        ax.axhline(20, color="tab:red", lw=1.0, ls="--", label="20% tuning-relevance floor")
        ax.set_xticks(x); ax.set_xticklabels(_short_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("time at torque rail [%]")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.set_title(f"Control authority: saturation duty cycle per joint "
                     f"(rail ~ {np.nanmedian(rails):.2f} Nm). High = the rail "
                     f"dominates and PH gains have no effect there.", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=_PLOT_DPI)
        plt.close(fig)

    def _save_all_plots():
        """Redraw and save all 5 diagnostic figures from the full history."""
        if len(_hist_t) < 2:
            return
        t_arr    = np.asarray(_hist_t)
        mode_arr = np.asarray(_hist_force_mode, dtype=bool)
        q        = np.asarray(_hist_q)
        qdes     = np.asarray(_hist_qdes)
        tautot   = np.asarray(_hist_tautot)
        tauimp   = np.asarray(_hist_tauimp)
        tauideal = np.asarray(_hist_tauideal)

        rails = _torque_rails(tautot)
        tautot_n   = tautot   / rails[None, :]
        tauideal_n = tauideal / rails[None, :]
        tauimp_n   = tauimp   / rails[None, :]

        for hand in ("L", "R"):
            _save_joint_tracking_figure(_track_paths[hand], hand, t_arr, mode_arr, q, qdes)
            _save_hand_figure(
                _tau_paths[hand], hand,
                "HAND12 torque / rail: delivered vs PH-law (pre-clip, ideal) vs impedance-only",
                "tau / tau_rail  (dimensionless)",
                [(tautot_n,   "delivered (post-clip)",     _DELIVERED_STYLE),
                 (tauideal_n, "PH law (pre-clip, ideal)",  _IDEAL_STYLE),
                 (tauimp_n,   "impedance-only component",  _IMPEDANCE_STYLE)],
                t_arr, mode_arr, hlines=(-1.0, 1.0))
        _save_control_authority_figure(_authority_path, mode_arr, tautot)
        print(f"  [PLOT] saved 5 diagnostic figures (L/R x {{tracking,torque}} + "
              f"control-authority), n={len(_hist_t)} samples")

    # Loop
    dt = sim.get_physics_dt()

    if args.step_response:
        run_step_response()
        return  # skip the camera/IK/grasp loop below; __main__ closes simulation_app

    step_idx = 0
    sleep_dt = 1.0 / max(1.0, args.max_fps)
    active_seg = -1
    q_des_active = q_init.clone()
    q17_prev   = np.zeros(17)   # last arm IK solution (theta17 ordering)
    q_h12_prev = np.zeros(12)   # last HAND12 analytical IK solution
    RL_default = np.eye(3); RR_default = np.eye(3)
    dq_max = float(args.ik_dq_max_deg) * math.pi / 180.0
    CAM_UPDATE_EVERY = 30   # re-detect cube from RealSense every N steps
    tau_h12_prev: np.ndarray | None = None  # last applied HAND12 torques for contact detection
    tau_h12_ideal_prev: np.ndarray | None = None  # PH law + task, BEFORE the finger-tau clamp

    try:
        while True:
            t = step_idx * dt
            t_in = t % total_T
            seg_i, seg_name = get_segment_index(t_in)

            # --- RealSense cube position update (follow_box.py pattern) ---
            if step_idx % CAM_UPDATE_EVERY == 0:
                _p_cam = cube_centroid_in_camera_frame(rs_depth_cam, class_name="cube", env_id=0)
                if _p_cam is not None:
                    _p_pelv = cube_cam_to_pelvis_from_torso(robot, _p_cam, 0, "torso_link")
                    cube_pos_pelvis_cam[0] = _p_pelv.detach().cpu().numpy()

            seg = segments[seg_i]

            if seg_i != active_seg:
                # Rebuild waypoints from the latest camera estimate before solving IK
                segments = make_segments()
                seg = segments[seg_i]
                active_seg = seg_i
                e_int.zero_()
                ph_ctrl.reset_integral()

                # Force-impedance segment transition hook -- skill spec Sec.5
                q_np_now   = robot.data.joint_pos[0].detach().cpu().numpy()
                q_tree_now = ph_ctrl.sim_to_tree_q(q_np_now)

                # Mode switch + gain schedule (DISC Ch.2 Sec.4.2)
                # Free-motion phases -> position mode (track joint references)
                # Contact phases     -> force mode (regulate tau_spring -> tau_des)
                if seg.name in ("S1-pregrasp", "S2-close-approach",
                                "S9-idle", "S10-idle"):
                    ph_ctrl.set_position_mode()
                    ph_ctrl.Kt_free  = float(args.Kt_free)
                    ph_ctrl.Bt_free  = float(args.Bt_free)
                    ph_ctrl.Kt_grasp = float(args.Kt_free)
                    ph_ctrl.Bt_grasp = float(args.Bt_free)
                    ph_ctrl.contact_active = False

                elif seg.name == "S3-grasp-PH":
                    # Switch to IDA-PBC force regulation.
                    # H_d minimum = desired contact force, not a joint angle.
                    ph_ctrl.set_force_mode(tau_des_grasp)
                    ph_ctrl.Kt_free  = float(args.Kt_free)
                    ph_ctrl.Bt_free  = float(args.Bt_free)
                    ph_ctrl.Kt_grasp = float(args.Kt_close)
                    ph_ctrl.Bt_grasp = float(args.Bt_close)
                    ph_ctrl.x_des          = None            # task impedance off until contact
                    ph_ctrl.contact_active = False            # auto-detect via spring

                elif seg.name in ("S4-lift-PH", "S5-moveair-PH"):
                    # Maintain force regulation; force contact flag on for Kt_grasp
                    ph_ctrl.set_force_mode(tau_des_grasp)
                    ph_ctrl.Kt_grasp = float(args.Kt_grasp)
                    ph_ctrl.Bt_grasp = float(args.Bt_grasp)
                    ph_ctrl.contact_active = True             # forced ON for impedance
                    if ph_ctrl.x_des is None:
                        ph_ctrl.set_task_target(q_tree_now)

                elif seg.name in ("S6-release", "S7-drop", "S8-idle"):
                    # Back to position mode -- fingers open to reference angles
                    ph_ctrl.set_position_mode()
                    ph_ctrl.Kt_free  = float(args.Kt_free)
                    ph_ctrl.Bt_free  = float(args.Bt_free)
                    ph_ctrl.Kt_grasp = float(args.Kt_free)
                    ph_ctrl.Bt_grasp = float(args.Bt_free)
                    ph_ctrl.x_des          = None
                    ph_ctrl.contact_active = False
                q_des = robot.data.joint_pos.clone()

                if seg.hands_pL is not None and seg.hands_pR is not None:
                    RL_seg = seg.RL_target if seg.RL_target is not None else RL_default
                    RR_seg = seg.RR_target if seg.RR_target is not None else RR_default
                    q_sol17, ik_info = ik_two_hands_dls(
                        pL_target=np.asarray(seg.hands_pL).reshape(3),
                        pR_target=np.asarray(seg.hands_pR).reshape(3),
                        RL_target=np.asarray(RL_seg).reshape(3, 3),
                        RR_target=np.asarray(RR_seg).reshape(3, 3),
                        use_waist=bool(args.use_waist), q0=q17_prev,
                        max_iters=int(args.ik_max_iters), lamb=float(args.ik_lambda),
                        alpha=float(args.ik_alpha), tol_p=float(args.ik_tol_p),
                        tol_w=float(args.ik_tol_w), dq_max=dq_max, w_pos=1.0, w_rot=1.0,
                    )
                    if not ik_info.converged:
                        pass
                    else:
                        fe = ik_info.final_errors
                        print(f"[IK]  {seg.name} converged in {ik_info.iters} iters: "
                              f"eL_p={fe['eL_p_norm']:.4f}m eR_p={fe['eR_p_norm']:.4f}m")
                    q17_prev = np.asarray(q_sol17).reshape(17)
                    q_des = apply_q17(q_des, q17_prev)

                # â”€â”€ Analytical finger IK â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                q_sim_cpu = robot.data.joint_pos[0].detach().cpu().numpy()

                if seg.use_grasp_ik:
                    # FK-project each fingertip onto cylinder surface, then solve analytically.
                    cx_, cy_, cz_ = cube_pos_pelvis_cam[0].tolist()
                    grasp_tgts = _compute_grasp_targets(q17_prev, q_sim_cpu, cx_, cy_, cz_)
                    th29_0 = _build_theta29(q17_prev, q_sim_cpu)
                    res = ik_fingers_analytical(grasp_tgts, th29_0, validate=False)
                    q_h12_prev[:6] = res.q_hand6_L
                    q_h12_prev[6:] = res.q_hand6_R
                    print(f"[FINGER IK] grasp: L={np.round(res.q_hand6_L, 3).tolist()}  "
                          f"R={np.round(res.q_hand6_R, 3).tolist()}")

                elif seg.finger_curl_L is not None:
                    # Direct curl-fraction mapping: 0 = joint lo (open), 1 = joint hi (closed).
                    th29_0 = _build_theta29(q17_prev, q_sim_cpu)
                    q_curl = ik_fingers_from_curl(
                        curl_L=seg.finger_curl_L,
                        curl_R=seg.finger_curl_R if seg.finger_curl_R is not None else seg.finger_curl_L,
                        theta29_0=th29_0,
                    )
                    q_h12_prev[:6] = q_curl[17:23]
                    q_h12_prev[6:] = q_curl[23:29]
                # else: hold â€” q_h12_prev unchanged from previous segment

                # Write HAND12 angles into q_des via the IK29 path
                q29_apply = np.zeros(29)
                q29_apply[:17] = q17_prev
                q29_apply[17:23] = q_h12_prev[:6]
                q29_apply[23:29] = q_h12_prev[6:]
                q_des = apply_q29(q_des, q29_apply)

                # Non-finger joint overrides (arm / waist overrides, if any)
                q_des = apply_pos_overrides(q_des, seg.joint_pos_overrides)
                q_des_active = q_des
                print(f"[SEG] -> {seg_name}")

                # Video: open writer on S1_pregrasp entry, close when S9_idle begins
                if args.video:
                    if seg_name == "S1-pregrasp" and not _video_recording:
                        _video_recording = True
                        _stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        _video_path = os.path.join(
                            args.video_folder, f"lift_fingerPH_{_stamp}.mp4"
                        )
                        _fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        _video_writer = cv2.VideoWriter(
                            _video_path, _fourcc, float(args.video_fps),
                            (args.video_width, args.video_height)
                        )
                        _video_frames_captured = 0
                        print(f"[VIDEO] Started recording (S1-pregrasp..S8-idle) -> {_video_path}")
                    elif seg_name == "S9-idle" and _video_recording:
                        _video_recording = False
                        if _video_writer is not None:
                            _video_writer.release()
                            _video_writer = None
                            print(f"[VIDEO] Saved {_video_frames_captured} frames -> {_video_path}")

            # PID on waist+arms
            q = robot.data.joint_pos
            qd = robot.data.joint_vel
            e = q_des_active - q
            e_dot = -qd
            if float(args.ki) != 0.0:
                e_int = torch.clamp(e_int + e * dt, -i_clamp, i_clamp)
            tau = kp * e + kd * e_dot + ki * e_int + get_gravity_comp_torque()

            # Optional per-region arm clamp (independent of global effort_abs_lim).
            # Applied before the PH overwrite so finger slots are unaffected.
            if _arm_lim is not None:
                tau[:, _arm_idx_t] = torch.clamp(tau[:, _arm_idx_t], -_arm_lim, _arm_lim)

            # Force-impedance PH on HAND12 (overwrites arm-PID zeros at those indices)
            q_des_h12 = q_des_active[0, ph_hand12_sim_idx].detach().cpu().numpy()
            ph_ctrl.q_des_hand12 = q_des_h12
            _tau_h12_last = None
            _tau_h12_ideal_last = None
            for env in range(num_envs):
                _tau_h12_t = ph_ctrl.compute_torch(
                    robot.data.joint_pos[env], robot.data.joint_vel[env],
                    dt=dt, tau_applied=tau_h12_prev,
                )
                _tau_h12_ideal_last = _tau_h12_t  # IDA-PBC law + task, BEFORE any clamp
                # Optional per-region finger clamp (independent of global effort_abs_lim).
                if _finger_lim is not None:
                    _tau_h12_t = torch.clamp(_tau_h12_t, -_finger_lim, _finger_lim)
                tau[env, ph_hand12_sim_idx] = _tau_h12_t
                _tau_h12_last = _tau_h12_t
            if _tau_h12_last is not None:
                tau_h12_prev = _tau_h12_last.detach().cpu().numpy()  # env=0 torques for next contact detect
            if _tau_h12_ideal_last is not None:
                tau_h12_ideal_prev = _tau_h12_ideal_last.detach().cpu().numpy()

            if effort_abs_lim is not None:
                tau = torch.clamp(tau, -effort_abs_lim, effort_abs_lim)

            robot.set_joint_effort_target(tau)

            scene.write_data_to_sim(); sim.step(); scene.update(dt)
            step_idx += 1

            # Capture frame during S7_drop
            if (
                _video_writer is not None
                and _video_annotator is not None
                and _video_recording
                and (step_idx % args.video_interval) == 0
                and (args.video_length == 0 or _video_frames_captured < args.video_length)
            ):
                rgba = _video_annotator.get_data()
                if rgba is not None and rgba.size > 0:
                    bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
                    _video_writer.write(bgr)
                    _video_frames_captured += 1

            if (step_idx % int(args.log_every)) == 0:
                tau_all = getattr(robot.data, "applied_torque", None)
                if tau_all is None:
                    tau_all = getattr(robot.data, "joint_torques", None)
                if tau_all is not None:
                    pos_29  = robot.data.joint_pos[0, log_indices].detach().cpu().numpy()
                    qdes_29 = q_des_active[0, log_indices].detach().cpu().numpy()
                    tau_29  = tau_all[0, log_indices].detach().cpu().numpy()
                    # impedance-only torque = total - pure PH base torque
                    tau_imp_12 = (tau_h12_prev - ph_ctrl.last_u_ph
                                  if tau_h12_prev is not None
                                  else np.zeros(12))
                    # "ideal" PH law (IDA-PBC + task term) BEFORE the --finger_tau_clamp
                    # clip -- lets offline analysis see exactly what saturation removed.
                    tau_ideal_12 = (tau_h12_ideal_prev if tau_h12_ideal_prev is not None
                                     else np.zeros(12))
                    # Fingertip positions (tree frame), HAND12 order, (12,3).
                    _q_sim_full = robot.data.joint_pos[0].detach().cpu().numpy()
                    _tip_xyz_now = _ftip_jac.tip_positions(ph_ctrl.sim_to_tree_q(_q_sim_full))
                    # Reference = where each finger should be touching the object,
                    # i.e. the live cube pose projected radially onto its own surface.
                    _cx_live, _cy_live, _cz_live = _live_object_pos_pelvis(scene, robot)
                    _ref_xyz_now = _object_surface_ref(_tip_xyz_now, _cx_live, _cy_live)
                    # Raw joint velocity + force-mode shaping-potential gradient
                    # (grad_Vd = Kf*(tau_spring(q_c) - tau_d)) -- diagnostics for
                    # the per-step torque chattering found by comparing against
                    # the Jelle Vinke reference logs.
                    _qdot_h12_now = robot.data.joint_vel[0, ph_hand12_sim_idx].detach().cpu().numpy()
                    _qc_now = _q_sim_full[ph_ctrl.hand12_sim_indices]
                    _gradVd_now = ph_ctrl.grad_Vd(_qc_now)
                    _Kt_now, _Bt_now = ph_ctrl._active_Kt_Bt()
                    csv_writer.writerow(
                        [float(step_idx * dt), seg_name]
                        + pos_29.tolist()
                        + qdes_29.tolist()
                        + tau_29.tolist()
                        + tau_imp_12.tolist()
                        + tau_ideal_12.tolist()
                        + _qdot_h12_now.tolist()
                        + _gradVd_now.tolist()
                        + _tip_xyz_now[:, 0].tolist()
                        + _tip_xyz_now[:, 1].tolist()
                        + _tip_xyz_now[:, 2].tolist()
                        + _ref_xyz_now[:, 0].tolist()
                        + _ref_xyz_now[:, 1].tolist()
                        + _ref_xyz_now[:, 2].tolist()
                        + [float(_Kt_now), float(_Bt_now), int(ph_ctrl.contact_active)]
                        + cube_pos_pelvis_cam[0].tolist()
                        + [float(_cx_live), float(_cy_live), float(_cz_live)]
                    )
                    csv_file.flush()

            if step_idx % args.print_every == 0:
                q_np  = robot.data.joint_pos[0, ph_hand12_sim_idx].detach().cpu().numpy()
                qd_np = q_des_active[0, ph_hand12_sim_idx].detach().cpu().numpy()
                # Accumulate full-run history across ALL phases (S1..S10).
                _tau_all_p = getattr(robot.data, "applied_torque", None)
                if _tau_all_p is None:
                    _tau_all_p = getattr(robot.data, "joint_torques", None)
                _tau_total_12_p = (_tau_all_p[0, ph_hand12_sim_idx].detach().cpu().numpy()
                                   if _tau_all_p is not None else np.zeros(12))
                _tau_imp_12_p = (tau_h12_prev - ph_ctrl.last_u_ph
                                 if tau_h12_prev is not None else np.zeros(12))
                _tau_ideal_12_p = (tau_h12_ideal_prev if tau_h12_ideal_prev is not None
                                    else np.zeros(12))
                _hist_t.append(float(t))
                _hist_seg.append(seg_name)
                _hist_q.append(q_np)
                _hist_qdes.append(qd_np)
                _hist_tautot.append(_tau_total_12_p)
                _hist_tauimp.append(_tau_imp_12_p)
                _hist_tauideal.append(_tau_ideal_12_p)
                _hist_force_mode.append(seg_name in _FORCE_MODE_SEGMENTS)
                _save_all_plots()
                rms = float(np.sqrt(np.mean((q_np - qd_np) ** 2)))
                _wz = (f"  RwristZ={robot.data.body_pos_w[0, _rwrist_bidx, 2].item():.4f}m"
                       if _rwrist_bidx is not None else "")
                print(f"[{seg_name:22s}] step={step_idx:06d} t={t:7.3f}s  RMS={rms:.4f} rad{_wz}")

            if sleep_dt > 0:
                time.sleep(sleep_dt)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        try:
            _save_all_plots()   # final redraw so the last phases are captured
            print("[INFO] Saved final per-hand phase plots.")
        except Exception as e:
            print(f"[PLOT] Error saving final plots: {e}")
        try:
            csv_file.flush(); csv_file.close()
            print(f"[INFO] Closed CSV: {csv_path}")
        except Exception:
            pass
        if _video_writer is not None:
            try:
                _video_writer.release()
                print(f"[VIDEO] Saved {_video_frames_captured} frames -> {_video_path}")
            except Exception as e:
                print(f"[VIDEO] Error finalizing video: {e}")
        if _video_render_product is not None:
            try:
                _video_render_product.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()
    simulation_app.close()