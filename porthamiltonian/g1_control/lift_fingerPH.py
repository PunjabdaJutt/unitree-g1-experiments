# lift_fingerPH.py
"""
PH-only finger control variant of lift_fingerTorque.py.

- Waist + arms: position PID (unchanged).
- HAND12 (12 finger proximal/thumb-base joints): PH IDA-PBC gravity-comp PD is
  the SOLE actuator. No finger position PID, no tau-PI, no thumb-yaw hold.
- Segments still drive q_des for the fingers via joint_pos_overrides; PH tracks it.

Run:
  conda activate env_isaaclab
  ./isaaclab.bat -p scripts/g1_control/lift_fingerPH.py --num_envs 1 --enable_cameras \\
      --ph_kp 4 --ph_kd 1.4 --ph_ki 0 --ph_i_clamp 0.35 \\
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

# IK params
parser.add_argument("--use_waist", action="store_true")
parser.add_argument("--ik_max_iters", type=int, default=2000)
parser.add_argument("--ik_lambda", type=float, default=0.2)
parser.add_argument("--ik_alpha", type=float, default=0.5)
parser.add_argument("--ik_tol_p", type=float, default=1e-4)
parser.add_argument("--ik_tol_w", type=float, default=1e-4)
parser.add_argument("--ik_dq_max_deg", type=float, default=1.0)

# PID gains (waist + arms only)
parser.add_argument("--kp", type=float, default=4.0)
parser.add_argument("--kd", type=float, default=1.0)
parser.add_argument("--ki", type=float, default= 1)
parser.add_argument("--i_clamp", type=float, default=0.35)

# PH finger controller
parser.add_argument("--ph_urdf_path", type=str,
                    default="scripts/g1_control/robot/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")
parser.add_argument("--ph_kp", type=float, default=4)
parser.add_argument("--ph_kd", type=float, default=1)
parser.add_argument("--ph_ki", type=float, default=0.0)
parser.add_argument("--ph_i_clamp", type=float, default=0.35)

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
parser.add_argument("--grasp_torque", type=float, default=0.3,
                    help="Desired contact torque for thumb-pitch joints in S3-S5 (Nm). "
                         "Finger joints receive 0.25x this value.")

# Torque limiting
parser.add_argument("--tau_clamp", type=float, default=0.0)
parser.add_argument("--effort_limit_scale", type=float, default=1.0)
parser.add_argument("--arm_tau_clamp", type=float, default=0.0,
                    help="Extra torque clamp applied only to waist+arm (17 DOF) joints [Nm]. "
                         "0 = no extra clamp beyond --tau_clamp / URDF effort limits.")
parser.add_argument("--finger_tau_clamp", type=float, default=0.0,
                    help="Torque clamp applied only to HAND12 (12 DOF) finger joints [Nm]. "
                         "0 = no extra clamp.")

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

from robotics.ik_hands import ik_two_hands_dls
from robotics.ik_fingers_analytical import ik_fingers_analytical, ik_fingers_from_curl
from robotics.fk_fingers import fk_fingers as _fk_fingers
from robotics.dh_table import compute_all_dh_chains as _compute_dh_chains


# -----------------------------------------------------------------------------
# Scene constants
# -----------------------------------------------------------------------------
TABLE_Z = 0.80
BOX_LEN_X = 0.06
BOX_LEN_Y = 0.03
BOX_LEN_Z = 0.10
BOX_HALF_Z = BOX_LEN_Z * 0.5
CUBE_MASS = 0.05

RS_ROOT = "{ENV_REGEX_NS}/Robot/torso_link/d435_link/realsense_d455/RSD455"

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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
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
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/realsense_d455",
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(
        device=args.device,
        gravity=(0.0, 0.0, -9.8),
        physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1),
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

    _ph_urdf = parse_urdf(args.ph_urdf_path)
    _ph_tree = build_tree(_ph_urdf)
    _ph_sys  = PHSystem(_ph_tree, control_joint_names=HAND12_NAMES)


    _ftip_jac = FingertipJacobian(
        ph_sys=_ph_sys,
        tip_link_names=TIP_LINK_NAMES,
        hand12_tree_idx=np.array(_ph_sys.control_indices),
    )

    ph_ctrl = PHContactForceController(
        ph_sys=_ph_sys,
        ftip_jac=_ftip_jac,
        sim_joint_names=joint_names,
        Kp=np.full(12, float(args.ph_kp)),
        Kd=np.full(12, float(args.ph_kd)),
        Ki=np.full(12, float(args.ph_ki)),
        i_clamp=float(args.ph_i_clamp),
        Kt_free=float(args.Kt_free),
        Kt_grasp=float(args.Kt_grasp),
        Bt_free=float(args.Bt_free),
        Bt_grasp=float(args.Bt_grasp),
        contact_threshold=float(args.contact_threshold),
        Kf=float(args.Kf),
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
        palmX_R_s2 = 0.0    # S2/S3 right hand: at cube X (no offset)
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
        pL_pre  = np.array([cx - palmX,  cy + (hy + y_wide),  cz + z_abv])
        pR_pre  = np.array([cx - palmX+0.5,  cy - (hy + (y_wide-0.05)),  cz - z_below])

        # S2: close approach -- right palm below cube centre, facing cube
        pL_cls  = np.array([cx - palmX+0.03,  cy + (hy + y_near),  cz + z_abv])
        pR_cls  = np.array([cx -palmX_R_s2,  cy - (hy + y_near),  cz + z_abv])
        pR_clss = np.array([cx -palmX_R_s2,  cy - (hy+  y_near),  cz + z_abv ]) 
        # S3: grasp IK target drops further to pre-compensate contact-force upward drift
        pL_grs  = np.array([cx - palmX,  cy + (hy + y_near),  cz + z_abv + z_grasp_bias])
        pR_grs  = np.array([cx,          cy - (hy + y_near),  cz + z_abv + z_grasp_bias])

        # S4: vertical lift -- keep palm contact geometry, raise Z
        pL_lft  = np.array([cx - palmX,  cy + (hy + y_near),  cz + z_lift])
        pR_lft  = np.array([cx - palmX,  cy - (hy + y_near),  cz + z_lift])

        # S5-S7: transport in -Y toward drop box, maintain lift height
        pL_box  = np.array([cx - palmX,  cy + (hy + y_near),  cz + z_lift])
        pR_box  = np.array([cx - palmX,  cy - (hy + y_box),   cz + z_lift])

        # S8-S10: safe idle retreat -- back to wide hover at cube height
        pL_idle = np.array([cx,     cy + (hy + y_wide),  cz])
        pR_idle = np.array([cx - palmX,  cy - (hy + y_near),  cz + z_lift])
        partial_curl=np.array([0, 0, 0, 0, 0, 0]) #(for setting the thumbs yaw/pitch)
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

    print("[INFO] PH-only script ready. Waist+arms: PID. HAND12: PH IDA-PBC gravity-comp PD.")

    
    # CSV logging
    # -------------------------------------------------------------------------
    log_indices = [idx_of_joint(n) for n in IK29_JOINT_NAMES]
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.log_dir, f"{args.log_prefix}_{stamp}_q_tau_29.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["time_s"]
        + [n + "_pos"  for n in IK29_JOINT_NAMES]
        + [n + "_qdes" for n in IK29_JOINT_NAMES]
        + [n + "_tau"  for n in IK29_JOINT_NAMES]
        + [n + "_tau_imp" for n in HAND12_NAMES]
        + ["contact_active"]
    )
    csv_file.flush()
    print(f"[INFO] Logging to: {csv_path}")

    # -------------------------------------------------------------------------
    # Live finger tracking figure (12 subplots, actual vs reference)
    # -------------------------------------------------------------------------
    _PLOT_WINDOW = 300          # number of steps shown in the rolling window
    _t_buf       = np.full(_PLOT_WINDOW, np.nan)
    _q_buf       = np.full((_PLOT_WINDOW, 12), np.nan)
    _qd_buf      = np.full((_PLOT_WINDOW, 12), np.nan)

    _fig, _axes = plt.subplots(4, 3, figsize=(14, 9), sharex=True)
    _axes = _axes.flatten()
    _lines_actual = []
    _lines_ref    = []
    _short_names  = [n.replace("_proximal", "").replace("_joint", "") for n in HAND12_NAMES]
    for k, ax in enumerate(_axes):
        la, = ax.plot([], [], color="tab:blue",   lw=1.2, label="actual")
        lr, = ax.plot([], [], color="tab:orange", lw=1.2, ls="--", label="ref")
        _lines_actual.append(la)
        _lines_ref.append(lr)
        ax.set_title(_short_names[k], fontsize=8)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
    _fig.suptitle("Finger joint tracking  (actual vs reference)", fontsize=10)
    _fig.tight_layout()
    _plot_path = os.path.join(args.log_dir, f"{args.log_prefix}_{stamp}_finger_tracking.png")

    def _update_live_plot(t_now, q_h12, qd_h12):
        """Roll buffers and save PNG snapshot. Called every print_every steps.
        Agg backend is used to avoid conflict with Isaac Sim's carb event loop."""
        _t_buf[:-1]    = _t_buf[1:]
        _q_buf[:-1]    = _q_buf[1:]
        _qd_buf[:-1]   = _qd_buf[1:]
        _t_buf[-1]     = t_now
        _q_buf[-1]     = q_h12
        _qd_buf[-1]    = qd_h12
        valid = ~np.isnan(_t_buf)
        t_v = _t_buf[valid]
        for k in range(12):
            _lines_actual[k].set_data(t_v, _q_buf[valid, k])
            _lines_ref[k].set_data(t_v, _qd_buf[valid, k])
            _axes[k].relim(); _axes[k].autoscale_view()
        _fig.savefig(_plot_path, dpi=90)
        print(f"  [PLOT] saved -> {_plot_path}")

    # Loop
    dt = sim.get_physics_dt()
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
            for env in range(num_envs):
                _tau_h12_t = ph_ctrl.compute_torch(
                    robot.data.joint_pos[env], robot.data.joint_vel[env],
                    dt=dt, tau_applied=tau_h12_prev,
                )
                # Optional per-region finger clamp (independent of global effort_abs_lim).
                if _finger_lim is not None:
                    _tau_h12_t = torch.clamp(_tau_h12_t, -_finger_lim, _finger_lim)
                tau[env, ph_hand12_sim_idx] = _tau_h12_t
                _tau_h12_last = _tau_h12_t
            if _tau_h12_last is not None:
                tau_h12_prev = _tau_h12_last.detach().cpu().numpy()  # env=0 torques for next contact detect

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
                    csv_writer.writerow(
                        [float(step_idx * dt)]
                        + pos_29.tolist()
                        + qdes_29.tolist()
                        + tau_29.tolist()
                        + tau_imp_12.tolist()
                        + [int(ph_ctrl.contact_active)]
                    )
                    csv_file.flush()

            if step_idx % args.print_every == 0:
                q_np  = robot.data.joint_pos[0, ph_hand12_sim_idx].detach().cpu().numpy()
                qd_np = q_des_active[0, ph_hand12_sim_idx].detach().cpu().numpy()
                _update_live_plot(t, q_np, qd_np)
                rms = float(np.sqrt(np.mean((q_np - qd_np) ** 2)))
                print(f"[{seg_name:22s}] step={step_idx:06d} t={t:7.3f}s  RMS={rms:.4f} rad")

            if sleep_dt > 0:
                time.sleep(sleep_dt)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
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