# two_arms_waist_fingers_segments_camera_pid_effort.py
"""
Edit the 10 segments for different actions

Run with arguments:
--use_waist --enable_cameras
"""

import argparse
import time
import math
import os
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import torch
import numpy as np

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Two arms + waist + fingers: segment-based targets (wrist IK or fingertip IK), effort PID, CSV logging."
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_fps", type=float, default=120.0)
parser.add_argument("--print_every", type=int, default=60)

# 10 segments
NUM_SEGS = 10
for k in range(1, NUM_SEGS + 1):
    parser.add_argument(f"--t_seg_{k}", type=float, default=1.5, help=f"Duration of segment {k} in seconds.")

# IK params (hands)
parser.add_argument("--use_waist", action="store_true", help="If set, solve waist + both arms. If not set, waist locked.")
parser.add_argument("--ik_max_iters", type=int, default=2000)
parser.add_argument("--ik_lambda", type=float, default=0.2)
parser.add_argument("--ik_alpha", type=float, default=0.5)
parser.add_argument("--ik_tol_p", type=float, default=1e-4)
parser.add_argument("--ik_tol_w", type=float, default=1e-4)
parser.add_argument("--ik_dq_max_deg", type=float, default=1.0)

# PID gains (waist+arms)
parser.add_argument("--kp", type=float, default=4.0)
parser.add_argument("--kd", type=float, default=1.0)
parser.add_argument("--ki", type=float, default=0.0)

# Optional separate finger gains (when finger position control is enabled by a segment)
parser.add_argument("--kp_finger", type=float, default=1.5)
parser.add_argument("--kd_finger", type=float, default=0.8)
parser.add_argument("--ki_finger", type=float, default=0.0)

# -------------------------------------------------------------------------
# Closed-loop finger torque controller
# -------------------------------------------------------------------------
parser.add_argument("--finger_tau_kp", type=float, default=0.0, help="P gain on (tau_des - tau_meas) for finger torque control.")
parser.add_argument("--finger_tau_ki", type=float, default=15.0, help="I gain on (tau_des - tau_meas) for finger torque control.")
parser.add_argument("--finger_tau_i_clamp", type=float, default=0.5, help="Anti-windup clamp for finger torque integrator.")
parser.add_argument("--finger_tau_total", type=float, default=1, help="Total torque budget per hand (Nm).")
parser.add_argument("--thumb_alpha", type=float, default=0.5, help="Fraction of total torque assigned to thumb pitch (rest split over 4 fingers).")

# Anti-windup & torque limiting helpers
parser.add_argument("--i_clamp", type=float, default=0.35)
parser.add_argument("--tau_clamp", type=float, default=0.0, help="Manual abs torque clamp (Nm). 0 disables.")
parser.add_argument("--effort_limit_scale", type=float, default=1.0)

# Torque CSV logging
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--log_every", type=int, default=1)
parser.add_argument("--log_prefix", type=str, default="robot")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Isaac Lab imports (after app launch)
# -----------------------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from scripts.g1_control.robot.unitree import UNITREE_G1_29DOF_CFG_HANDS as ROBOT_CFG
from scripts.g1_control.robot.unitree import UNITREE_G1_29DOF_CFG_HANDS_GRAVITY as ROBOT_CFG_GRAVITY
import isaaclab.utils.math as math_utils
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg

# --- Existing wrist IK (17 DoF) ---
from robotics.ik_hands import ik_two_hands_dls
from robotics.ik_fingers import ik_two_hands_with_fingers_dls
from robotics.fk_hands import fk_hands
from robotics.fk_fingers import fk_fingers


# -----------------------------------------------------------------------------
# Scene constants
# -----------------------------------------------------------------------------
TABLE_Z = 0.80
BOX_LEN_X = 0.05
BOX_LEN_Y = 0.05
BOX_LEN_Z = 0.10
BOX_HALF_Z = BOX_LEN_Z * 0.5
CUBE_MASS = 0.1

RS_ROOT = "{ENV_REGEX_NS}/Robot/torso_link/d435_link/realsense_d455/RSD455"


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
## camera object recognition logic
def _find_semantic_id_for_class(sem_info: dict | None, class_name: str) -> int | None:
    if sem_info is None:
        return None
    id_to_labels = sem_info.get("idToLabels") or sem_info.get("id_to_labels")
    if isinstance(id_to_labels, dict):
        for k, v in id_to_labels.items():
            if isinstance(v, dict) and v.get("class") == class_name:
                try:
                    return int(k)
                except Exception:
                    pass
            if isinstance(v, str) and v == class_name:
                try:
                    return int(k)
                except Exception:
                    pass

    labels = sem_info.get("labels")
    if isinstance(labels, list):
        for item in labels:
            if isinstance(item, dict) and item.get("class") == class_name:
                try:
                    return int(item.get("id"))
                except Exception:
                    pass
    return None


def cube_centroid_in_camera_frame(cam, class_name: str = "cube", env_id: int = 0, max_samples: int = 5000):
    out = cam.data.output
    if ("semantic_segmentation" not in out) or (("depth" not in out) and ("distance_to_image_plane" not in out)):
        return None

    depth_key = "depth" if "depth" in out else "distance_to_image_plane"
    sem = out["semantic_segmentation"]  # (N,H,W,1)
    dep = out[depth_key]  # (N,H,W,1)

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
        ys = ys[sel]
        xs = xs[sel]

    z = dep_e[ys, xs]
    valid = torch.isfinite(z) & (z > 0.0)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if ys.numel() == 0:
        return None

    K = cam.data.intrinsic_matrices[env_id]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = xs.to(z.dtype)
    v = ys.to(z.dtype)

    X = (u - cx) / fx * z
    Y = (v - cy) / fy * z
    Z = z
    return torch.stack([X, Y, Z], dim=-1).mean(dim=0)


def cube_cam_to_pelvis_from_torso(
    robot,
    p_cam_optical: torch.Tensor,
    env_id: int = 0,
    torso_name: str = "torso_link",
    # URDF fixed joint: torso_link -> d435_link
    d435_xyz=(0.0576235, 0.01753, 0.42987),
    d435_rpy=(0.0, 0.8307767239493009, 0.0),
    cam_offset_x=0.01,
) -> torch.Tensor:
    """
    camera optical point -> pelvis frame using:
      world_T_torso (from articulation) * torso_T_d435 (URDF) * d435_T_cam (offset)
    plus optical->link axes conversion:
      optical: x right, y down, z forward
      link:    x forward, y left, z up
      p_link = [ z, -x, -y ]
    """
    device = p_cam_optical.device
    dtype = p_cam_optical.dtype

    body_names = list(getattr(robot.data, "body_names", [])) or list(getattr(robot.data, "link_names", []))
    if torso_name not in body_names:
        raise RuntimeError(f"Body/link '{torso_name}' not found. Available (first 30): {body_names[:30]}")
    torso_idx = body_names.index(torso_name)

    torso_pos_w = robot.data.body_pos_w[env_id, torso_idx, :]
    torso_quat_w = robot.data.body_quat_w[env_id, torso_idx, :]  # (wxyz)

    pelvis_pos_w = robot.data.root_link_pos_w[env_id]
    pelvis_quat_w = robot.data.root_link_quat_w[env_id]  # (wxyz)

    t_t_d = torch.tensor(d435_xyz, device=device, dtype=dtype)
    rpy = torch.tensor(d435_rpy, device=device, dtype=dtype)
    q_t_d = math_utils.quat_from_euler_xyz(rpy[0], rpy[1], rpy[2])  # wxyz

    t_d_c = torch.tensor([cam_offset_x, 0.0, 0.0], device=device, dtype=dtype)
    p_cam_in_d435 = torch.stack([p_cam_optical[2], -p_cam_optical[0], -p_cam_optical[1]], dim=0)

    p_d = t_d_c + p_cam_in_d435
    p_t = t_t_d + math_utils.quat_apply(q_t_d.unsqueeze(0), p_d.unsqueeze(0))[0]
    p_w = torso_pos_w + math_utils.quat_apply(torso_quat_w.unsqueeze(0), p_t.unsqueeze(0))[0]

    dp_w = (p_w - pelvis_pos_w).unsqueeze(0)
    p_pelvis = math_utils.quat_apply_inverse(pelvis_quat_w.unsqueeze(0), dp_w)[0]
    return p_pelvis


# limit joint torque
def get_effort_abs_limits(robot, device, dtype, num_envs: int, num_joints: int) -> torch.Tensor | None:
    lim = getattr(robot.data, "joint_effort_limits", None)
    if lim is None:
        return None

    lim_t = torch.as_tensor(lim, device=device, dtype=dtype)

    if lim_t.ndim == 2 and lim_t.shape[-1] == 2:
        absmax = torch.maximum(lim_t[:, 0].abs(), lim_t[:, 1].abs())
        absmax = absmax.unsqueeze(0).expand(num_envs, -1)
        return absmax
    if lim_t.ndim == 3 and lim_t.shape[-1] == 2:
        absmax = torch.maximum(lim_t[..., 0].abs(), lim_t[..., 1].abs())
        return absmax
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
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100, 100)),
    )

    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.95, 0.95, 0.95), intensity=3000.0),
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=[0.9, 1.0, 0.05],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0025, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, TABLE_Z - 0.025), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=[BOX_LEN_X, BOX_LEN_Y, BOX_LEN_Z],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            physics_material=RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=2.5, restitution=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
            semantic_tags=[("class", "cube")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.45, TABLE_Z + BOX_HALF_Z + 0.02),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            copy_from_source=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.45, 0.85),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    robot = ROBOT_CFG_GRAVITY.replace(prim_path="{ENV_REGEX_NS}/Robot")

    realsense_d455 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/realsense_d455",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Sensors/Intel/RealSense/rsd455.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.01, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    rs_depth = CameraCfg(
        prim_path=f"{RS_ROOT}/Camera_Pseudo_Depth",
        spawn=None,
        update_period=0.0,
        width=640,
        height=480,
        data_types=["depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
    )


# -----------------------------------------------------------------------------
# Joint name mapping
# -----------------------------------------------------------------------------
IK17_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# 29D vector order: [waist(3), left arm(7), right arm(7), left hand(6), right hand(6)]
IK29_JOINT_NAMES = [
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    # left hand (map to proximal/thumb-base joints only)
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_ring_proximal_joint",
    "L_pinky_proximal_joint",
    # right hand
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
]

HAND12_NAMES = IK29_JOINT_NAMES[17:]


# -----------------------------------------------------------------------------
# Segment specification
# -----------------------------------------------------------------------------
@dataclass
class SegmentSpec:
    name: str
    T: float

    # If set: use ik_hands to compute 17D target from pL/pR (pelvis frame)
    hands_pL: Optional[np.ndarray] = None
    hands_pR: Optional[np.ndarray] = None

    # If set: use fingertip IK to compute 29D target (waist+arms+hands)
    fingertip_targets: Optional[dict] = None  # dict structure you provided

    # Optional direct position overrides in radians (applied after IK results)
    joint_pos_overrides: Optional[Dict[str, float]] = None

    # Optional direct torque overrides in Nm (overwrites PID for those joints)
    joint_tau_overrides: Optional[Dict[str, float]] = None

    # orientation for the wrists
    RL_target: Optional[np.ndarray] = None  # left wrist orientation, pelvis frame
    RR_target: Optional[np.ndarray] = None  # right wrist orientation, pelvis frame

    # NEW: enable closed-loop finger torque control during this segment
    finger_tau_control: bool = False
    finger_tau_total: Optional[float] = None  # per-hand torque budget (Nm). None -> args.finger_tau_total
    thumb_alpha: Optional[float] = None       # None -> args.thumb_alpha


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Simulation
    sim_cfg = sim_utils.SimulationCfg(device=args.device, gravity=(0.0, 0.0, -9.8))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([0, 2.2, 1.5], [0.2, 0.4, 0.9])

    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=3.0))
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    rs_depth_cam = scene["rs_depth"]

    # Warm-up steps so camera outputs are valid
    dt_boot = sim.get_physics_dt()
    for _ in range(30):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt_boot)

    # Cube center from camera once
    cube_pos_pelvis_cam = None
    for _ in range(300):
        p_cam = cube_centroid_in_camera_frame(rs_depth_cam, class_name="cube", env_id=0)
        if p_cam is not None:
            p_pelvis = cube_cam_to_pelvis_from_torso(robot=robot, p_cam_optical=p_cam, env_id=0, torso_name="torso_link")
            cube_pos_pelvis_cam = p_pelvis.detach().cpu().numpy()
            break
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt_boot)

    if cube_pos_pelvis_cam is None:
        raise RuntimeError("Cube not visible in camera (semantic+depth). Check FOV, semantic tags, --enable_cameras.")

    print(
        f"[INFO] cube center (pelvis frame, from camera) = "
        f"[{cube_pos_pelvis_cam[0]:+.3f}, {cube_pos_pelvis_cam[1]:+.3f}, {cube_pos_pelvis_cam[2]:+.3f}] m"
    )

    # Joint indices (by name)
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    joint_names = list(robot.data.joint_names)

    def idx_of_joint(name: str) -> int:
        if name not in joint_names:
            close = [jn for jn in joint_names if name.split("_joint")[0] in jn][:30]
            raise RuntimeError(f"Joint '{name}' not found. Partial matches: {close}")
        return joint_names.index(name)

    ik17_indices = [idx_of_joint(n) for n in IK17_JOINT_NAMES]
    ik29_indices = [idx_of_joint(n) for n in IK29_JOINT_NAMES]
    hand12_indices = [idx_of_joint(n) for n in HAND12_NAMES]

    # Initial pose for all joints
    q_init = robot.data.joint_pos.clone()
    num_envs, num_joints = q_init.shape

    ########### gravity compensation helper ##################
    def get_gravity_comp_torque(robot, num_envs: int, num_joints: int, device, dtype) -> torch.Tensor:
        """
        Returns gravity-compensation torques shaped (num_envs, num_joints).

        Uses PhysX view API:
        - get_gravity_compensation_forces()  -> compensation torques (ADD to command)
        Handles floating-base case where the tensor may include 6 root dofs.
        """
        view = getattr(robot, "root_physx_view", None)
        if view is None:
            view = getattr(robot, "_root_physx_view", None)

        if view is None:
            return torch.zeros((num_envs, num_joints), device=device, dtype=dtype)

        if hasattr(view, "get_gravity_compensation_forces"):
            g = view.get_gravity_compensation_forces()
        else:
            return torch.zeros((num_envs, num_joints), device=device, dtype=dtype)

        g = torch.as_tensor(g, device=device, dtype=dtype).view(num_envs, -1)

        # Floating base often returns (6 + num_joints) where first 6 are root DOFs.
        if g.shape[1] == num_joints + 6:
            g = g[:, 6:]
        elif g.shape[1] != num_joints:
            # Try a last-resort trim
            if g.shape[1] > num_joints:
                g = g[:, -num_joints:]
            else:
                return torch.zeros((num_envs, num_joints), device=device, dtype=dtype)
        return g

    # -------------------------------------------------------------------------
    # Build 10 segments (EDIT THIS FUNCTION FOR YOUR TASK)
    # -------------------------------------------------------------------------
    seg_T = [float(getattr(args, f"t_seg_{k}")) for k in range(1, NUM_SEGS + 1)]

    def make_segments() -> List[SegmentSpec]:
        ######## segment helpers ##########
        cx, cy, cz = cube_pos_pelvis_cam.tolist()
        hy = BOX_LEN_Y * 0.5
        clearance_y_l = 0.1
        clearance_y_r = 0.08
        clearance_y_r_2 = 0.05
        clearance_z_1 = 0.25
        clearance_z_2 = 0.05
        box_y = 0.3
        box_x = 0.0
        handPalmX = 0.03

        p_init_l = np.array([0.24, 0.15, 0.09], dtype=float)
        p_init_r = np.array([0.24, -0.15, 0.09], dtype=float)
        # targets
        pL_1 = np.array([cx-handPalmX, cy + (hy + clearance_y_l*2), cz], dtype=float)
        pR_1 = np.array([cx-handPalmX, cy - (hy + clearance_y_r), cz - clearance_z_2], dtype=float)

        pL_2 = np.array([cx-handPalmX, cy + (hy + clearance_y_l*2), cz], dtype=float)
        pR_2 = np.array([cx-handPalmX, cy - (hy + clearance_y_r_2), cz - clearance_z_2], dtype=float)

        pL_4 = np.array([cx-handPalmX, cy + (hy + clearance_y_l*2), cz], dtype=float)
        pR_4 = np.array([cx-handPalmX, cy - (hy + clearance_y_r_2), cz + clearance_z_1 ], dtype=float)

        pL_5 = np.array([cx-handPalmX, cy + (hy + clearance_y_l*2), cz], dtype=float)
        pR_5 = np.array([cx-handPalmX+box_x, cy - (hy + box_y), cz + clearance_z_1 ], dtype=float)

        pL = np.array([cx, cy + (hy + clearance_y_l), cz], dtype=float)
        pR = np.array([cx-handPalmX, cy - (hy + clearance_y_r_2), cz + clearance_z_1], dtype=float)

        # Fingers open/neutral targets (position control segments)
        fingers_zero = {
            "L_thumb_proximal_yaw_joint": 1.7,
            "L_thumb_proximal_pitch_joint": 1.7,
            "L_index_proximal_joint": 0.0,
            "L_middle_proximal_joint": 0.0,
            "L_ring_proximal_joint": 0.0,
            "L_pinky_proximal_joint": 0.0,
            "R_thumb_proximal_yaw_joint": 1.7,
            "R_thumb_proximal_pitch_joint": 0.0,
            "R_index_proximal_joint": 0.0,
            "R_middle_proximal_joint": 0.0,
            "R_ring_proximal_joint": 0.0,
            "R_pinky_proximal_joint": 0.0,
        }

        segs: List[SegmentSpec] = []

        RR_45 = np.array([
            [ np.sqrt(2)/2, -np.sqrt(2)/2, 0.0],
            [ np.sqrt(2)/2,  np.sqrt(2)/2, 0.0],
            [ 0.0,           0.0,          1.0],
        ])

        # ######### segment 0 ###########
        # segs.append(
        #     SegmentSpec(
        #         name="init",
        #         T=30,
        #         hands_pL=p_init_l,
        #         hands_pR=p_init_r,
        #         RL_target=np.eye(3),
        #         RR_target=np.eye(3),
        #         joint_pos_overrides=fingers_zero,
        #         finger_tau_control=False,
        #     )
        # )

        ######### segment 1 ###########
        segs.append(
            SegmentSpec(
                name="S1_pregrasp_fingers_zero",
                T=seg_T[0],
                hands_pL=pL_1,
                hands_pR=pR_1,
                RL_target=np.eye(3),
                RR_target=RR_45,
                joint_pos_overrides=fingers_zero,
            )
        )

        ######### segment 2 ###########
        segs.append(
            SegmentSpec(
                name="S2_close_to_target",
                T=seg_T[1],
                hands_pL=pL_2,
                hands_pR=pR_2,
                RL_target=np.eye(3),
                RR_target=RR_45,
                joint_pos_overrides=fingers_zero,
            )
        )

        # -----------------------------------------------------------------
        # From here on: NO more "tau->q" mapping.
        # We enable closed-loop finger torque control for grasp/hold segments.
        # Keep thumb yaw in position (optional) to maintain thumb orientation.
        # -----------------------------------------------------------------

        ######### segment 3 ###########
        segs.append(
            SegmentSpec(
                name="S3_grasp_target_tau_control",
                T=seg_T[2],
                hands_pL=pL_2,
                hands_pR=pR_2,
                RL_target=np.eye(3),
                RR_target=RR_45,
                joint_pos_overrides={
                    "L_thumb_proximal_yaw_joint": 1.7,
                    "R_thumb_proximal_yaw_joint": 1.7,
                },
                finger_tau_control=True,
                finger_tau_total=float(args.finger_tau_total),
                thumb_alpha=float(args.thumb_alpha),
            )
        )

        ######### segment 4 ###########
        segs.append(
            SegmentSpec(
                name="S4_lift_target_tau_control",
                T=seg_T[3],
                hands_pL=pL_4,
                hands_pR=pR_4,
                RL_target=np.eye(3),
                RR_target=np.eye(3),
                joint_pos_overrides={
                    "L_thumb_proximal_yaw_joint": 1.7,
                    "R_thumb_proximal_yaw_joint": 1.7,
                },
                finger_tau_control=True,
                finger_tau_total=float(args.finger_tau_total),
                thumb_alpha=float(args.thumb_alpha),
            )
        )

        ######### segment 5 ###########
        segs.append(
            SegmentSpec(
                name="S5_move_in_air_tau_control",
                T=seg_T[4],
                hands_pL=pL_5,
                hands_pR=pR_5,
                RL_target=np.eye(3),
                RR_target=np.eye(3),
                joint_pos_overrides={
                    "L_thumb_proximal_yaw_joint": 1.7,
                    "R_thumb_proximal_yaw_joint": 1.7,
                },
                finger_tau_control=True,
                finger_tau_total=float(args.finger_tau_total),
                thumb_alpha=float(args.thumb_alpha),
            )
        )

        ######### segment 6 ###########
        segs.append(
            SegmentSpec(
                name="S6_move_in_air_tau_control",
                T=seg_T[5],
                hands_pL=pL_5,
                hands_pR=pR_5,
                RL_target=np.eye(3),
                RR_target=np.eye(3),
                joint_pos_overrides=fingers_zero,
                finger_tau_control=False,
            )
        )

        ######### segment 7 ###########
        segs.append(
            SegmentSpec(
                name="S7_drop_it",
                T=seg_T[6],
                hands_pL=pL_5,
                hands_pR=pR_5,
                RL_target=np.eye(3),
                RR_target=np.eye(3),
                joint_pos_overrides=fingers_zero,
                finger_tau_control=False,
            )
        )

        ######### segment 8-10 ###########
        segs.append(SegmentSpec(name="S8_dummy_same_target",  T=seg_T[7], hands_pL=pL, hands_pR=pR, joint_pos_overrides=fingers_zero))
        segs.append(SegmentSpec(name="S9_dummy_same_target",  T=seg_T[8], hands_pL=pL, hands_pR=pR, joint_pos_overrides=fingers_zero))
        segs.append(SegmentSpec(name="S10_dummy_same_target", T=seg_T[9], hands_pL=pL, hands_pR=pR, joint_pos_overrides=fingers_zero))

        return segs

    segments = make_segments()
    if len(segments) != NUM_SEGS:
        raise RuntimeError(f"Expected {NUM_SEGS} segments, got {len(segments)}.")

    total_T = sum(s.T for s in segments)
    if total_T <= 0.0:
        raise RuntimeError("Total segment time is <= 0. Check --t_seg_k values.")

    def get_segment_index(t_in: float) -> tuple[int, str]:
        cum = 0.0
        for i, s in enumerate(segments):
            if t_in < cum + s.T:
                return i, s.name
            cum += s.T
        return len(segments) - 1, segments[-1].name

    # -------------------------------------------------------------------------
    # Builders for desired state and torque overrides
    # -------------------------------------------------------------------------
    def apply_q17_into_full(q_full: torch.Tensor, q17: np.ndarray) -> torch.Tensor:
        q = q_full.clone()
        q17 = np.asarray(q17, dtype=float).reshape(17)
        vals = torch.tensor(q17, device=device, dtype=q.dtype)
        idxs = torch.tensor(ik17_indices, device=device, dtype=torch.long)
        q[:, idxs] = vals
        return q

    def apply_q29_into_full(q_full: torch.Tensor, q29: np.ndarray) -> torch.Tensor:
        q = q_full.clone()
        q29 = np.asarray(q29, dtype=float).reshape(29)
        vals = torch.tensor(q29, device=device, dtype=q.dtype)
        idxs = torch.tensor(ik29_indices, device=device, dtype=torch.long)
        q[:, idxs] = vals
        return q

    def apply_pos_overrides(q_full: torch.Tensor, overrides: Optional[Dict[str, float]]) -> torch.Tensor:
        if not overrides:
            return q_full
        q = q_full.clone()
        for jn, val in overrides.items():
            q[:, idx_of_joint(jn)] = float(val)
        return q

    def build_tau_override_tensor(tau_dict: Optional[Dict[str, float]]) -> torch.Tensor:
        tau = torch.zeros((num_envs, num_joints), device=device, dtype=dtype)
        if not tau_dict:
            return tau
        for jn, val in tau_dict.items():
            tau[:, idx_of_joint(jn)] = float(val)
        return tau

    # -------------------------------------------------------------------------
    # PID gains & masks
    # -------------------------------------------------------------------------
    waist_arm_mask = torch.zeros((num_joints,), device=device, dtype=dtype)
    waist_arm_mask[torch.tensor(ik17_indices, device=device)] = 1.0
    waist_arm_mask = waist_arm_mask.unsqueeze(0).expand(num_envs, -1)

    finger_mask = torch.zeros((num_joints,), device=device, dtype=dtype)
    finger_mask[torch.tensor(hand12_indices, device=device)] = 1.0
    finger_mask = finger_mask.unsqueeze(0).expand(num_envs, -1)

    THUMB_YAW_JOINTS = [
        "L_thumb_proximal_yaw_joint",
        "R_thumb_proximal_yaw_joint",
    ]
    thumb_yaw_indices = [idx_of_joint(n) for n in THUMB_YAW_JOINTS]

    thumb_yaw_mask = torch.zeros((num_joints,), device=device, dtype=dtype)
    thumb_yaw_mask[torch.tensor(thumb_yaw_indices, device=device)] = 1.0
    thumb_yaw_mask = thumb_yaw_mask.unsqueeze(0).expand(num_envs, -1)

    # Gains for thumb yaw holding (reuse finger gains or define dedicated ones)
    kp_thumb_yaw = torch.full((num_envs, num_joints), float(args.kp_finger), device=device, dtype=dtype) * thumb_yaw_mask
    kd_thumb_yaw = torch.full((num_envs, num_joints), float(args.kd_finger), device=device, dtype=dtype) * thumb_yaw_mask
    ki_thumb_yaw = torch.full((num_envs, num_joints), float(args.ki_finger), device=device, dtype=dtype) * thumb_yaw_mask

    finger_mask = finger_mask * (1.0 - thumb_yaw_mask)


    kp_base = torch.full((num_envs, num_joints), float(args.kp), device=device, dtype=dtype) * waist_arm_mask
    kd_base = torch.full((num_envs, num_joints), float(args.kd), device=device, dtype=dtype) * waist_arm_mask
    ki_base = torch.full((num_envs, num_joints), float(args.ki), device=device, dtype=dtype) * waist_arm_mask

    kp_f = torch.full((num_envs, num_joints), float(args.kp_finger), device=device, dtype=dtype) * finger_mask
    kd_f = torch.full((num_envs, num_joints), float(args.kd_finger), device=device, dtype=dtype) * finger_mask
    ki_f = torch.full((num_envs, num_joints), float(args.ki_finger), device=device, dtype=dtype) * finger_mask

    e_int = torch.zeros((num_envs, num_joints), device=device, dtype=dtype)
    i_clamp = float(args.i_clamp)

    # -------------------------------------------------------------------------
    # Finger torque-control joints + mask (thumb pitch + 4 fingers per hand)
    # -------------------------------------------------------------------------
    FINGER_TAU_JOINTS = [
        "L_thumb_proximal_pitch_joint",
        "L_index_proximal_joint",
        "L_middle_proximal_joint",
        "L_ring_proximal_joint",
        "L_pinky_proximal_joint",
        "R_thumb_proximal_pitch_joint",
        "R_index_proximal_joint",
        "R_middle_proximal_joint",
        "R_ring_proximal_joint",
        "R_pinky_proximal_joint",
    ]
    finger_tau_indices = [idx_of_joint(n) for n in FINGER_TAU_JOINTS]
    finger_tau_mask = torch.zeros((num_joints,), device=device, dtype=dtype)
    finger_tau_mask[torch.tensor(finger_tau_indices, device=device)] = 1.0
    finger_tau_mask = finger_tau_mask.unsqueeze(0).expand(num_envs, -1)

    # Finger torque integrator
    e_tau_int = torch.zeros((num_envs, num_joints), device=device, dtype=dtype)
    tau_i_clamp = float(args.finger_tau_i_clamp)

    # Torque limits
    effort_abs_lim = get_effort_abs_limits(robot, device, dtype, num_envs, num_joints)
    if effort_abs_lim is not None:
        effort_abs_lim = effort_abs_lim * float(args.effort_limit_scale)

    manual_tau_clamp = float(args.tau_clamp)
    if manual_tau_clamp > 0.0:
        manual_lim = torch.full((num_envs, num_joints), manual_tau_clamp, device=device, dtype=dtype)
        effort_abs_lim = manual_lim if effort_abs_lim is None else torch.minimum(effort_abs_lim, manual_lim)

    print("\n[INFO] Segment engine ready.")
    print("[INFO] Waist+arms always PID position.")
    print("[INFO] Fingers: either position PID (segments with finger pos targets) OR closed-loop torque control (segments with finger_tau_control=True).")

    # -------------------------------------------------------------------------
    # CSV logger (log all 29 joints)
    # -------------------------------------------------------------------------
    LOG_JOINTS = IK29_JOINT_NAMES  # 29 = waist+arms+prox hand
    log_indices = [idx_of_joint(n) for n in LOG_JOINTS]

    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.log_dir, f"{args.log_prefix}_{stamp}_q_tau_29.csv")

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = (
        ["time_s"]
        + [n + "_pos" for n in LOG_JOINTS]
        + [n + "_qdes" for n in LOG_JOINTS]
        + [n + "_tau" for n in LOG_JOINTS]
    )
    csv_writer.writerow(header)
    csv_file.flush()
    print(f"[INFO] Streaming 29-joint pos+torque CSV to: {csv_path}")

    # -------------------------------------------------------------------------
    # Runtime state
    # -------------------------------------------------------------------------
    dt = sim.get_physics_dt()
    step_idx = 0
    sleep_dt = 1.0 / max(1.0, args.max_fps)

    active_seg = -1
    q_des_active = q_init.clone()
    tau_override_active = torch.zeros((num_envs, num_joints), device=device, dtype=dtype)
    finger_pos_enabled = False
    finger_tau_enabled = False 
    prev_finger_tau_enabled = False


    # warm-start buffers
    q17_prev = np.zeros(17, dtype=float)
    q29_prev = np.zeros(29, dtype=float)

    # default orientation targets (identity)
    RL_default = np.eye(3)
    RR_default = np.eye(3)
    dq_max = float(args.ik_dq_max_deg) * math.pi / 180.0

    try:
        while True:
            t = step_idx * dt
            t_in = t % total_T

            seg_i, seg_name = get_segment_index(t_in)
            seg = segments[seg_i]

            # --------------------------
            # Segment change: compute q_des once
            # --------------------------
            if seg_i != active_seg:
                active_seg = seg_i
                e_int.zero_()
                finger_tau_enabled = bool(getattr(seg, "finger_tau_control", False))

                # Only reset torque integrator on transitions (off->on or on->off)
                if finger_tau_enabled != prev_finger_tau_enabled:
                    e_tau_int.zero_()

                prev_finger_tau_enabled = finger_tau_enabled

                # start from current
                q_des = robot.data.joint_pos.clone()

                # (A) ik_hands: compute 17D from wrist targets
                if seg.hands_pL is not None and seg.hands_pR is not None:
                    RL_seg = seg.RL_target if seg.RL_target is not None else RL_default
                    RR_seg = seg.RR_target if seg.RR_target is not None else RR_default

                    q_sol17, info = ik_two_hands_dls(
                        pL_target=np.asarray(seg.hands_pL, dtype=float).reshape(3),
                        pR_target=np.asarray(seg.hands_pR, dtype=float).reshape(3),
                        RL_target=np.asarray(RL_seg, dtype=float).reshape(3, 3),
                        RR_target=np.asarray(RR_seg, dtype=float).reshape(3, 3),
                        use_waist=bool(args.use_waist),
                        q0=q17_prev,  # warm-start
                        max_iters=int(args.ik_max_iters),
                        lamb=float(args.ik_lambda),
                        alpha=float(args.ik_alpha),
                        tol_p=float(args.ik_tol_p),
                        tol_w=float(args.ik_tol_w),
                        dq_max=dq_max,
                        w_pos=1.0,
                        w_rot=1.0,
                    )
                    q17_prev = np.asarray(q_sol17, dtype=float).reshape(17)
                    q_des = apply_q17_into_full(q_des, q17_prev)

                # (B) fingertip IK: compute 29D from fingertip targets
                if seg.fingertip_targets is not None:
                    q_sol29, info29 = ik_two_hands_with_fingers_dls(
                        targets=seg.fingertip_targets,
                        q0=q29_prev,
                        use_waist=bool(args.use_waist),
                        max_iters=int(args.ik_max_iters),
                        lamb=float(args.ik_lambda),
                        alpha=float(args.ik_alpha),
                        tol_p=float(args.ik_tol_p),
                        tol_w=float(args.ik_tol_w),
                        dq_max=dq_max,
                        w_pos=1.0,
                        w_rot=1.0,
                    )
                    q29_prev = np.asarray(q_sol29, dtype=float).reshape(29)
                    q_des = apply_q29_into_full(q_des, q29_prev)

                # Apply explicit position overrides (wins over IK)
                q_des = apply_pos_overrides(q_des, seg.joint_pos_overrides)

                # Torque overrides tensor
                tau_override_active = build_tau_override_tensor(seg.joint_tau_overrides)

                # Determine if finger position control should be enabled this segment:
                finger_pos_enabled = False
                if seg.fingertip_targets is not None:
                    finger_pos_enabled = True
                if seg.joint_pos_overrides is not None:
                    if any(jn in seg.joint_pos_overrides for jn in HAND12_NAMES):
                        finger_pos_enabled = True

                # If torque-control is enabled, disable finger position PID to avoid fighting
                if finger_tau_enabled:
                    finger_pos_enabled = False

                q_des_active = q_des
                print(f"[SEG] -> {seg_name} | finger_pos_enabled={finger_pos_enabled} | finger_tau_enabled={finger_tau_enabled}")

            # --------------------------
            # PID compute
            # --------------------------
            q = robot.data.joint_pos
            q_dot = robot.data.joint_vel
            q_dot_des = torch.zeros_like(q_dot)

            e = q_des_active - q
            e_dot = q_dot_des - q_dot

            kp_eff = kp_base + (kp_f if finger_pos_enabled else 0.0) + kp_thumb_yaw
            kd_eff = kd_base + (kd_f if finger_pos_enabled else 0.0) + kd_thumb_yaw
            ki_eff = ki_base + (ki_f if finger_pos_enabled else 0.0) + ki_thumb_yaw


            any_ki_active = (
                (args.ki != 0.0)
                or (finger_pos_enabled and args.ki_finger != 0.0)
                or (args.ki_finger != 0.0)  # for thumb yaw hold if you want integral there
            )
            if any_ki_active:
                e_int = e_int + e * dt
                e_int = torch.clamp(e_int, -i_clamp, i_clamp)

            tau_pid = kp_eff * e + kd_eff * e_dot + ki_eff * e_int

            # Gravity-compensation torques (feedforward)
            tau_g = get_gravity_comp_torque(robot, num_envs, num_joints, device, dtype)

            # Base command torque
            tau = tau_pid + tau_g

            # ------------------------------------------------------------
            # Closed-loop finger torque control
            # ------------------------------------------------------------
            if finger_tau_enabled:
                # measured torques
                if hasattr(robot.data, "applied_torque"):
                    tau_meas = robot.data.applied_torque
                elif hasattr(robot.data, "joint_torques"):
                    tau_meas = robot.data.joint_torques
                else:
                    # fallback (keeps code running, but torque loop won't be meaningful)
                    tau_meas = tau

                tau_total = float(seg.finger_tau_total) if seg.finger_tau_total is not None else float(args.finger_tau_total)
                alpha = float(seg.thumb_alpha) if seg.thumb_alpha is not None else float(args.thumb_alpha)

                tau_thumb = alpha * tau_total
                tau_4 = (1.0 - alpha) * tau_total / 4.0

                tau_des = torch.zeros((num_envs, num_joints), device=device, dtype=dtype)

                # left hand desired torques
                tau_des[:, idx_of_joint("L_thumb_proximal_pitch_joint")] = tau_thumb
                tau_des[:, idx_of_joint("L_index_proximal_joint")] = tau_4
                tau_des[:, idx_of_joint("L_middle_proximal_joint")] = tau_4
                tau_des[:, idx_of_joint("L_ring_proximal_joint")] = tau_4
                tau_des[:, idx_of_joint("L_pinky_proximal_joint")] = tau_4

                # right hand desired torques
                tau_des[:, idx_of_joint("R_thumb_proximal_pitch_joint")] = tau_thumb
                tau_des[:, idx_of_joint("R_index_proximal_joint")] = tau_4
                tau_des[:, idx_of_joint("R_middle_proximal_joint")] = tau_4
                tau_des[:, idx_of_joint("R_ring_proximal_joint")] = tau_4
                tau_des[:, idx_of_joint("R_pinky_proximal_joint")] = tau_4

                # PI on torque error (only for finger torque joints)
                e_tau = (tau_des - tau_meas) * finger_tau_mask
                Kp_tau = float(args.finger_tau_kp)
                Ki_tau = float(args.finger_tau_ki)

                if Ki_tau != 0.0:
                    e_tau_int = e_tau_int + e_tau * dt
                    e_tau_int = torch.clamp(e_tau_int, -tau_i_clamp, tau_i_clamp)

                tau_cmd_fingers = Kp_tau * e_tau + Ki_tau * e_tau_int

                # overwrite finger torques with torque-controller output
                tau = torch.where(finger_tau_mask > 0.0, tau_cmd_fingers, tau)

            # Apply torque overrides: overwrite tau where override != 0
            if seg.joint_tau_overrides:
                override_mask = (tau_override_active != 0.0)
                tau = torch.where(override_mask, tau_override_active, tau)

            # Clamp
            if effort_abs_lim is not None:
                tau = torch.clamp(tau, -effort_abs_lim, effort_abs_lim)

            robot.set_joint_effort_target(tau)

            # Step sim
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            step_idx += 1

            # --------------------------
            # CSV logging (29 joints)
            # --------------------------
            if (step_idx % int(args.log_every)) == 0:
                joint_pos_all = robot.data.joint_pos[0]
                q_des_all = q_des_active[0]

                if hasattr(robot.data, "applied_torque"):
                    tau_all = robot.data.applied_torque[0]
                elif hasattr(robot.data, "joint_torques"):
                    tau_all = robot.data.joint_torques[0]
                else:
                    tau_all = None

                if tau_all is not None:
                    pos_29   = joint_pos_all[log_indices].detach().cpu().numpy().astype(float)
                    qdes_29  = q_des_all[log_indices].detach().cpu().numpy().astype(float)
                    tau_29   = tau_all[log_indices].detach().cpu().numpy().astype(float)

                    row = (
                        [float(step_idx * dt)]
                        + pos_29.tolist()
                        + qdes_29.tolist()
                        + tau_29.tolist()
                    )
                    csv_writer.writerow(row)
                    csv_file.flush()

            # prints
            if step_idx % args.print_every == 0:
                print(f"[{seg_name:18s}] step={step_idx:06d} t={t:7.3f}s")

                cube_pos_w = scene["cube"].data.root_pos_w[0, :3]
                pelvis_pos_w = robot.data.root_link_pos_w[0]
                pelvis_quat_w = robot.data.root_link_quat_w[0]
                dp_w = (cube_pos_w - pelvis_pos_w).unsqueeze(0)
                cube_pos_pelvis_gt = math_utils.quat_apply_inverse(pelvis_quat_w.unsqueeze(0), dp_w)[0].cpu().numpy()
                print(
                    f"cube center (pelvis frame, GT) = "
                    f"[{cube_pos_pelvis_gt[0]:+.3f}, {cube_pos_pelvis_gt[1]:+.3f}, {cube_pos_pelvis_gt[2]:+.3f}] m"
                )

            if sleep_dt > 0:
                time.sleep(sleep_dt)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        try:
            csv_file.flush()
            csv_file.close()
            print(f"[INFO] Closed CSV: {csv_path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
    simulation_app.close()
