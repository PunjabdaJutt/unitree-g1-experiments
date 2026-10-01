# mujoco_common.py
"""Shared MuJoCo model-loading/fidelity glue for the G1+Inspire-hand DFQ URDF.

Used by both `mujoco_ph_harness.py` (finger step-response diagnostic) and
`mujoco_lift_finger.py` (full pick-and-place scene). Physical-fidelity
numbers below are transcribed from `scripts/g1_control/robot/unitree.py`, the
Isaac reference `ArticulationCfg` -- see that file for the reasoning behind
each one:
  - armature=0.01 on every joint. Load-bearing: raw URDF link inertia for the
    finger segments is ~1e-7 kg.m^2, and the PH controller's gains were tuned
    assuming armature dominates the effective inertia (~0.01) -- skipping
    this makes the same gains wildly unstable in MuJoCo.
  - Legs (hip x3, knee) and feet (ankle x2) hold their spawn pose via native
    MuJoCo passive joint stiffness/damping (150/5 and 40/4 respectively,
    springref = spawn pose) -- Isaac's real implicit PD for these groups.
    They're dynamically decoupled from the waist/arm/hand subtree (both hang
    off the same fixed pelvis), so this has no effect on finger fidelity.
  - The 12 unactuated finger intermediate/distal ("mimic") joints get
    damping=0.05, zero stiffness -- a pure passive damper, matching Isaac's
    "hand_mimic_passive" actuator group.
  - Waist/arm/HAND12 joints get zero native stiffness/damping -- pure
    external torque every step, exactly like Isaac's set_joint_effort_target
    path.
  - The URDF's root link (pelvis) has no parent joint, so MuJoCo's URDF
    compiler fuses it into the world body automatically -- equivalent to
    Isaac's `fix_root_link=True`. Do not add a freejoint to the robot.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import mujoco

from scripts.g1_control.ph import HAND12_MIMIC_JOINTS

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_URDF = os.path.join(
    REPO_ROOT, "unitree_g1_inspire_assets", "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")

# Fingertip (most distal) link names, HAND12 order -- copied verbatim from
# lift_fingerPH.py (~L756). Thumb shares one distal link for yaw + pitch.
TIP_LINK_NAMES = [
    "L_thumb_distal", "L_thumb_distal",
    "L_index_intermediate", "L_middle_intermediate", "L_ring_intermediate", "L_pinky_intermediate",
    "R_thumb_distal", "R_thumb_distal",
    "R_index_intermediate", "R_middle_intermediate", "R_ring_intermediate", "R_pinky_intermediate",
]
# HAND12 order: L_thumb_yaw(0), L_thumb_pitch(1), L_idx..L_pnk(2-5),
#               R_thumb_yaw(6), R_thumb_pitch(7), R_idx..R_pnk(8-11).
THUMB_YAW_IDX = (0, 6)
THUMB_PITCH_IDX = (1, 7)

ARMATURE = 0.01

# Spawn pose, copied from robot/unitree.py's InitialStateCfg.joint_pos
# (regex keys there expanded to concrete names; everything else defaults 0).
DEFAULT_JOINT_POS: Dict[str, float] = {
    "left_hip_pitch_joint": 0.0, "right_hip_pitch_joint": 0.0,
    "left_knee_joint": 0.0, "right_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0, "right_ankle_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0, "right_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0, "right_shoulder_roll_joint": 0.0,
    "left_elbow_joint": 0.0, "right_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0, "right_wrist_roll_joint": 0.0,
}
LEG_JOINTS = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
]
FOOT_JOINTS = [
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def read_patched_urdf_text(urdf_path: Path) -> str:
    """DFQ URDF mesh filenames are `meshes/xxx.STL` relative to the URDF's
    own directory, but the URDF's own `meshdir="meshes"` compiler hint
    doubles that to `meshes/meshes/xxx.STL` once compiled from a tempfile
    elsewhere -- rewrite meshdir to the URDF's absolute parent directory
    instead so the existing `meshes/...` filenames resolve correctly (same
    tempfile-rewrite trick as ros2_ws/.../mujoco_mirror.py, different
    substitution)."""
    text = urdf_path.read_text()
    return re.sub(r'meshdir="meshes"', f'meshdir="{urdf_path.parent}"', text)


def load_model(urdf_path: Path) -> mujoco.MjModel:
    """Load the DFQ URDF directly as a standalone MjModel (no extra scene
    bodies). Use `load_spec` instead when the caller needs to add scene
    objects (table, graspable object, ...) before compiling."""
    text = read_patched_urdf_text(urdf_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".urdf")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        return mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)


def load_spec(urdf_path: Path) -> mujoco.MjSpec:
    """Load the DFQ URDF as an editable MjSpec, so callers can add scene
    bodies (table, graspable object, ...) via `spec.worldbody.add_body(...)`
    before `spec.compile()`."""
    text = read_patched_urdf_text(urdf_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".urdf")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        return mujoco.MjSpec.from_file(tmp_path)
    finally:
        os.unlink(tmp_path)


def joint_addr(model: mujoco.MjModel, name: str):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"joint '{name}' not found in MuJoCo model")
    return jid, int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def apply_fidelity(model: mujoco.MjModel) -> None:
    """Post-load pass replicating the Isaac reference's per-joint-group
    armature/stiffness/damping (see module docstring)."""
    model.dof_armature[:] = ARMATURE
    for n in LEG_JOINTS:
        jid, qadr, dadr = joint_addr(model, n)
        model.jnt_stiffness[jid] = 150.0
        model.dof_damping[dadr] = 5.0
        model.qpos_spring[qadr] = DEFAULT_JOINT_POS.get(n, 0.0)
    for n in FOOT_JOINTS:
        jid, qadr, dadr = joint_addr(model, n)
        model.jnt_stiffness[jid] = 40.0
        model.dof_damping[dadr] = 4.0
        model.qpos_spring[qadr] = 0.0
    for n in HAND12_MIMIC_JOINTS:
        _, _, dadr = joint_addr(model, n)
        model.dof_damping[dadr] = 0.05
    # waist/arms/wrists/HAND12: left at the URDF-import default (0.0 native
    # stiffness/damping, confirmed -- the URDF has no <dynamics> tags at all),
    # matching Isaac's zero-PD "pure external torque" actuator groups.


def hinge_joint_ids(model: mujoco.MjModel) -> np.ndarray:
    """All robot joints are single-DOF hinges (confirmed: 53 revolute, 0
    free/ball in the DFQ URDF). Scenes built via `load_spec` may add extra
    bodies (e.g. a free-floating graspable object) with free/ball joints
    that don't fit the one-name-per-scalar-DOF assumption `sim_joint_names`
    needs -- filter those out so `joint_name_list`/callers only ever see the
    robot's own joints, regardless of what else is in the scene."""
    return np.array([i for i in range(model.njnt)
                      if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE])


def joint_name_list(model: mujoco.MjModel) -> List[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in hinge_joint_ids(model)]


def joint_addrs(model: mujoco.MjModel):
    """(names, qposadr, dofadr) for the robot's hinge joints only, in the
    same order as `joint_name_list` -- use these instead of
    `model.jnt_qposadr`/`model.jnt_dofadr` directly whenever the scene may
    contain non-robot bodies."""
    ids = hinge_joint_ids(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in ids]
    return names, model.jnt_qposadr[ids].copy(), model.jnt_dofadr[ids].copy()


# -----------------------------------------------------------------------------
# Camera-based object detection
#
# Mirrors lift_fingerPH.py's cube_centroid_in_camera_frame() +
# cube_cam_to_pelvis_from_torso(), but simpler on both counts:
#   - Segmentation here returns MuJoCo's own geom id directly (we pick that id
#     ourselves when building the scene), so no semantic-class lookup dance.
#   - No separate camera->pelvis transform chain is needed: our MuJoCo scene
#     never applies Isaac's arbitrary spawn transform (world pos (0,0,0.793),
#     +90deg yaw about Z), so the URDF's fixed root fuses to MuJoCo's world
#     body at identity -- our world frame already IS pelvis frame. Points
#     recovered in MuJoCo world coordinates are therefore already in the
#     frame ik_two_hands_dls/fk_chain expect, with no extra conversion.
# -----------------------------------------------------------------------------
def detect_object_world_pos(
    renderer: "mujoco.Renderer",
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    target_geom_name: str,
    max_samples: int = 2000,
) -> Optional[np.ndarray]:
    """Render `camera_name`, find `target_geom_name`'s segmented pixels,
    unproject EVERY valid pixel's own depth to a 3D point, and average the
    resulting 3D points. Returns None if the object isn't visible.

    Averaging per-pixel 3D points (not pixel coordinates first) matters once
    the object is partially occluded: a non-convex/split visible silhouette
    (e.g. a hand passing in front of the object) can have an averaged PIXEL
    coordinate that lands in the gap between the visible pieces -- off the
    object entirely -- so reading one depth sample there gives a garbage
    point. This showed up as the detected object position jumping several cm
    between re-detections right as a hand approached it, which then fed
    straight into segment IK targets as bogus lift heights. Mirrors
    lift_fingerPH.py's cube_centroid_in_camera_frame(), which does the same
    per-pixel-then-average order for the same reason -- the original version
    of this function here didn't, which was the bug."""
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_geom_name)
    if target_id < 0:
        raise KeyError(f"geom '{target_geom_name}' not found in MuJoCo model")

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name)
    seg = renderer.render()
    mask = (seg[..., 0] == target_id) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    renderer.disable_segmentation_rendering()
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    if ys.size > max_samples:
        sel = np.random.default_rng().choice(ys.size, size=max_samples, replace=False)
        ys, xs = ys[sel], xs[sel]

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render()
    renderer.disable_depth_rendering()
    d = depth[ys, xs].astype(float)
    valid = np.isfinite(d) & (d > 0.0)
    if not valid.any():
        return None
    ys, xs, d = ys[valid], xs[valid], d[valid]

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    fovy = float(model.cam_fovy[cam_id])
    height, width = renderer.height, renderer.width
    fy = height / (2.0 * np.tan(np.radians(fovy) / 2.0))
    fx = fy
    px_cx, px_cy = width / 2.0, height / 2.0

    # Pinhole unprojection in OpenCV-style camera coords (x-right, y-down,
    # z-forward) for every valid pixel at once, then converted to MuJoCo
    # camera-local axes (x-right, y-up, z-backward -- cameras look down
    # local -Z) before rotating into world.
    X = (xs.astype(float) - px_cx) / fx * d
    Y = (ys.astype(float) - px_cy) / fy * d
    p_cam_cv = np.stack([X, Y, d], axis=-1)                       # (N,3)
    p_cam_mj = np.stack([p_cam_cv[:, 0], -p_cam_cv[:, 1], -p_cam_cv[:, 2]], axis=-1)

    cam_pos_w = data.cam_xpos[cam_id]
    cam_mat_w = data.cam_xmat[cam_id].reshape(3, 3)
    p_world = cam_pos_w + p_cam_mj @ cam_mat_w.T                  # (N,3)
    return p_world.mean(axis=0)
