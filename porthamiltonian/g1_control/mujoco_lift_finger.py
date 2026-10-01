# mujoco_lift_finger.py
"""MuJoCo port of lift_fingerPH.py: pick a cylinder off a table and place it
in a box, using two-hand IK + waist/arm PID for reaching and the PH IDA-PBC
controller (ph/ph_impedance.PHContactForceController) for the HAND12 finger
joints. Segment choreography (S1-pregrasp .. S10-idle), IK targets, and the
PH position/force mode + gain-schedule switching are ported line-for-line
from `lift_fingerPH.py`'s `make_segments()` and its main loop's
segment-transition block -- see those for the original.

Camera-based object detection (not ground truth) drives the reference
trajectory, mirroring the IsaacLab reference's RealSense pipeline: a virtual
camera mounted on torso_link is rendered every CAM_UPDATE_EVERY steps,
segmentation finds the cube's pixels, depth unprojects its 3D position.
Two things simplify this relative to Isaac:
  - Segmentation returns MuJoCo's own geom id directly (no semantic-class
    lookup needed -- see mujoco_common.detect_object_world_pos).
  - No camera->pelvis transform chain is needed: this MuJoCo scene never
    applies Isaac's arbitrary spawn transform (world pos (0,0,0.793), +90deg
    yaw about Z) -- the URDF's fixed root fuses to MuJoCo's world body at
    identity, so MuJoCo world frame already IS pelvis frame. Scene objects
    below are placed using Isaac's world-frame constants converted through
    that same transform (pelvis_x=world_y, pelvis_y=-world_x,
    pelvis_z=world_z-0.793), confirmed correct by a validation render this
    session (recovered position within ~2-3cm of the placed object).

Run:
  python scripts/g1_control/mujoco_lift_finger.py
  python scripts/g1_control/mujoco_lift_finger.py --headless --cycles 1
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import mujoco

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.g1_control.ph import (
    parse_urdf, build_tree, PHSystem,
    FingertipJacobian, PHContactForceController, JointPDIController,
    IK17_JOINT_NAMES, HAND12_NAMES,
    ik_two_hands_dls, fk_fingers, ik_fingers_analytical, ik_fingers_from_curl,
)
from scripts.g1_control.mujoco_common import (
    DEFAULT_URDF, TIP_LINK_NAMES, THUMB_YAW_IDX, THUMB_PITCH_IDX, ARMATURE,
    DEFAULT_JOINT_POS, load_spec, apply_fidelity, joint_addrs,
    detect_object_world_pos,
)


# -----------------------------------------------------------------------------
# Scene constants (from lift_fingerPH.py's SceneCfg / module constants).
# World-frame values are Isaac's; pelvis-frame values (what this script
# places objects at, since MuJoCo world == pelvis frame here) are derived via
# pelvis_x=world_y, pelvis_y=-world_x, pelvis_z=world_z-0.793 -- Isaac's own
# spawn transform, spelled out in lift_fingerPH.py's camera-fallback comment.
# -----------------------------------------------------------------------------
_SPAWN_Z = 0.793


def _world_to_pelvis(wx: float, wy: float, wz: float) -> np.ndarray:
    return np.array([wy, -wx, wz - _SPAWN_Z])


TABLE_Z = 0.80
BOX_LEN_X, BOX_LEN_Y, BOX_LEN_Z = 0.03, 0.03, 0.10
BOX_HALF_Z = BOX_LEN_Z * 0.5
CUBE_MASS = 0.05
CYL_RADIUS = BOX_LEN_Y * 0.84

TABLE_POS_PELVIS = _world_to_pelvis(0.0, 1.2, TABLE_Z)
CUBE_POS_PELVIS = _world_to_pelvis(0.05, 0.5, TABLE_Z + BOX_HALF_Z + 0.07)
BOX_POS_PELVIS = _world_to_pelvis(0.4, 0.45, 0.85)

# d435 mount offset from torso_link, copied from lift_fingerPH.py's
# cube_cam_to_pelvis_from_torso default args (d435_xyz). Aimed via an
# explicit look-direction vector at a point over the table (see
# mujoco_common's camera-pipeline notes for why: hand-deriving the
# URDF-rpy-vs-MuJoCo-euler convention for this mount failed in testing).
_CAM_POS_LOCAL = np.array([0.0576235, 0.01753, 0.42987])
_CAM_TARGET_LOCAL = np.array([TABLE_POS_PELVIS[0], TABLE_POS_PELVIS[1] * 0.3, TABLE_POS_PELVIS[2] - 0.35])
CAM_FOVY = 58.0
CAM_WIDTH, CAM_HEIGHT = 320, 240

CAM_UPDATE_EVERY = 30  # physics steps between re-detections, matches reference


# -----------------------------------------------------------------------------
# Segment spec (verbatim structure from lift_fingerPH.py)
# -----------------------------------------------------------------------------
@dataclass
class SegmentSpec:
    name: str
    T: float
    hands_pL: Optional[np.ndarray] = None
    hands_pR: Optional[np.ndarray] = None
    use_grasp_ik: bool = False
    finger_curl_L: Optional[float] = 0.0
    finger_curl_R: Optional[float] = 0.0
    joint_pos_overrides: Optional[Dict[str, float]] = None
    RL_target: Optional[np.ndarray] = None
    RR_target: Optional[np.ndarray] = None


def make_segments(cube_pos: np.ndarray, seg_T: List[float]) -> List[SegmentSpec]:
    """Ported verbatim from lift_fingerPH.py's make_segments() -- geometry
    constants, waypoint formulas, and the finger-control legend are unchanged;
    only the cube-position source differs (camera detection here too, same as
    the reference, just without the Isaac-specific transform chain)."""
    cx, cy, cz = cube_pos.tolist()
    hy = BOX_LEN_Y * 0.5

    palmX = 0.04
    palmX_R_s2 = 0.1
    y_wide = 0.06
    y_near = 0.045
    z_abv = -0.10
    z_grasp_bias = -0.10
    z_below = 0.05
    z_lift = 0.20
    y_box = 0.30

    RL_id = np.eye(3)
    RR_id = np.eye(3)
    _c, _s = math.cos(math.radians(10)), math.sin(math.radians(10))
    # NOTE: reference's own RR_face is NOT orthogonal (row [1]'s `_s*1.1`
    # term) -- reproduced exactly, not "fixed", to match the ported behavior.
    RR_face = np.array([
        [_c, -_s, 0.0],
        [_s * 1.1, _c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    pL_pre = np.array([cx - palmX, cy + (hy + y_wide + 0.05), cz + z_abv])
    pR_pre = np.array([cx - palmX + 0.5, cy - (hy + (y_wide - 0.05)), cz - z_below])

    pL_cls = np.array([cx - palmX + 0.01, cy + (hy + y_near+0.1), cz + z_abv])
    pR_cls = np.array([cx - palmX_R_s2, cy - (hy + y_near), cz + z_abv])
    pR_clss = np.array([cx - palmX_R_s2, cy - (hy + y_near), cz + z_abv])

    pL_lft = np.array([cx - palmX, cy + (hy + y_near), cz + z_lift])
    pR_lft = np.array([cx - palmX, cy - (hy + y_near), cz + z_lift])

    pL_box = np.array([cx - palmX, cy + (hy + y_near + 0.3), cz + z_lift])
    pR_box = np.array([cx - palmX, cy - (hy + y_box), cz + z_lift])

    partial_curl = np.array([0.5, 0, 0, 0, 0, 0])

    # Gain schedule (applied in the segment-transition hook, see main loop):
    #   S1, S2, S9, S10 : Kt_free / Bt_free   -- free motion, contact OFF
    #   S3-grasp-PH     : Kt_close / Bt_close -- grasp closure, auto-detect
    #   S4-lift-PH, S5-moveair-PH : Kt_grasp / Bt_grasp -- lift in contact, forced ON
    #   S6-release, S7-drop, S8-idle : Kt_free / Bt_free -- release, contact OFF
    return [
        SegmentSpec("S1-pregrasp", seg_T[0], pL_pre, pR_cls, False, partial_curl, partial_curl,None, RL_id, RR_id),
        SegmentSpec("S2-close-approach", seg_T[1], pL_cls, pR_cls, False, partial_curl, partial_curl, None, RL_id, RR_face),
        SegmentSpec("S3-grasp-PH", seg_T[2], pL_cls, pR_clss, True, None, None, None, RL_id, RR_face),
        SegmentSpec("S4-lift-PH", seg_T[3], pL_cls, pR_lft, False, None, None, None, RL_id, RL_id),
        SegmentSpec("S5-moveair-PH", seg_T[4], pL_cls, pR_box, False, None, None, None, RL_id, RL_id),
        SegmentSpec("S6-release", seg_T[5], pL_cls, pR_box, False, 0.0, 0.0, None, RL_id, RL_id),
        SegmentSpec("S7-drop", seg_T[6], pL_cls, pR_box, False, 0.0, 0.0, None, RL_id, RL_id),
        SegmentSpec("S8-idle", seg_T[7], pL_pre, pL_pre, False, 0.0, 0.0, None, RL_id, RL_id),
        SegmentSpec("S9-idle", seg_T[8], pL_pre, pR_pre, False, 0.0, 0.0, None, RL_id, RR_id),
        SegmentSpec("S10-idle", seg_T[9], pL_pre, pR_pre, False, 0.0, 0.0, None, RL_id, RR_id),
    ]


def get_segment_index(t_in: float, segments: List[SegmentSpec]):
    cum = 0.0
    for i, s in enumerate(segments):
        if t_in < cum + s.T:
            return i, s.name
        cum += s.T
    return len(segments) - 1, segments[-1].name


# -----------------------------------------------------------------------------
# Scene construction
# -----------------------------------------------------------------------------
def build_scene(urdf_path: Path) -> mujoco.MjModel:
    spec = load_spec(urdf_path)
    wb = spec.worldbody
    torso = next(b for b in spec.bodies if b.name == "torso_link")

    forward = _CAM_TARGET_LOCAL - _CAM_POS_LOCAL
    forward = forward / np.linalg.norm(forward)
    torso.add_camera(name="d435", pos=_CAM_POS_LOCAL.tolist(), zaxis=(-forward).tolist(), fovy=CAM_FOVY)

    table = wb.add_body(name="table", pos=TABLE_POS_PELVIS.tolist())
    table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1.0, 0.4, 0.025], rgba=[0.4, 0.4, 0.4, 1])

    cube = wb.add_body(name="cube", pos=CUBE_POS_PELVIS.tolist())
    cube.add_freejoint()
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[CYL_RADIUS, (BOX_LEN_Z + 0.02) / 2.0, 0.0],
                  mass=CUBE_MASS, rgba=[0.0, 0.8, 0.0, 1],
                  friction=[1.5, 0.005, 0.0001])

    # Placement box: open bin approximated as bottom + 4 walls (no KLT mesh
    # asset available in MuJoCo -- exact bin geometry isn't load-bearing for
    # verifying the controller/choreography port, only a plausible target).
    bin_half = 0.12
    bin_wall_h = 0.06
    bx, by, bz = BOX_POS_PELVIS.tolist()
    bin_body = wb.add_body(name="place_box", pos=[bx, by, bz])
    bin_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[bin_half, bin_half, 0.005],
                      rgba=[0.55, 0.35, 0.15, 1])
    for dx, dy, sx, sy in [
        (bin_half, 0.0, 0.005, bin_half), (-bin_half, 0.0, 0.005, bin_half),
        (0.0, bin_half, bin_half, 0.005), (0.0, -bin_half, bin_half, 0.005),
    ]:
        bin_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[dx, dy, bin_wall_h / 2.0],
                          size=[sx, sy, bin_wall_h / 2.0], rgba=[0.55, 0.35, 0.15, 1])

    return spec.compile()


# -----------------------------------------------------------------------------
# Finger IK helpers (ported from lift_fingerPH.py's module-level helpers)
# -----------------------------------------------------------------------------
def _build_theta29(q17: np.ndarray, hand12_sim_vals: np.ndarray) -> np.ndarray:
    th = np.zeros(29, dtype=float)
    th[:17] = q17
    th[17:23] = hand12_sim_vals[:6]
    th[23:29] = hand12_sim_vals[6:]
    return th


def _compute_grasp_targets(q17: np.ndarray, hand12_sim_vals: np.ndarray,
                            cx: float, cy: float, cz: float) -> dict:
    th29 = _build_theta29(q17, hand12_sim_vals)
    fk = fk_fingers(th29)
    targets: dict = {"L": {}, "R": {}}
    for side, fk_tips in (("L", fk.T_L), ("R", fk.T_R)):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            p = fk_tips[finger][:3, 3]
            dx = float(p[0]) - cx
            dy = float(p[1]) - cy
            d = float(np.hypot(dx, dy))
            if d < 1e-3:
                continue
            targets[side][finger] = np.array([
                cx + dx * CYL_RADIUS / d, cy + dy * CYL_RADIUS / d, float(p[2]),
            ])
    return targets


def apply_pos_overrides(q_des: np.ndarray, sim_joint_names: List[str],
                         overrides: Optional[Dict[str, float]]) -> None:
    if not overrides:
        return
    for jn, v in overrides.items():
        q_des[sim_joint_names.index(jn)] = float(v)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MuJoCo pick-and-place port of lift_fingerPH.py.")
    p.add_argument("--urdf_path", type=str, default=DEFAULT_URDF)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--dt", type=float, default=0.004)
    p.add_argument("--cycles", type=int, default=0,
                    help="Run N full S1..S10 loops then exit. 0 = loop forever (default).")

    for k in range(1, 11):
        p.add_argument(f"--t_seg_{k}", type=float, default=1.5)

    p.add_argument("--kp", type=float, default=4.0)
    p.add_argument("--kd", type=float, default=0.7)
    p.add_argument("--ki", type=float, default=1.0)
    p.add_argument("--i_clamp", type=float, default=0.35)

    p.add_argument("--ph_kp", type=float, default=4.0)
    p.add_argument("--ph_kd", type=float, default=0.7)
    p.add_argument("--ph_ki", type=float, default=0.1)
    p.add_argument("--ph_i_clamp", type=float, default=0.35)
    p.add_argument("--ph_kp_thumb_yaw", type=float, default=1.0)
    p.add_argument("--Kf", type=float, default=1.0)
    p.add_argument("--K_spr", type=float, default=2.5)
    p.add_argument("--K_spr_thumb_pitch", type=float, default=14.0)
    p.add_argument("--grasp_torque", type=float, default=5.0)
    p.add_argument("--finger_tau_clamp", type=float, default=2.0)

    p.add_argument("--Kt_free", type=float, default=5.0)
    p.add_argument("--Bt_free", type=float, default=0.5)
    p.add_argument("--Kt_close", type=float, default=150.0)
    p.add_argument("--Bt_close", type=float, default=8.0)
    p.add_argument("--Kt_grasp", type=float, default=250.0)
    p.add_argument("--Bt_grasp", type=float, default=12.0)
    p.add_argument("--contact_threshold", type=float, default=0.3)

    p.add_argument("--no_waist", action="store_true",
                    help="Disable waist pitch/roll in the two-hand IK (arms-only reach). "
                         "Use to check whether waist coupling is the source of large "
                         "hand Z excursions when the cube target Y changes.")
    p.add_argument("--ik_max_iters", type=int, default=2000)
    p.add_argument("--ik_lambda", type=float, default=0.2)
    p.add_argument("--ik_alpha", type=float, default=0.5)
    p.add_argument("--ik_tol_p", type=float, default=1e-4)
    p.add_argument("--ik_tol_w", type=float, default=1e-4)
    p.add_argument("--ik_dq_max_deg", type=float, default=1.0)
    p.add_argument("--print_every", type=int, default=250)
    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = build_argparser().parse_args()
    seg_T = [float(getattr(args, f"t_seg_{k}")) for k in range(1, 11)]

    urdf_path = Path(args.urdf_path).resolve()
    model = build_scene(urdf_path)
    apply_fidelity(model)
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)

    sim_joint_names, qposadr, dofadr = joint_addrs(model)

    q0 = np.array([DEFAULT_JOINT_POS.get(n, 0.0) for n in sim_joint_names])
    data.qpos[qposadr] = q0
    mujoco.mj_forward(model, data)

    # ----- PH HAND12 controller (unchanged from lift_fingerPH.py) -----
    urdf_model = parse_urdf(str(urdf_path))
    tree = build_tree(urdf_model)
    ph_sys = PHSystem(tree, control_joint_names=HAND12_NAMES)
    ftip_jac = FingertipJacobian(
        ph_sys=ph_sys, tip_link_names=TIP_LINK_NAMES,
        hand12_tree_idx=np.array(ph_sys.control_indices),
    )
    ph_kp_arr = np.full(12, args.ph_kp)
    ph_kp_arr[list(THUMB_YAW_IDX)] = args.ph_kp_thumb_yaw
    K_spr_arr = np.full(12, args.K_spr)
    K_spr_arr[list(THUMB_PITCH_IDX)] = args.K_spr_thumb_pitch

    ph_ctrl = PHContactForceController(
        ph_sys=ph_sys, ftip_jac=ftip_jac, sim_joint_names=sim_joint_names,
        Kp=ph_kp_arr, Kd=np.full(12, args.ph_kd), Ki=np.full(12, args.ph_ki),
        i_clamp=args.ph_i_clamp, Kf=args.Kf, K_spr=K_spr_arr,
        Kt_free=args.Kt_free, Bt_free=args.Bt_free,
        Kt_grasp=args.Kt_free, Bt_grasp=args.Bt_free,
        contact_threshold=args.contact_threshold,
    )
    print(f"[PH] IDA-PBC controller ready: NB={tree.nb}, HAND12 sim indices "
          f"{ph_ctrl.hand12_sim_indices}, {len(ph_ctrl.missing_in_sim)} tree DOFs not in sim.")

    _gt = float(args.grasp_torque)
    tau_des_grasp = np.array([
        0.0, _gt, _gt * 0.25, _gt * 0.25, _gt * 0.25, _gt * 0.25,
        0.0, _gt, _gt * 0.25, _gt * 0.25, _gt * 0.25, _gt * 0.25,
    ])

    num_robot_joints = len(sim_joint_names)
    ik17_idx = [sim_joint_names.index(n) for n in IK17_JOINT_NAMES]
    hand12_idx = ph_ctrl.hand12_sim_indices
    hold_ctrl = JointPDIController(
        num_joints=num_robot_joints, joint_indices=ik17_idx,
        Kp=args.kp, Kd=args.kd, Ki=args.ki, i_clamp=args.i_clamp,
    )
    grav_mask = np.zeros(num_robot_joints)
    grav_mask[ik17_idx] = 1.0

    effort_limits = np.array([urdf_model.joints_by_name[n].effort for n in sim_joint_names])

    viewer = None
    if not args.headless:
        import mujoco.viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)

    # ----- boot cube detection (camera, with a spawn-pose fallback like the reference) -----
    for _ in range(10):
        mujoco.mj_step(model, data)
    cube_pos = detect_object_world_pos(renderer, model, data, "d435", "cube_geom")
    if cube_pos is None:
        print("[WARN] Cube not detected by camera at boot -- using spawn-pose fallback.")
        cube_pos = CUBE_POS_PELVIS.copy()
    print(f"[INFO] cube position (pelvis frame) = [{cube_pos[0]:+.3f}, {cube_pos[1]:+.3f}, {cube_pos[2]:+.3f}] m")

    segments = make_segments(cube_pos, seg_T)
    total_T = sum(s.T for s in segments)

    q_des_active = q0.copy()
    q17_prev = np.zeros(17)
    q_h12_prev = q0[hand12_idx].copy()
    active_seg = -1
    step_idx = 0
    completed_cycles = 0
    dq_max = math.radians(args.ik_dq_max_deg)

    print(f"[INFO] total_T={total_T:.1f}s per cycle, dt={args.dt}, cycles="
          f"{'infinite' if args.cycles == 0 else args.cycles}")

    try:
        while True:
            t = step_idx * args.dt
            t_in = t % total_T
            if step_idx > 0 and t_in < args.dt and t > 0:
                completed_cycles += 1
                if args.cycles and completed_cycles >= args.cycles:
                    break
            seg_i, seg_name = get_segment_index(t_in, segments)

            if step_idx % CAM_UPDATE_EVERY == 0:
                _p = detect_object_world_pos(renderer, model, data, "d435", "cube_geom")
                if _p is not None:
                    cube_pos = _p

            seg = segments[seg_i]

            if seg_i != active_seg:
                segments = make_segments(cube_pos, seg_T)
                seg = segments[seg_i]
                active_seg = seg_i
                hold_ctrl.reset_integral()
                ph_ctrl.reset_integral()

                q_tree_now = ph_ctrl.sim_to_tree_q(data.qpos[qposadr])

                if seg.name in ("S1-pregrasp", "S2-close-approach", "S9-idle", "S10-idle"):
                    ph_ctrl.set_position_mode()
                    ph_ctrl.Kt_grasp, ph_ctrl.Bt_grasp = args.Kt_free, args.Bt_free
                    ph_ctrl.contact_active = False
                elif seg.name == "S3-grasp-PH":
                    ph_ctrl.set_force_mode(tau_des_grasp)
                    ph_ctrl.Kt_grasp, ph_ctrl.Bt_grasp = args.Kt_close, args.Bt_close
                    ph_ctrl.x_des = None
                    ph_ctrl.contact_active = False
                elif seg.name in ("S4-lift-PH", "S5-moveair-PH"):
                    ph_ctrl.set_force_mode(tau_des_grasp)
                    ph_ctrl.Kt_grasp, ph_ctrl.Bt_grasp = args.Kt_grasp, args.Bt_grasp
                    ph_ctrl.contact_active = True
                    if ph_ctrl.x_des is None:
                        ph_ctrl.set_task_target(q_tree_now)
                elif seg.name in ("S6-release", "S7-drop", "S8-idle"):
                    ph_ctrl.set_position_mode()
                    ph_ctrl.Kt_grasp, ph_ctrl.Bt_grasp = args.Kt_free, args.Bt_free
                    ph_ctrl.x_des = None
                    ph_ctrl.contact_active = False

                q_des = q0.copy()
                if seg.hands_pL is not None and seg.hands_pR is not None:
                    RL_seg = seg.RL_target if seg.RL_target is not None else np.eye(3)
                    RR_seg = seg.RR_target if seg.RR_target is not None else np.eye(3)
                    q_sol17, ik_info = ik_two_hands_dls(
                        pL_target=np.asarray(seg.hands_pL).reshape(3),
                        pR_target=np.asarray(seg.hands_pR).reshape(3),
                        RL_target=np.asarray(RL_seg).reshape(3, 3),
                        RR_target=np.asarray(RR_seg).reshape(3, 3),
                        use_waist=not args.no_waist, q0=q17_prev,
                        max_iters=args.ik_max_iters, lamb=args.ik_lambda, alpha=args.ik_alpha,
                        tol_p=args.ik_tol_p, tol_w=args.ik_tol_w, dq_max=dq_max,
                    )
                    q17_prev = np.asarray(q_sol17).reshape(17)
                    for i, n in enumerate(IK17_JOINT_NAMES):
                        q_des[sim_joint_names.index(n)] = q17_prev[i]
                    if ik_info.converged:
                        fe = ik_info.final_errors
                        print(f"[IK]  {seg.name} converged in {ik_info.iters} iters: "
                              f"eL_p={fe['eL_p_norm']:.4f}m eR_p={fe['eR_p_norm']:.4f}m")

                hand12_sim_vals_now = data.qpos[qposadr[hand12_idx]]
                if seg.use_grasp_ik:
                    cx_, cy_, cz_ = cube_pos.tolist()
                    grasp_tgts = _compute_grasp_targets(q17_prev, hand12_sim_vals_now, cx_, cy_, cz_)
                    th29_0 = _build_theta29(q17_prev, hand12_sim_vals_now)
                    res = ik_fingers_analytical(grasp_tgts, th29_0, validate=False)
                    q_h12_prev[:6] = res.q_hand6_L
                    q_h12_prev[6:] = res.q_hand6_R
                elif seg.finger_curl_L is not None:
                    th29_0 = _build_theta29(q17_prev, hand12_sim_vals_now)
                    q_curl = ik_fingers_from_curl(
                        curl_L=seg.finger_curl_L,
                        curl_R=seg.finger_curl_R if seg.finger_curl_R is not None else seg.finger_curl_L,
                        theta29_0=th29_0,
                    )
                    q_h12_prev[:6] = q_curl[17:23]
                    q_h12_prev[6:] = q_curl[23:29]
                # else: hold -- q_h12_prev unchanged

                for i, n in enumerate(HAND12_NAMES):
                    q_des[sim_joint_names.index(n)] = q_h12_prev[i]
                apply_pos_overrides(q_des, sim_joint_names, seg.joint_pos_overrides)
                q_des_active = q_des
                ph_ctrl.q_des_hand12 = q_des_active[hand12_idx].copy()
                print(f"[SEG] t={t:6.2f}s -> {seg_name}  cube=[{cube_pos[0]:+.3f},"
                      f"{cube_pos[1]:+.3f},{cube_pos[2]:+.3f}]")

            # ----- per-step torque -----
            q_sim = data.qpos[qposadr].copy()
            qd_sim = data.qvel[dofadr].copy()

            tau_ideal_12 = ph_ctrl.compute(q_sim, qd_sim, dt=args.dt)
            tau_delivered_12 = np.clip(tau_ideal_12, -args.finger_tau_clamp, args.finger_tau_clamp)

            grav = np.zeros(num_robot_joints)
            grav[ik17_idx] = data.qfrc_bias[dofadr][ik17_idx]
            tau_hold = hold_ctrl.compute(q_sim, qd_sim, q_des_active, args.dt, gravity_comp=grav * grav_mask)
            tau_full = tau_hold + ph_ctrl.scatter_to_full(tau_delivered_12, num_robot_joints)
            tau_full = np.clip(tau_full, -effort_limits, effort_limits)

            data.qfrc_applied[dofadr] = tau_full
            mujoco.mj_step(model, data)
            step_idx += 1

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()

            if step_idx % args.print_every == 0:
                cube_world = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")]
                print(f"[RUN] t={t:6.2f}s seg={seg_name}  cube_actual_z={cube_world[2]:+.3f}  "
                      f"|tau_hand12|max={np.max(np.abs(tau_delivered_12)):.2f}Nm")
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        renderer.close()


if __name__ == "__main__":
    main()
