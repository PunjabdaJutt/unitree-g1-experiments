# pin_kinematics.py
"""Pinocchio-based replacement for kinematics.py's fk_chain/ik_two_hands_dls,
for the 17-DOF waist+two-arm chain (IK17_JOINT_NAMES order). Also provides a
right-arm-ONLY (7 DOF), shoulder-origin FK/IK layer (shoulder_origin_pin/
ik_right_arm_dls_pin/fk_right_arm_pin, near the bottom of this file) for
grasp_ph.py's pick-and-place pipeline -- built on top of the same 17-DOF
solve (waist fixed, left arm locked), not a separate reduced model.

REQUIRES THE SEPARATE `pinocchio` CONDA ENV, NOT this repo's .venv --
`pip install pin` into .venv installs a real, importable package, but its
compiled extension throws `undefined symbol: EIGENPY_ARRAY_APIPyArray_RUNTIME_VERSION`
at import time (a genuine ABI mismatch in that wheel chain -- confirmed
2026-09-10 across pin 4.0.0/4.1.0 and numpy 2.3.5/2.5.3, so it isn't a
version-pinning problem on our end). `conda create -n pinocchio -c
conda-forge python=3.12 pinocchio` installs a working build instead. Run
anything in this file with that env's interpreter directly (not .venv, not
a bare `conda activate` in a shell that's already sourced .venv -- that
leftover PATH/PYTHONPATH/VIRTUAL_ENV state shadows the conda env's own
python even after activating; safest is the conda env's python by full
path, e.g.:
    /home/ammar/miniconda3/envs/pinocchio/bin/python3
This module does NOT import anything from kinematics.py / urdf_parser.py /
robot_indices.py -- those are numpy+this-repo's-own-URDF-parser-based and
live happily in .venv, but importing them here would pull .venv's own
(broken) numpy/eigenpy expectations into a script meant to run under the
conda env's own numpy. IK17_JOINT_NAMES is duplicated below (FREE_17) --
small, frozen, unlikely to change -- rather than crossing that boundary.

Math: same damped least-squares update law as kinematics.py's own module
docstring (dq = J^T(JJ^T + lamb^2 I)^-1 e), but using Pinocchio's own
forwardKinematics/Jacobian/log6 instead of this repo's hand-rolled
fk_chain/geometric_jacobian -- cross-validated 2026-09-10: FK for both
wrist frames at q=0 matches kinematics.fk_chain's output to 4.4e-7m.

Reduced model: the URDF's full 53-DOF model (29 body + 12x2 raw finger
joints) is reduced via pin.buildReducedModel() down to exactly the 17
waist+arm joints IK17_JOINT_NAMES already names -- everything else (legs,
every finger joint) is locked at 0, matching kinematics.py's own
IK17-only scope (this repo's IK never touches legs or fingers either, see
grasp_ph.py's own READY_POSE comment on why legs can't be touched at all).
Pinocchio happened to preserve URDF declaration order across the
reduction, which already matches IK17_JOINT_NAMES joint-for-joint --
verified, not assumed (see build_reduced_model()'s own assertion).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# `conda activate pinocchio` does NOT clear PYTHONPATH -- if a shell has ROS2
# sourced (even just via .bashrc, not deliberately), /opt/ros/*/lib/python*/
# site-packages stays on sys.path (PYTHONPATH entries land ahead of the venv/
# conda env's own site-packages) and shadows this conda env's own Pinocchio
# with ROS2's system copy, which is built against NumPy 1.x and crashes on
# import against this env's NumPy 2.x (confirmed 2026-09-10/11 -- same
# ImportError either way: "compiled using NumPy 1.x cannot be run in NumPy
# 2.x"). Strip any ROS site-packages entries before importing, so this module
# works regardless of how the shell that launched it was set up -- not
# something the caller has to remember to do right.
sys.path = [p for p in sys.path if "/opt/ros/" not in p]

import pinocchio as pin  # noqa: E402

DEFAULT_URDF = ("/home/ammar/unitree-g1-experiments/simulation/Port-Hamiltonian-G1-Finger-Control/"
                 "unitree_g1_inspire_assets/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")

# Same order as ph/robot_indices.py's IK17_JOINT_NAMES -- duplicated here
# rather than imported, see module docstring.
FREE_17 = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# The 7-DOF right-arm-only subset of FREE_17, and its complement (left arm) --
# see ik_right_arm_dls_pin's own docstring for what these are used for.
RIGHT_ARM_JOINT_NAMES = [n for n in FREE_17 if n.startswith("right_")]
LEFT_ARM_JOINT_NAMES = [n for n in FREE_17 if n.startswith("left_")]
_RIGHT_ARM_IDX = [FREE_17.index(n) for n in RIGHT_ARM_JOINT_NAMES]

_MODEL_CACHE: Dict[str, pin.Model] = {}


def build_reduced_model(urdf_path: str = DEFAULT_URDF) -> pin.Model:
    """Lazily parse + reduce the G1+Inspire-Hand URDF down to the 17
    waist+arm DOF, cached per urdf_path. Asserts the reduced model's own
    joint order matches FREE_17 exactly -- if a future URDF revision ever
    reorders joints, this fails loudly here rather than silently feeding
    q17 values into the wrong columns downstream."""
    if urdf_path in _MODEL_CACHE:
        return _MODEL_CACHE[urdf_path]
    full = pin.buildModelFromUrdf(urdf_path)
    all_names = [full.names[i] for i in range(1, full.njoints)]
    missing = [n for n in FREE_17 if n not in all_names]
    if missing:
        raise ValueError(f"FREE_17 names not found in URDF: {missing}")
    lock_ids = [full.getJointId(n) for n in all_names if n not in FREE_17]
    q_ref = pin.neutral(full)
    reduced = pin.buildReducedModel(full, lock_ids, q_ref)
    reduced_names = [reduced.names[i] for i in range(1, reduced.njoints)]
    assert reduced_names == FREE_17, (
        f"reduced model joint order {reduced_names} != FREE_17 {FREE_17} -- "
        "pin.buildReducedModel() no longer preserves URDF declaration order; "
        "IK below assumes q17[i] is FREE_17[i], this would silently corrupt it."
    )
    _MODEL_CACHE[urdf_path] = reduced
    return reduced


def fk_chain_pin(model: pin.Model, data: pin.Data, q17: np.ndarray, frame_name: str) -> np.ndarray:
    """(4,4) T_frame_from_pelvis, matching kinematics.fk_chain's return
    convention (same numpy 4x4, root_link="pelvis" implicitly -- pelvis IS
    this reduced model's root, there being no floating/free joint in this
    URDF chain between pelvis and the waist)."""
    pin.forwardKinematics(model, data, q17)
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId(frame_name)
    T = np.eye(4)
    T[:3, :3] = data.oMf[fid].rotation
    T[:3, 3] = data.oMf[fid].translation
    return T


@dataclass
class IKInfo:
    converged: bool = False
    iters: int = 0
    final_errors: Dict[str, float] = field(default_factory=dict)


def _pose_error(current: pin.SE3, target_p: np.ndarray, target_R: Optional[np.ndarray]):
    """World-frame (linear, angular) twist error, consistent with
    LOCAL_WORLD_ALIGNED Jacobians (see module docstring's math note) --
    e_lin = target - current (plain Euclidean, matches Jv directly);
    e_ang = current.R @ log3(current.R^T @ target.R) if a target_R was
    given, else None (position-only task, matches kinematics.py's own
    RL_target/RR_target=None convention: rotation rows only get stacked in
    when a target orientation is actually supplied)."""
    e_lin = target_p - current.translation
    e_ang = None
    if target_R is not None:
        R_err = current.rotation.T @ target_R
        e_ang = current.rotation @ pin.log3(R_err)
    return e_lin, e_ang


def ik_two_hands_dls_pin(
    pL_target: np.ndarray, pR_target: np.ndarray,
    RL_target: Optional[np.ndarray] = None, RR_target: Optional[np.ndarray] = None,
    use_waist: bool = True, lock_joint_names: Optional[Sequence[str]] = None,
    waist_weight: float = 1.0, wrist_weight: float = 1.0,
    joint_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    q0: Optional[np.ndarray] = None,
    max_iters: int = 2000, lamb: float = 0.05, alpha: float = 0.8,
    tol_p: float = 1e-4, tol_w: float = 1e-4, dq_max: Optional[float] = None,
    w_pos: float = 1.0, w_rot: float = 1.0, model: Optional[pin.Model] = None,
) -> Tuple[np.ndarray, IKInfo]:
    """Pinocchio port of kinematics.ik_two_hands_dls -- same signature,
    same DLS update law, same use_waist/lock_joint_names/waist_weight/
    wrist_weight/joint_bounds semantics (see kinematics.py's own docstrings
    for what each does and why -- not repeated here). Only the FK/Jacobian
    engine underneath differs. See module docstring for the cross-validation
    this was checked against."""
    model = model or build_reduced_model()
    data = model.createData()
    n = len(FREE_17)
    q = np.zeros(n) if q0 is None else np.asarray(q0, dtype=float).reshape(n).copy()
    lo = model.lowerPositionLimit.copy()
    hi = model.upperPositionLimit.copy()
    for name, (b_lo, b_hi) in (joint_bounds or {}).items():
        jname = name if name.endswith("_joint") else name + "_joint"
        if jname not in FREE_17:
            raise ValueError(f"joint_bounds: {name!r} is not a FREE_17 entry")
        idx = FREE_17.index(jname)
        lo[idx] = max(lo[idx], b_lo)
        hi[idx] = min(hi[idx], b_hi)
    wrist_idx = [i for i, name in enumerate(FREE_17) if "wrist" in name]

    lock_idx: List[int] = []
    for name in (lock_joint_names or []):
        jname = name if name.endswith("_joint") else name + "_joint"
        if jname not in FREE_17:
            raise ValueError(f"lock_joint_names: {name!r} is not a FREE_17 entry")
        lock_idx.append(FREE_17.index(jname))

    pL_target = np.asarray(pL_target, dtype=float).reshape(3)
    pR_target = np.asarray(pR_target, dtype=float).reshape(3)
    fid_L = model.getFrameId("left_wrist_yaw_link")
    fid_R = model.getFrameId("right_wrist_yaw_link")

    errs: Dict[str, float] = {}
    for it in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)
        T_L, T_R = data.oMf[fid_L], data.oMf[fid_R]

        e_rows: List[np.ndarray] = []
        J_rows: List[np.ndarray] = []
        errs = {}

        eL_p, eL_w = _pose_error(T_L, pL_target, RL_target)
        errs["eL_p_norm"] = float(np.linalg.norm(eL_p))
        JL = pin.getFrameJacobian(model, data, fid_L, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        e_rows.append(w_pos * eL_p); J_rows.append(w_pos * JL[:3])
        if eL_w is not None:
            errs["eL_w_norm"] = float(np.linalg.norm(eL_w))
            e_rows.append(w_rot * eL_w); J_rows.append(w_rot * JL[3:])

        eR_p, eR_w = _pose_error(T_R, pR_target, RR_target)
        errs["eR_p_norm"] = float(np.linalg.norm(eR_p))
        JR = pin.getFrameJacobian(model, data, fid_R, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        e_rows.append(w_pos * eR_p); J_rows.append(w_pos * JR[:3])
        if eR_w is not None:
            errs["eR_w_norm"] = float(np.linalg.norm(eR_w))
            e_rows.append(w_rot * eR_w); J_rows.append(w_rot * JR[3:])

        pos_ok = errs["eL_p_norm"] < tol_p and errs["eR_p_norm"] < tol_p
        rot_ok = (RL_target is None or errs.get("eL_w_norm", 0.0) < tol_w) and \
                 (RR_target is None or errs.get("eR_w_norm", 0.0) < tol_w)
        if pos_ok and rot_ok:
            return q, IKInfo(converged=True, iters=it, final_errors=errs)

        e = np.concatenate(e_rows)
        J = np.vstack(J_rows)
        if not use_waist or lock_idx:
            J = J.copy()
            if not use_waist:
                J[:, :3] = 0.0
            for idx in lock_idx:
                J[:, idx] = 0.0

        if waist_weight != 1.0 or wrist_weight != 1.0:
            inv_sqrt_w = np.ones(J.shape[1])
            inv_sqrt_w[:3] = 1.0 / waist_weight
            for idx in wrist_idx:
                inv_sqrt_w[idx] = 1.0 / wrist_weight
            Jw = J * inv_sqrt_w[np.newaxis, :]
            JwJwt = Jw @ Jw.T
            dq = inv_sqrt_w * (Jw.T @ np.linalg.solve(JwJwt + (lamb ** 2) * np.eye(JwJwt.shape[0]), e))
        else:
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + (lamb ** 2) * np.eye(JJt.shape[0]), e)

        step = alpha * dq
        if dq_max is not None:
            step = np.clip(step, -dq_max, dq_max)
        q = pin.integrate(model, q, step)
        q = np.clip(q, lo, hi)

    return q, IKInfo(converged=False, iters=max_iters, final_errors=errs)


def shoulder_origin_pin(waist_q: Sequence[float] = (0.0, 0.0, 0.0),
                         model: Optional[pin.Model] = None) -> np.ndarray:
    """(3,) position of the right_shoulder_pitch joint's pivot -- the fixed
    point right_shoulder_pitch/roll/yaw and right_elbow/wrist all rotate
    about or downstream of, never translating it -- expressed in pelvis
    coordinates, at the given waist angles (waist_yaw, waist_roll,
    waist_pitch). Right-arm joint angles don't matter for this (the pivot's
    own position is invariant to them), so this only ever needs a single FK
    call at zero arm angles."""
    model = model or build_reduced_model()
    data = model.createData()
    q17 = np.zeros(len(FREE_17))
    q17[0], q17[1], q17[2] = waist_q
    T = fk_chain_pin(model, data, q17, "right_shoulder_pitch_link")
    return T[:3, 3].copy()


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def ik_right_arm_dls_pin(
    p_target: np.ndarray, R_target: Optional[np.ndarray] = None,
    waist_q: Sequence[float] = (0.0, 0.0, 0.0),
    q0_7: Optional[np.ndarray] = None,
    joint_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    waist_weight: float = 1.0, wrist_weight: float = 1.0,
    max_iters: int = 200, lamb: float = 0.05, alpha: float = 0.8,
    tol_p: float = 1e-4, tol_w: float = 1e-4, dq_max: Optional[float] = None,
    w_pos: float = 1.0, w_rot: float = 1.0, model: Optional[pin.Model] = None,
    axis_local: Optional[np.ndarray] = None, axis_target: Optional[np.ndarray] = None,
    aim_length: float = 0.1, tol_axis_deg: float = 1.0, aim_weight: float = 1.0,
) -> Tuple[np.ndarray, IKInfo, np.ndarray]:
    """Right-arm-ONLY (7 DOF) IK. `p_target`/`R_target`/`axis_target` are in
    the SHOULDER frame (see module section docstring above for exactly what
    that is).

    Built on ik_two_hands_dls_pin's existing, cross-validated 17-DOF solve
    rather than a new reduced model: waist_q is a fixed, known parameter
    (not a free variable -- there's no waist column to weight/lock/free
    here at all, stronger than the old use_waist=False, which merely zeroed
    it), and the left arm is locked at all-zeros via lock_joint_names, so
    the only 7 columns that can ever move are RIGHT_ARM_JOINT_NAMES's.

    The left-wrist term ik_two_hands_dls_pin always computes internally is
    given a target equal to the left wrist's OWN FK position at this same
    frozen (waist_q, left_arm=0) configuration -- not a real constraint
    (locked columns make its Jacobian contribution zero anyway), but the
    DLS convergence check still requires eL_p_norm < tol_p, so this has to
    already be satisfied by the frozen pose for CONVERGED to ever be true.

    axis_local/axis_target: OPTIONAL alternative to R_target -- a partial
    (2-DOF) "aim constraint" instead of a full 3-DOF orientation match,
    freeing the wrist's roll about its own approach axis as real spare
    redundancy for the solver instead of pinning it to whatever
    Gram-Schmidt happened to produce (see trajectory_gen._build_approach_
    rotation's own docstring: that roll was only ever "as close to
    forward_ref as geometry allows", never an exact requirement).
    axis_local: (3,) unit vector in right_wrist_yaw_link's OWN local frame
    (e.g. trajectory_gen.PALM_NORMAL_LOCAL -- the palm-normal/plunge
    direction). axis_target: (3,) unit vector, SHOULDER frame, the
    direction that local axis should point once solved (e.g. a waypoint's
    own `-normal`; recoverable directly as `wp.R @ PALM_NORMAL_LOCAL`
    without re-deriving `normal` separately, since that's exactly what the
    original R was built to satisfy). When axis_target is given, R_target
    is ignored entirely and a self-contained aim-constrained DLS loop runs
    instead of delegating to ik_two_hands_dls_pin (whose fixed 3-DOF
    RR_target/RL_target error structure has no room for a 2-DOF task).

    Solved via a standard "aim/look-at" IK trick: a virtual point
    `aim_length` meters along axis_local from the wrist origin is driven
    toward the same point measured from the target pose. Its Jacobian is
    the wrist's own linear Jacobian corrected for that rigid offset (JR[:3]
    - skew(aim_length * current_axis_world) @ JR[3:]) -- ordinary rigid-
    body point-Jacobian algebra, not a new primitive. Convergence checks
    the ANGLE between current and target axis directly (tol_axis_deg)
    rather than the aim point's raw residual norm (which conflates
    position-task error with axis error, since the aim point sits at the
    wrist position plus an offset -- see aim_weight below).

    aim_weight: same weighted-DLS mechanism as wrist_weight/waist_weight
    elsewhere in this module, but scaling the AIM TASK's error/Jacobian
    ROWS (not a joint-column weight) -- >1 makes hitting the axis direction
    relatively more important than position within the combined solve, <1
    the reverse; 1.0 (default) weights them equally, same convention as
    w_pos/w_rot elsewhere in this module.

    Returns (q7, IKInfo, t_shoulder) -- t_shoulder is shoulder_origin_pin
    (waist_q), handed back so a caller can log/verify without a second
    call. IKInfo.final_errors carries "axis_err_deg" instead of
    "eR_w_norm" when axis_target was used."""
    model = model or build_reduced_model()
    t_shoulder = shoulder_origin_pin(waist_q, model=model)

    p_target = np.asarray(p_target, dtype=float).reshape(3)
    p_target_pelvis = p_target + t_shoulder

    q17_seed = np.zeros(len(FREE_17))
    q17_seed[0], q17_seed[1], q17_seed[2] = waist_q
    if q0_7 is not None:
        q0_7 = np.asarray(q0_7, dtype=float).reshape(7)
        for i, idx in enumerate(_RIGHT_ARM_IDX):
            q17_seed[idx] = q0_7[i]

    data0 = model.createData()
    fid_L = model.getFrameId("left_wrist_yaw_link")
    pin.forwardKinematics(model, data0, q17_seed)
    pin.updateFramePlacements(model, data0)
    pL_hold = data0.oMf[fid_L].translation.copy()

    if axis_target is None:
        R_target_pelvis = None if R_target is None else np.asarray(R_target, dtype=float)
        q17, info = ik_two_hands_dls_pin(
            pL_target=pL_hold, pR_target=p_target_pelvis,
            RL_target=None, RR_target=R_target_pelvis,
            use_waist=False, lock_joint_names=LEFT_ARM_JOINT_NAMES,
            waist_weight=waist_weight, wrist_weight=wrist_weight,
            joint_bounds=joint_bounds, q0=q17_seed,
            max_iters=max_iters, lamb=lamb, alpha=alpha,
            tol_p=tol_p, tol_w=tol_w, dq_max=dq_max,
            w_pos=w_pos, w_rot=w_rot, model=model,
        )
        q7 = q17[_RIGHT_ARM_IDX]
        return q7, info, t_shoulder

    # ---- Aim-constrained solve (frees roll about axis_local) ----
    axis_local = np.asarray(axis_local, dtype=float)
    axis_local = axis_local / np.linalg.norm(axis_local)
    axis_target_pelvis = np.asarray(axis_target, dtype=float)
    axis_target_pelvis = axis_target_pelvis / np.linalg.norm(axis_target_pelvis)
    aim_point_target = p_target_pelvis + aim_length * axis_target_pelvis

    q = q17_seed.copy()
    lo = model.lowerPositionLimit.copy()
    hi = model.upperPositionLimit.copy()
    for name, (b_lo, b_hi) in (joint_bounds or {}).items():
        jname = name if name.endswith("_joint") else name + "_joint"
        if jname not in FREE_17:
            raise ValueError(f"joint_bounds: {name!r} is not a FREE_17 entry")
        idx = FREE_17.index(jname)
        lo[idx] = max(lo[idx], b_lo)
        hi[idx] = min(hi[idx], b_hi)
    wrist_idx = [i for i, name in enumerate(FREE_17) if "wrist" in name]
    lock_idx = [FREE_17.index(name) for name in LEFT_ARM_JOINT_NAMES]

    data = model.createData()
    fid_R = model.getFrameId("right_wrist_yaw_link")

    errs: Dict[str, float] = {}
    for it in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)
        T_L, T_R = data.oMf[fid_L], data.oMf[fid_R]

        eL_p = pL_hold - T_L.translation
        JL = pin.getFrameJacobian(model, data, fid_L, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        eR_p = p_target_pelvis - T_R.translation
        JR = pin.getFrameJacobian(model, data, fid_R, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        current_axis_world = T_R.rotation @ axis_local
        aim_point_current = T_R.translation + aim_length * current_axis_world
        e_aim = aim_point_target - aim_point_current
        J_aim = JR[:3] - _skew(aim_length * current_axis_world) @ JR[3:]

        axis_cos = float(np.clip(np.dot(current_axis_world, axis_target_pelvis), -1.0, 1.0))
        errs = {
            "eL_p_norm": float(np.linalg.norm(eL_p)),
            "eR_p_norm": float(np.linalg.norm(eR_p)),
            "axis_err_deg": float(np.degrees(np.arccos(axis_cos))),
        }

        pos_ok = errs["eL_p_norm"] < tol_p and errs["eR_p_norm"] < tol_p
        axis_ok = errs["axis_err_deg"] < tol_axis_deg
        if pos_ok and axis_ok:
            return q[_RIGHT_ARM_IDX], IKInfo(converged=True, iters=it, final_errors=errs), t_shoulder

        e = np.concatenate([w_pos * eL_p, w_pos * eR_p, aim_weight * e_aim])
        J = np.vstack([w_pos * JL[:3], w_pos * JR[:3], aim_weight * J_aim])
        J = J.copy()
        J[:, :3] = 0.0  # this solver never treats waist as free -- see docstring
        for idx in lock_idx:
            J[:, idx] = 0.0

        if waist_weight != 1.0 or wrist_weight != 1.0:
            inv_sqrt_w = np.ones(J.shape[1])
            inv_sqrt_w[:3] = 1.0 / waist_weight
            for idx in wrist_idx:
                inv_sqrt_w[idx] = 1.0 / wrist_weight
            Jw = J * inv_sqrt_w[np.newaxis, :]
            JwJwt = Jw @ Jw.T
            dq = inv_sqrt_w * (Jw.T @ np.linalg.solve(JwJwt + (lamb ** 2) * np.eye(JwJwt.shape[0]), e))
        else:
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + (lamb ** 2) * np.eye(JJt.shape[0]), e)

        step = alpha * dq
        if dq_max is not None:
            step = np.clip(step, -dq_max, dq_max)
        q = pin.integrate(model, q, step)
        q = np.clip(q, lo, hi)

    return q[_RIGHT_ARM_IDX], IKInfo(converged=False, iters=max_iters, final_errors=errs), t_shoulder


def fk_right_arm_pin(q7: np.ndarray, waist_q: Sequence[float] = (0.0, 0.0, 0.0),
                      frame_name: str = "right_wrist_yaw_link",
                      model: Optional[pin.Model] = None) -> np.ndarray:
    """(4,4) T_frame_from_shoulder for the 7-DOF right arm only -- same
    pelvis-aligned-axes shoulder frame as shoulder_origin_pin/
    ik_right_arm_dls_pin above (rotation is untouched from the underlying
    pelvis-frame FK; only the translation's origin is shifted to the
    shoulder pivot)."""
    model = model or build_reduced_model()
    q7 = np.asarray(q7, dtype=float).reshape(7)
    q17 = np.zeros(len(FREE_17))
    q17[0], q17[1], q17[2] = waist_q
    for i, idx in enumerate(_RIGHT_ARM_IDX):
        q17[idx] = q7[i]
    T = fk_chain_pin(model, model.createData(), q17, frame_name)
    t_shoulder = shoulder_origin_pin(waist_q, model=model)
    T_out = T.copy()
    T_out[:3, 3] = T[:3, 3] - t_shoulder
    return T_out


def gravity_right_arm_pin(q7: np.ndarray, waist_q: Sequence[float] = (0.0, 0.0, 0.0),
                           model: Optional[pin.Model] = None) -> np.ndarray:
    """(7,) generalized gravity torque [Nm] for the right arm only, RIGHT_ARM_JOINT_NAMES
    order -- pin.computeGeneralizedGravity() on the same 17-DOF reduced model FK/IK above
    already use, at q17 = [waist_q, left_arm=0, q7]. Left arm's true angles don't affect this:
    RNEA's generalized force at a joint depends only on its own downstream subtree, and the
    left arm isn't downstream of any right-arm joint in this pelvis-rooted tree (no closed
    loops) -- so zeroing it here is exact, not an approximation, PROVIDED the real left arm
    isn't bearing load from something outside the kinematic chain (e.g. leaning on a table)
    while this is used."""
    model = model or build_reduced_model()
    data = model.createData()
    q7 = np.asarray(q7, dtype=float).reshape(7)
    q17 = np.zeros(len(FREE_17))
    q17[0], q17[1], q17[2] = waist_q
    for i, idx in enumerate(_RIGHT_ARM_IDX):
        q17[idx] = q7[i]
    pin.computeGeneralizedGravity(model, data, q17)
    return np.array([data.g[idx] for idx in _RIGHT_ARM_IDX])
