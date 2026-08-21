# G1 Inspire-Hand: Kinematic Architecture, MDH Extraction, and Analytical IK

Full write-up backing the summary in `../SKILL.md`. Covers: which module is the FK source of
truth, how the MDH table is derived (and why it can't be used for FK), and the closed-form IK
used for the HAND12 proximal joints, including the original Python source.

## 1. Kinematic implementation overview

- **`fk_fingers.py` — ground truth.** Hard-codes URDF joint origins (`xyz`, `rpy`), joint axes,
  and mimic multipliers for the full chain (pelvis → waist → arm → wrist → hand base →
  fingers). FK is computed via direct composition:

  `T ← T · Fixed(xyz, rpy) · Rotate(axis, q)`

  This produces the "verified to < 0.001 mm" reference results.
- **`dh_table.py`** automatically derives a Modified DH table from the same URDF data. This
  serves as a structured representation and sanity check, but is **not** used for FK
  computation.
- **`dh_kinematics.py`** is the controller-facing wrapper. Although it exposes `fk_dh()`, it
  does not use the MDH rows for computation; instead it re-walks the URDF chain directly. The
  MDH table is retained only for inspection and printing.

**The URDF chain is the single source of truth.**

## 2. MDH convention

Craig's Modified DH convention (1989):

`T_{i-1→i} = R_x(α_{i-1}) · T_x(a_{i-1}) · R_z(θ_i + θ_off) · T_z(d_i)`

For joint `i`:

- `a`: distance along `x_{i-1}` between consecutive z-axes
- `α`: angle from `z_{i-1}` to `z_i` about `x_{i-1}`
- `d`: translation along `z_i`
- `θ_off`: angle between `x_{i-1}` and `x_i` about `z_i` at `q = 0`

## 3. MDH parameter extraction

### 3.1 Step 1 — URDF chain traversal

At `q = 0`, the chain is traversed once. For each joint:

- `z_k`: joint axis expressed in the hand-base frame
- `o_k`: joint origin position in the hand-base frame

### 3.2 Step 2 — Canonical frame construction

The DH x-axis is defined as the common perpendicular between consecutive z-axes:

```
x = (z_a × z_b) / ||z_a × z_b||      if not parallel
x = projected origin difference       if parallel
```

Each frame is completed as `y = z × x`.

### 3.3 Step 3 — MDH scalars

For consecutive frames:

```
α     = atan2( (z_{i-1} × z_i) · x_{i-1},  z_{i-1} · z_i )
a     = (o_i − o_{i-1}) · x_{i-1}
d     = (o_i − o_{i-1}) · z_i
θ_off = atan2( (x_{i-1} × x_i) · z_i,  x_{i-1} · x_i )
```

### 3.4 Step 4 — Mimic mapping

Mimic joints are tagged using a reference joint name and a mimic scaling factor
(1.0, 1.6, or 2.4 — see the parameter tables).

## 4. MDH limitation and the 137mm offset

The proximal joints have an offset of approximately 137mm along the −Y direction:

`[0.00028533, -0.1371, 0.01295]`

MDH can only represent offsets along X (via `a`) and Z (via `d`), not Y. Therefore:

- A pure MDH chain would introduce a systematic ~137mm error.
- The implementation avoids this by **not** using MDH rows for FK.

Instead, the call chain is:

`fk_dh() → _fk_urdf_chain()`

ensuring exact agreement with the URDF-based FK.

## 5. Interpretation of table columns

- `a, α, d, θ_off`: MDH parameters (displayed in mm and degrees)
- Axis sign: `sign(axis_z)`
- Joint limits: extracted from `HAND12_LIMITS`
- Type: actuated (`act`) or mimic (`1.6x` / `2.4x` / `1.0x`)

The 137.10mm value in `a` (L/R middle proximal) is a projection artifact from the canonical
frame construction, not real geometry.

## 6. Usage in controller

`DHFingertipKinematics.fk_tips()`:

- Computes fingertip pose per finger using `fk_dh()`
- Left-multiplies by the live hand-base transform `T_hb`
- Uses URDF-consistent FK throughout

The Jacobian is computed via finite differences: `J ≈ ∂x/∂q`.

## 7. Design rationale

Using a single FK implementation avoids inconsistencies between:

- URDF-based FK
- Featherstone-tree FK (`ph_system.py`)

This ensures stable impedance control:

`τ = J_c^T (K_t(x_d − x) + ···)`

Consistency in `x` is critical for numerical stability of the impedance term.

---

## 8. Inverse kinematics — overview

LaTeX/thesis conversion of the Python module `ik_fingers_analytical.py`. It implements
analytical inverse kinematics for the Inspire-Hand DFQ proximal finger joints, replacing a
damped least-squares (DLS) iterative loop with closed-form or near-closed-form solvers.

## 9. Mathematical summary

For each non-thumb finger, the fingertip position in the proximal-joint body frame follows:

`p_tip(q) = p_{1,fixed} + R_{1,fixed} · R_z(−q) · p_link`

Given a target fingertip position, the joint angle is recovered through projection in the
proximal frame:

```
v = R_{1,fixed}^T (p_des − p_{1,fixed})
q = −atan2( p_x v_y − p_y v_x,  p_x v_x + p_y v_y )
```

The thumb uses a two-stage solution: yaw from azimuth, then pitch from a reduced 1-D nonlinear
solve with Newton refinement.

## 10. Python source (`ik_fingers_analytical.py`)

Theory recap embedded in the module docstring:

- Each non-thumb finger (index/middle/ring/pinky) has ONE actuated DOF (proximal joint,
  axis = `[0,0,-1]`) with the intermediate joint mimicking it 1:1. Because the intermediate
  offset is a fixed vector that rotates together with the proximal joint, the fingertip traces
  a **circle** in the proximal-joint body frame:

  `p_tip(q) = p1_fixed + R1_fixed @ Rz(-q) @ p_link_vec`

  where `p1_fixed` = proximal joint origin in hand_base frame (URDF), `R1_fixed` = fixed RPY
  rotation at the proximal joint (URDF), `p_link_vec` = offset to the intermediate joint (=
  fingertip) in the proximal child frame (URDF), `q` = proximal joint angle (positive = curl
  inward).

  Given a desired fingertip `p_des`, solve analytically:

  ```
  v = R1_fixed.T @ (p_des - p1_fixed)          # target in proximal body frame
  q = -atan2(px*vy - py*vx, px*vx + py*vy)     # angle from p_link_xy to v_xy
  ```

  where `(px, py) = p_link_vec[:2]`. This is exact when `||v_xy|| == ||p_link_xy||`; otherwise
  `v` is projected onto the reachable circle.

- **Thumb (2 DOF: yaw + pitch)**:
  1. **Yaw** — solve from the target's azimuth in the yaw-joint body frame.
  2. **Pitch** — given fixed yaw, express the target in the pitch-joint frame and solve the
     3-link effective chain (`pitch + 1.6×pitch + 2.4×pitch`) using a fast 1-D Newton step
     (≤4 iterations, always converges).

- **I/O conventions**: `theta29 (29,) = [waist(3), left_arm(7), right_arm(7), left_hand6(6),
  right_hand6(6)]` — same as `fk_fingers.py`. `q_hand6 (6,)` per hand =
  `[thumb_yaw, thumb_pitch, index, middle, ring, pinky]` — matches `HAND12_SIM_INDICES` order
  within each hand. Isaac Lab HAND12 sim indices (`robot_indices.py`): Left
  `[29, 30, 33, 35, 37, 39]`, Right `[41, 42, 45, 47, 49, 51]`.

```python
"""
ik_fingers_analytical.py
========================
Analytical (closed-form) IK for the Inspire-Hand DFQ proximal finger joints.
Replaces the DLS loop in ik_fingers.py for the HAND12 joints.


Theory
------
Each non-thumb finger (index/middle/ring/pinky) has ONE actuated DOF (proximal
joint, axis=[0,0,-1]) with the intermediate joint mimicking it 1:1. Because
the intermediate offset is a fixed vector that rotates together with the proximal
joint, the fingertip traces a CIRCLE in the proximal-joint body frame.


    p_tip(q) = p1_fixed + R1_fixed @ Rz(-q) @ p_link_vec


where:
    p1_fixed   = proximal joint origin in hand_base frame  (from URDF)
    R1_fixed   = fixed RPY rotation at the proximal joint  (from URDF)
    p_link_vec = offset to the intermediate joint (= fingertip) in the
                 proximal child frame  (from URDF)
    q          = proximal joint angle (positive = curl inward)


Given a desired fingertip p_des, find q analytically:


    v = R1_fixed.T @ (p_des - p1_fixed)          # target in proximal body frame
    q = -atan2(px*vy - py*vx, px*vx + py*vy)     # angle from p_link_xy to v_xy


where (px,py) = p_link_vec[:2]. This is exact when ||v_xy|| = ||p_link_xy|| and
we project v onto the reachable circle otherwise.


Thumb (2 DOF: yaw + pitch)
    Step 1 --- Yaw: solve from target's azimuth in the yaw-joint body frame.
    Step 2 --- Pitch: given fixed yaw, express target in pitch-joint frame and
               solve the 3-link effective chain (pitch + 1.6xpitch + 2.4xpitch)
               using a fast 1-D Newton step (<=4 iterations, always converges).


Input/Output conventions
-------------------------
theta29 (29,) = [waist(3), left_arm(7), right_arm(7), left_hand6(6), right_hand6(6)]
    --- same as fk_fingers.py


q_hand6 (6,) per hand = [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
    --- this matches HAND12_SIM_INDICES order within each hand


Isaac Lab HAND12 sim indices (from robot_indices.py):
    Left:  [29, 30, 33, 35, 37, 39]
    Right: [41, 42, 45, 47, 49, 51]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .fk_fingers import (
    _Rx, _Ry, _Rz, _T, rpy2R_urdf,
    _fk_to_wrist_yaw_left, _fk_to_wrist_yaw_right,
    L_BASE_XYZ, L_BASE_RPY,
    R_BASE_XYZ, R_BASE_RPY,
    L_FINGER_CHAINS, R_FINGER_CHAINS,
    fk_fingers,
)
from .robot_indices import HAND12_LIMITS_LO, HAND12_LIMITS_HI

ArrayLike = np.ndarray

# Precomputed non-thumb chain data: (origin_xyz, origin_rpy, link_vec, axis_sign)
# link_vec = xyz of the intermediate (mimic) joint in the proximal child frame.
_L_NON_THUMB: dict[str, tuple] = {
    'index':  (L_FINGER_CHAINS['index'][0][0],  L_FINGER_CHAINS['index'][0][1],
               L_FINGER_CHAINS['index'][1][0],  -1),
    'middle': (L_FINGER_CHAINS['middle'][0][0], L_FINGER_CHAINS['middle'][0][1],
               L_FINGER_CHAINS['middle'][1][0], -1),
    'ring':   (L_FINGER_CHAINS['ring'][0][0],   L_FINGER_CHAINS['ring'][0][1],
               L_FINGER_CHAINS['ring'][1][0],   -1),
    'pinky':  (L_FINGER_CHAINS['pinky'][0][0],  L_FINGER_CHAINS['pinky'][0][1],
               L_FINGER_CHAINS['pinky'][1][0],  -1),
}
_R_NON_THUMB: dict[str, tuple] = {
    'index':  (R_FINGER_CHAINS['index'][0][0],  R_FINGER_CHAINS['index'][0][1],
               R_FINGER_CHAINS['index'][1][0],  +1),
    'middle': (R_FINGER_CHAINS['middle'][0][0], R_FINGER_CHAINS['middle'][0][1],
               R_FINGER_CHAINS['middle'][1][0], +1),
    'ring':   (R_FINGER_CHAINS['ring'][0][0],   R_FINGER_CHAINS['ring'][0][1],
               R_FINGER_CHAINS['ring'][1][0],   +1),
    'pinky':  (R_FINGER_CHAINS['pinky'][0][0],  R_FINGER_CHAINS['pinky'][0][1],
               R_FINGER_CHAINS['pinky'][1][0],  +1),
}

_L_THUMB_CHAIN = L_FINGER_CHAINS['thumb']
_R_THUMB_CHAIN = R_FINGER_CHAINS['thumb']


def _finger_ik_1dof(
    p_des_in_handbase: np.ndarray,
    origin_xyz: list,
    origin_rpy: list,
    link_vec: list,
    axis_sign: int,
    q_lo: float,
    q_hi: float,
) -> float:
    """
    Analytical IK for a 1-DOF finger (index/middle/ring/pinky).

    The proximal joint rotates about its local Z with sign `axis_sign`.
    The fingertip = intermediate joint origin = proximal_child_frame
    displaced by `link_vec` and then rotated by the same joint angle (mimic 1.0).

    Because the intermediate joint is a FIXED frame step (no extra rotation),
    the fingertip in the PROXIMAL BODY FRAME is just:
        p_tip_local = Rz(axis_sign * q) @ link_vec

    and in hand_base frame:
        p_tip = p1 + R1_fixed @ p_tip_local

    Inversion:
        v = R1_fixed.T @ (p_des - p1)            (target in proximal body frame)
        q = -axis_sign * atan2(px*vy - py*vx,
                                px*vx + py*vy)

    where (px,py) = link_vec[:2]. Project v onto the reachable XY circle if
    the target is out of reach.
    """
    p1 = np.asarray(origin_xyz, dtype=float)
    R1 = rpy2R_urdf(origin_rpy)
    lv = np.asarray(link_vec, dtype=float)

    v = R1.T @ (p_des_in_handbase - p1)

    px, py = lv[0], lv[1]
    vx, vy = v[0], v[1]

    r_v = np.hypot(vx, vy)
    r_lv = np.hypot(px, py)

    if r_v < 1e-9 or r_lv < 1e-9:
        return 0.0

    # project v onto the reachable circle (radius = r_lv)
    vx = vx * (r_lv / r_v)
    vy = vy * (r_lv / r_v)

    cross = px * vy - py * vx
    dot   = px * vx + py * vy

    angle_to_v = np.arctan2(cross, dot)
    q = angle_to_v / axis_sign

    return float(np.clip(q, q_lo, q_hi))


def _thumb_ik_2dof(
    p_des_in_handbase: np.ndarray,
    thumb_chain: list,
    axis_sign_yaw: int,
    axis_sign_pitch: int,
    yaw_lo: float,
    yaw_hi: float,
    pitch_lo: float,
    pitch_hi: float,
) -> tuple[float, float]:
    """
    Analytical thumb IK (yaw + pitch).

    Step 1 --- Yaw:
        Transform target to yaw-joint body frame, solve azimuth angle.

    Step 2 --- Pitch:
        With yaw fixed, transform target to pitch-joint body frame.
        The effective tip distance from the pitch-joint origin is:
            r(p) = ||[l1*cos(p) + l2*cos(2.6p),
                      l1*sin(p) + l2*sin(2.6p)]||

        Solve via law-of-cosines approximation then 1-step Newton refinement.
    """
    yaw_xyz,   yaw_rpy,   yaw_ax,   _ = thumb_chain[0]
    pitch_xyz, pitch_rpy, pitch_ax, _ = thumb_chain[1]
    inter_xyz, inter_rpy, inter_ax, _ = thumb_chain[2]
    dist_xyz,  dist_rpy,  dist_ax,  _ = thumb_chain[3]

    p_yaw = np.asarray(yaw_xyz, dtype=float)
    R_yaw = rpy2R_urdf(yaw_rpy)

    p_pitch_local = np.asarray(pitch_xyz, dtype=float)
    R_pitch = rpy2R_urdf(pitch_rpy)

    l1 = float(np.linalg.norm(inter_xyz))
    l2 = float(np.linalg.norm(dist_xyz))

    m1, m2 = 1.0, 2.6
    dm = m2 - m1

    v_yaw_body = R_yaw.T @ (p_des_in_handbase - p_yaw)

    if np.hypot(v_yaw_body[0], v_yaw_body[1]) < 1e-6:
        q_yaw = 0.0
    else:
        ref_angle = np.arctan2(p_pitch_local[1], p_pitch_local[0])
        tgt_angle = np.arctan2(v_yaw_body[1],     v_yaw_body[0])
        angle_diff = tgt_angle - ref_angle
        q_yaw = angle_diff / axis_sign_yaw

    q_yaw = float(np.clip(q_yaw, yaw_lo, yaw_hi))

    T_yaw_joint  = _T(R_yaw, p_yaw)
    Rz_yaw       = _Rz(-q_yaw)
    T_yaw_child  = T_yaw_joint @ _T(Rz_yaw, [0.0, 0.0, 0.0])
    T_pitch_joint = T_yaw_child @ _T(R_pitch, p_pitch_local)

    p_pitch_hb    = T_pitch_joint[:3, 3]
    R_pitch_world = T_pitch_joint[:3, :3]
    v_pitch_body  = R_pitch_world.T @ (p_des_in_handbase - p_pitch_hb)

    r_des = float(np.hypot(v_pitch_body[0], v_pitch_body[1]))

    r_des_sq = r_des ** 2
    cos_arg  = (r_des_sq - l1**2 - l2**2) / (2.0 * l1 * l2 + 1e-12)
    cos_arg  = float(np.clip(cos_arg, -1.0, 1.0))
    p_init   = np.arccos(cos_arg) / dm

    for _ in range(4):
        r_p    = np.sqrt(l1**2 + l2**2 + 2*l1*l2*np.cos(dm*p_init)) + 1e-30
        dr_dp  = -2*l1*l2*dm*np.sin(dm*p_init) / (2*r_p + 1e-30)
        residual = r_p - r_des
        if abs(dr_dp) < 1e-10:
            break
        p_init -= residual / dr_dp

    q_pitch = float(np.clip(p_init, pitch_lo, pitch_hi))
    return q_yaw, q_pitch


@dataclass
class FingerIKResult:
    q_sol:             np.ndarray
    q_hand6_L:         np.ndarray
    q_hand6_R:         np.ndarray
    roundtrip_errors:  dict

# Remaining helper and public functions omitted here for brevity in the source
# archival above (the thesis appendix truncates the listing at this point).
# Insert the rest of the original ik_fingers_analytical.py source if a full
# verbatim archival listing is required.
```

## 11. Notes

- The non-thumb fingers use exact geometric inversion in the proximal frame, with radial
  projection for unreachable targets.
- The thumb is handled separately because it has coupled yaw and pitch behavior.
- Joint limits are enforced with clipping after each analytical solve.
- The module includes optional FK validation and a direct curl-to-joint mapping utility
  (referenced elsewhere as `ik_fingers_from_curl`, used by `lift_fingerPH.py`'s segment
  scheduler for the finger-curl-fraction control mode).

## Known gap vs. this repo

As of this repo's current state, `robotics/fk_fingers.py`, `robotics/dh_table.py`,
`robotics/dh_kinematics.py`, `robotics/ik_fingers_analytical.py`, and `robotics/robot_indices.py`
are imported by `scripts/g1_control/lift_fingerPH.py` but are **not present** in this
repository's tree — this reference document describes their intended contents so the import
surface can be reconstructed or verified against, not code that is currently checked in here.
