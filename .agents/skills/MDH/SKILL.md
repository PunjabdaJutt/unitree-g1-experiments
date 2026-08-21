---
name: MDH
description: Reference for Unitree G1 Inspire-Hand DFQ finger kinematics — Modified DH (Craig 1989) parameter tables, the URDF-chain-is-source-of-truth architecture (fk_fingers.py vs dh_table.py vs dh_kinematics.py), the 137mm Y-offset MDH limitation, and closed-form analytical IK for the HAND12 proximal joints. Use when writing, debugging, or reviewing code touching G1 finger forward/inverse kinematics, DH tables, fingertip Jacobians, or the hand-base kinematic chain.
---

# G1 Inspire-Hand MDH & Analytical IK

Converted from `MSc_Thesis_Ammar_Akhlaq_Ahmed_Humanoid_Robot_g1_Port_hamiltonian_Control (2).pdf`
(this skill's directory) — the thesis appendix documenting the Modified DH tables and the
closed-form finger IK for the Unitree G1 + Inspire-Hand DFQ.

## When to apply

- Implementing, debugging, or reviewing `fk_fingers.py`, `dh_table.py`, `dh_kinematics.py`,
  or `ik_fingers_analytical.py` (or any equivalent finger FK/IK code).
- Answering questions about MDH parameters (a, α, d, θ_off) for specific G1 hand joints.
- Reviewing controller code that calls a fingertip Jacobian / `DHFingertipKinematics.fk_tips()`
  or does task-space impedance control on the fingers.
- Explaining or sanity-checking the ~137mm anomaly in the `a` column of the MDH table.
- Cross-checking `ph_system.py` (Featherstone-tree FK, used by the PH pipeline — see the
  `ph_pipeline_guide` reference in this repo) against the URDF-chain FK used by the hand
  controller — these are two independent FK implementations and must not be silently mixed.

## Key facts

- **Convention**: Craig's Modified DH (1989):
  `T_{i-1→i} = Rx(α_{i-1}) · Tx(a_{i-1}) · Rz(θ_i + θ_off) · Tz(d_i)`.
  Base frame is `{L|R}_hand_base_link` (wrist yaw + fixed URDF offset).
- **The URDF chain is the single source of truth for FK — not the MDH table.**
  - `fk_fingers.py` is ground truth: hardcodes URDF joint origins/axes/mimic multipliers and
    composes `T ← T · Fixed(xyz, rpy) · Rotate(axis, q)` directly. Verified to < 0.001 mm.
  - `dh_table.py` *derives* an MDH table from the same URDF data, for inspection/sanity-check
    only — it is never used to compute FK.
  - `dh_kinematics.py` exposes `fk_dh()`, but internally it re-walks the URDF chain
    (`fk_dh() → _fk_urdf_chain()`); it does **not** consume the MDH rows either.
- **Why the MDH table can't drive FK**: MDH can only represent frame offsets along X (via `a`)
  and Z (via `d`) of the previous frame. The real proximal-joint offset has a component along
  Y (~137mm, exact vector `[0.00028533, -0.1371, 0.01295]` in the hand-base frame). Using the
  MDH rows for FK would introduce a systematic ~137mm position error. The `a = 137.10mm` entry
  for the middle-finger proximal joint is a **projection artifact** of the canonical
  common-perpendicular construction, not real geometry — don't treat it as physical.
- **Mimic joints** are not independently actuated; angle = URDF `<mimic>` multiplier ×
  parent proximal joint angle. Scale factors: 1.0× (index/middle/ring/pinky intermediate),
  1.6× (thumb intermediate), 2.4× (thumb distal).
- **HAND12** = the 12 actuated proximal/thumb-base joints (6 per hand: thumb_yaw, thumb_pitch,
  index, middle, ring, pinky). Isaac Lab sim indices: Left `[29,30,33,35,37,39]`,
  Right `[41,42,45,47,49,51]`.
- **Analytical IK** (`ik_fingers_analytical.py`) replaces a DLS iterative loop:
  - Non-thumb fingers: 1 DOF each, exact closed-form via circle projection in the proximal
    body frame (`q = -atan2(px·vy - py·vx, px·vx + py·vy)`); radially projects unreachable
    targets onto the reachable circle.
  - Thumb: 2 DOF (yaw + pitch), solved in two stages — yaw from target azimuth in the
    yaw-joint frame, then pitch via a law-of-cosines initial guess + ≤4 Newton iterations
    (always converges) on the effective 3-link chain (pitch + 1.6×pitch + 2.4×pitch).
  - Joint limits enforced by clipping after each analytical solve.
- **Consistency guarantees validated in the thesis**: `fk_dh` matches `fk_fingers` to
  < 0.001 mm over 200 random configurations; all MDH rows are internally self-consistent to
  < 4×10⁻⁹ rad/m.
- **Impedance-control implication**: `τ = J_c^T (K_t(x_d - x) + ...)`. The Jacobian and the FK
  it differentiates must come from the *same* implementation (the URDF chain). Do not mix this
  with the Featherstone-tree FK from `ph_system.py` for the same joints — that inconsistency
  breaks numerical stability of the impedance term.

## Reference details

- Full MDH parameter tables (all fingers, both hands, with notes): `references/mdh_parameter_tables.md`
- MDH extraction math (canonical frame construction, per-parameter formulas), the 137mm
  limitation writeup, and the full analytical-IK derivation + Python source
  (`ik_fingers_analytical.py`): `references/analytical_ik_fingers.md`
