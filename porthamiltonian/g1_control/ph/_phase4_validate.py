"""Phase 4 sanity check: PHSystem construction + structural / energy checks.

Run:
    PYTHONPATH=. python scripts/g1_control/ph/_phase4_validate.py
"""
from __future__ import annotations

import os
import numpy as np

from scripts.g1_control.ph import parse_urdf, build_tree, PHSystem


URDF = os.path.join(
    "scripts", "g1_control", "robot", "robots", "g1_description",
    "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf",
)

# HAND12 control joints (proximal fingers), from IK29_JOINT_NAMES[17:].
HAND12_NAMES = [
    "L_thumb_proximal_yaw_joint",  "L_thumb_proximal_pitch_joint",
    "L_index_proximal_joint",      "L_middle_proximal_joint",
    "L_ring_proximal_joint",       "L_pinky_proximal_joint",
    "R_thumb_proximal_yaw_joint",  "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",      "R_middle_proximal_joint",
    "R_ring_proximal_joint",       "R_pinky_proximal_joint",
]


def main() -> None:
    urdf = parse_urdf(URDF)
    tree = build_tree(urdf)
    NB = tree.nb
    print(f"NB = {NB}")

    # light uniform damping so dissipation is non-trivial
    d = 0.05 * np.ones(NB)
    ph = PHSystem(tree, control_joint_names=HAND12_NAMES, damping=d)

    # -- structural matrices -------------------------------------------------
    J = ph.J_struct(); R = ph.R_struct(); G = ph.G_struct()
    assert J.shape == (2 * NB, 2 * NB)
    assert np.allclose(J + J.T, 0.0), "J not skew"
    eigR = np.linalg.eigvalsh((R + R.T) / 2)
    assert eigR.min() >= -1e-12, f"R not PSD, min eig {eigR.min():.3e}"
    assert G.shape == (2 * NB, 12)
    # B picks the right indices
    for k, n in enumerate(HAND12_NAMES):
        i = tree.joint_names.index(n)
        assert G[NB + i, k] == 1.0
    print("structural J/R/G  OK  (skew / PSD / column picks HAND12)")

    # -- energies ------------------------------------------------------------
    q = np.zeros(NB); p = np.zeros(NB)
    U0 = ph.U(q); H0 = ph.H(q, p)
    print(f"U(0)          = {U0: .4f} J")
    print(f"H(0, 0)       = {H0: .4f} J   (= U(0) since p=0)")
    assert abs(H0 - U0) < 1e-9

    # Lifting the pelvis link is not possible (fixed base); raise 1 rad at
    # the first movable joint and check U changes.
    qp = q.copy(); qp[0] = 1.0
    dU = ph.U(qp) - U0
    print(f"dU at q[0]=1  = {dU:+.4f} J   (joint {tree.joint_names[0]})")

    # -- grad_H_p = M^-1 p ---------------------------------------------------
    rng = np.random.default_rng(0)
    q_rand = 0.1 * rng.standard_normal(NB)
    # Sample p via p = M @ qdot_rand so qdot stays O(1) despite ill-conditioned M
    # (finger intermediate/distal joints have ~1e-7 inertia — tiny eigenvalues).
    qdot_rand = rng.standard_normal(NB)
    p_rand = ph.M(q_rand) @ qdot_rand
    qdot = ph.grad_H_p(q_rand, p_rand)
    err = np.linalg.norm(qdot - qdot_rand)
    print(f"||grad_p H - M^-1 p||  = {err:.2e}")
    assert err < 1e-8

    # -- power balance: dH/dt = u^T B^T qdot - qdot^T D qdot -----------------
    u = rng.standard_normal(12)
    pb = ph.power_balance(q_rand, p_rand, u)
    print(f"P_in={pb['P_in']:+.4f}  P_diss={pb['P_diss']:+.4f}  "
          f"dH/dt={pb['dH_dt']:+.4f}")
    assert pb["P_diss"] >= -1e-12

    # Numerical dH/dt along the PH vector field (small step, central diff)
    qdot, pdot = ph.dynamics(q_rand, p_rand, u)
    eps = 1e-5
    H_plus  = ph.H(q_rand + eps * qdot, p_rand + eps * pdot)
    H_minus = ph.H(q_rand - eps * qdot, p_rand - eps * pdot)
    dH_num = (H_plus - H_minus) / (2 * eps)
    print(f"dH/dt (numeric)    = {dH_num:+.4f}   "
          f"(analytic {pb['dH_dt']:+.4f}, rel err {abs(dH_num-pb['dH_dt'])/(abs(pb['dH_dt'])+1e-9):.1e})")
    assert abs(dH_num - pb["dH_dt"]) < 1e-2 * (abs(pb["dH_dt"]) + 1.0)

    # -- lossless when D=0, u=0 ---------------------------------------------
    ph0 = PHSystem(tree, control_joint_names=HAND12_NAMES, damping=None)
    p_rand0 = ph0.M(q_rand) @ qdot_rand
    qdot0, pdot0 = ph0.dynamics(q_rand, p_rand0, np.zeros(12))
    Hp = ph0.H(q_rand + eps * qdot0, p_rand0 + eps * pdot0)
    Hm = ph0.H(q_rand - eps * qdot0, p_rand0 - eps * pdot0)
    dH0 = (Hp - Hm) / (2 * eps)
    print(f"dH/dt (lossless)   = {dH0:+.2e}   (should be ~0)")
    assert abs(dH0) < 1e-2

    print("\nPhase 4 validation PASSED.")


if __name__ == "__main__":
    main()
