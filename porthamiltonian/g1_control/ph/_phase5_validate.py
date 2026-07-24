"""Phase 5 sanity check: PHFingerController index mapping and control law.

We synthesize a plausible sim joint ordering by taking the tree joint names
(this matches the Isaac articulation when the URDF isn't reduced). A real
Isaac run would pass `robot.data.joint_names` in.

Run:
    PYTHONPATH=. python scripts/g1_control/ph/_phase5_validate.py
"""
from __future__ import annotations

import os
import numpy as np

from scripts.g1_control.ph import (
    PHFingerController, PHSystem, build_tree, parse_urdf,
)


URDF = os.path.join(
    "scripts", "g1_control", "robot", "robots", "g1_description",
    "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf",
)

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
    ph = PHSystem(tree, control_joint_names=HAND12_NAMES)

    # Synthesize a sim joint ordering that happens to be the reverse of the
    # tree order, to make sure the index mapping isn't accidentally identity.
    sim_names = list(reversed(tree.joint_names))
    N = len(sim_names)

    Kp = 5.0 * np.ones(12)
    Kd = 0.2 * np.ones(12)
    q_des = 0.1 * np.arange(12)
    ctrl = PHFingerController(ph, sim_names, Kp, Kd, q_des_hand12=q_des)

    # -- index mapping ------------------------------------------------------
    for k, name in enumerate(HAND12_NAMES):
        assert sim_names[ctrl.hand12_sim_indices[k]] == name
        assert tree.joint_names[ctrl.hand12_tree_indices[k]] == name
    print(f"HAND12 sim indices = {ctrl.hand12_sim_indices}")
    print(f"HAND12 tree indices = {ctrl.hand12_tree_indices}")
    print(f"tree DOFs missing from sim (synth): {len(ctrl.missing_in_sim)}  (0 expected here)")
    assert len(ctrl.missing_in_sim) == 0

    # sim_to_tree_q round-trip on a random sim q
    rng = np.random.default_rng(0)
    q_sim = 0.1 * rng.standard_normal(N)
    q_tree = ctrl.sim_to_tree_q(q_sim)
    for ti, si in ctrl.tree_to_sim_pairs:
        assert q_tree[ti] == q_sim[si]
    print("sim->tree lift OK")

    # -- at q=q_des, qdot=0, control output equals g_c (gravity comp only) --
    q_sim_des = np.zeros(N)
    for k, name in enumerate(HAND12_NAMES):
        q_sim_des[ctrl.hand12_sim_indices[k]] = q_des[k]
    tau = ctrl.compute(q_sim_des, np.zeros(N))
    g_c = ctrl.gravity_torques_hand12(q_sim_des)
    err = np.linalg.norm(tau - g_c)
    print(f"tau(q=q*, qdot=0) - g_c ||  = {err:.2e}   (should be ~0)")
    assert err < 1e-12

    # -- at q != q_des, qdot != 0: tau = g_c - Kp*e - Kd*qd ----------------
    qdot_sim = 0.2 * rng.standard_normal(N)
    tau = ctrl.compute(q_sim, qdot_sim)
    q_c = q_sim[ctrl.hand12_sim_indices]
    qd_c = qdot_sim[ctrl.hand12_sim_indices]
    g_c = ctrl.gravity_torques_hand12(q_sim)
    tau_ref = g_c - Kp * (q_c - q_des) - Kd * qd_c
    err = np.linalg.norm(tau - tau_ref)
    print(f"control law residual        = {err:.2e}")
    assert err < 1e-12

    # -- scatter to full sim tau -------------------------------------------
    tau_full = ctrl.scatter_to_full(tau, N)
    assert tau_full.shape == (N,)
    for k, si in enumerate(ctrl.hand12_sim_indices):
        assert tau_full[si] == tau[k]
    # all non-HAND12 entries zero
    nonhand = [i for i in range(N) if i not in ctrl.hand12_sim_indices]
    assert np.all(tau_full[nonhand] == 0.0)
    print("scatter_to_full OK")

    # -- gravity torques at HAND12 agree with full grad_q H restriction ----
    # (full grad uses central FD on H; restricted uses central FD on U; at
    # p=0, H=U, so they must match.)
    full_grad = ph.grad_H_q(ctrl.sim_to_tree_q(q_sim), np.zeros(tree.nb))
    g_full_restricted = full_grad[ctrl.hand12_tree_indices]
    err = np.linalg.norm(g_c - g_full_restricted)
    print(f"||g_c_fast - grad_q H|HAND12|| = {err:.2e}")
    assert err < 1e-6

    # -- missing-sim case: drop one tree joint from sim and verify fallback -
    drop = tree.joint_names[0]  # e.g. left_hip_pitch
    sim_names_reduced = [n for n in sim_names if n != drop]
    ctrl2 = PHFingerController(ph, sim_names_reduced, Kp, Kd, q_des_hand12=q_des)
    assert drop in ctrl2.missing_in_sim
    q_sim2 = 0.1 * rng.standard_normal(len(sim_names_reduced))
    q_tree2 = ctrl2.sim_to_tree_q(q_sim2)
    assert q_tree2[0] == 0.0  # missing joint defaults to 0
    print(f"missing-in-sim fallback: {len(ctrl2.missing_in_sim)} tree DOFs -> 0 in lift")

    print("\nPhase 5 validation PASSED.")


if __name__ == "__main__":
    main()
