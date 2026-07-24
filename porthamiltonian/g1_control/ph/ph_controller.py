"""Phase 5 — PH-based finger controller.

Provides `PHFingerController`, a passivity-based controller acting on the
HAND12 proximal finger DOFs (the PH control port `B`). It plugs into the
torque-injection site at
[lift_fingerTorque.py:1100](scripts/g1_control/lift_fingerTorque.py#L1100)
via `robot.set_joint_effort_target(tau)`.

Control law (gravity-compensated PID == IDA-PBC with quadratic potential shaping
on the controlled subspace and integral augmentation):

    dz/dt = e = q_c - q_c*
    u = B^T (dU/dq)  -  Kp (q_c - q_c*)  -  Kd qdot_c  -  Ki z

Augmented storage function (passivity certificate):
    H̃(q, p, z) = H_d(q, p) + 1/2 z^T Ki z
with H_d(q, p) = 1/2 p^T M(q)^{-1} p + U_nc(q) + 1/2 e^T Kp e.
With Kp, Kd, Ki >= 0 one shows dH̃/dt <= u_ext^T y_ext, so the augmented
closed loop remains passive w.r.t. the remaining (arm/waist) ports.
See: Ortega et al., IEEE TAC 2002 (doi:10.1109/9.993poker).

Index mapping:
- `ph.tree.joint_names`        : 53 tree DOFs (all movable URDF joints).
- `sim_joint_names`            : simulator articulation order (num_joints).
- HAND12 appears in both; we resolve sim<->tree indices by name.
- Tree joints missing from sim (e.g. finger intermediate/distal if sim URDF
  is reduced) default to 0 when reading q for FK/U.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .ph_system import PHSystem


@dataclass
class PHFingerController:
    ph: PHSystem
    sim_joint_names: Sequence[str]
    Kp: np.ndarray                          # (nu,)
    Kd: np.ndarray                          # (nu,)
    Ki: np.ndarray | None = None            # (nu,) integral gain; None → zeros
    i_clamp: float = 0.35                   # symmetric clamp applied to each z_k
    q_des_hand12: np.ndarray | None = None  # (nu,), defaults to zeros

    # resolved at __post_init__
    hand12_sim_indices: List[int] = field(init=False)
    hand12_tree_indices: List[int] = field(init=False)
    tree_to_sim_pairs: List[Tuple[int, int]] = field(init=False)
    missing_in_sim: List[str] = field(init=False)
    _z: np.ndarray = field(init=False)      # integral state (nu,)

    def __post_init__(self) -> None:
        nu = len(self.ph.control_joint_names)
        self.Kp = np.asarray(self.Kp, float).reshape(-1)
        self.Kd = np.asarray(self.Kd, float).reshape(-1)
        Ki_arr: np.ndarray = (
            np.zeros(nu) if self.Ki is None
            else np.asarray(self.Ki, float).reshape(-1)
        )
        self.Ki = Ki_arr
        if self.Kp.size != nu or self.Kd.size != nu or Ki_arr.size != nu:
            raise ValueError(f"Kp/Kd/Ki must have size {nu}")
        if np.any(self.Kp < 0) or np.any(self.Kd < 0) or np.any(Ki_arr < 0):
            raise ValueError("Kp, Kd, Ki must be non-negative for passivity")
        self._z = np.zeros(nu)
        self.q_des_hand12 = (
            np.zeros(nu) if self.q_des_hand12 is None
            else np.asarray(self.q_des_hand12, float).reshape(-1)
        )
        if self.q_des_hand12.size != nu:
            raise ValueError(f"q_des_hand12 must have size {nu}")

        sim = list(self.sim_joint_names)
        sim_name_to_idx = {n: i for i, n in enumerate(sim)}
        tree_names = self.ph.tree.joint_names

        # HAND12 must be present in both
        missing_in_sim_hand12 = [n for n in self.ph.control_joint_names if n not in sim_name_to_idx]
        if missing_in_sim_hand12:
            raise ValueError(f"HAND12 joints missing from simulator: {missing_in_sim_hand12}")

        self.hand12_sim_indices = [sim_name_to_idx[n] for n in self.ph.control_joint_names]
        self.hand12_tree_indices = list(self.ph.control_indices)

        self.tree_to_sim_pairs = [
            (ti, sim_name_to_idx[n]) for ti, n in enumerate(tree_names) if n in sim_name_to_idx
        ]
        self.missing_in_sim = [n for n in tree_names if n not in sim_name_to_idx]

    @property
    def integral_state(self) -> np.ndarray:
        """Read-only view of the current integral state z (nu,)."""
        return self._z

    # -------------------------------------------------------------- mapping
    def sim_to_tree_q(self, q_sim: np.ndarray) -> np.ndarray:
        """Lift sim-ordered q (num_joints,) to tree-ordered (NB,). Missing tree
        DOFs default to 0 (consistent with the reduced-URDF rest pose)."""
        q_tree = np.zeros(self.ph.tree.nb)
        for ti, si in self.tree_to_sim_pairs:
            q_tree[ti] = q_sim[si]
        return q_tree

    # ----------------------------------------------- restricted gravity comp
    def gravity_torques_hand12(self, q_sim: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """(dU/dq) evaluated only at HAND12 tree indices, via central FD on U.

        O(nu) U-evaluations == O(12) FK walks; much cheaper than the full
        NB-wide grad_H_q used in Phase 4 validation.
        """
        q_tree = self.sim_to_tree_q(q_sim)
        g = np.zeros(len(self.hand12_tree_indices))
        U0_plus_cache = {}
        for k, i in enumerate(self.hand12_tree_indices):
            qp = q_tree.copy(); qp[i] += eps
            qm = q_tree.copy(); qm[i] -= eps
            g[k] = (self.ph.U(qp) - self.ph.U(qm)) / (2.0 * eps)
        return g

    # ------------------------------------------------------------- control
    def reset_integral(self) -> None:
        """Zero the integral state z. Call whenever q_des_hand12 changes abruptly."""
        self._z[:] = 0.0

    def compute(self, q_sim: np.ndarray, qdot_sim: np.ndarray, dt: float = 0.0) -> np.ndarray:
        """Return tau_hand12 (nu,) in HAND12 order (matches ph.control_joint_names).

        Updates the integral state z += e * dt in-place before returning.
        Pass dt=0 to disable integral action for this step (e.g. during warmup).
        """
        q_sim = np.asarray(q_sim, float).reshape(-1)
        qdot_sim = np.asarray(qdot_sim, float).reshape(-1)
        q_c = q_sim[self.hand12_sim_indices]
        qd_c = qdot_sim[self.hand12_sim_indices]
        e = q_c - self.q_des_hand12
        if dt > 0.0:
            self._z = np.clip(self._z + e * dt, -self.i_clamp, self.i_clamp)
        g_c = self.gravity_torques_hand12(q_sim)
        return g_c - self.Kp * e - self.Kd * qd_c - self.Ki * self._z

    def scatter_to_full(self, tau_hand12: np.ndarray, num_joints: int) -> np.ndarray:
        """Place (nu,) torques at their sim indices in a fresh (num_joints,) vector."""
        out = np.zeros(num_joints)
        for k, si in enumerate(self.hand12_sim_indices):
            out[si] = tau_hand12[k]
        return out

    # ---------------------------------------torch glue
    def compute_torch(self, q_sim_t, qdot_sim_t, dt: float = 0.0):
        """Single-env torch convenience. Input / output on the same device+dtype.

        Caller is responsible for selecting a single env row before calling
        (or wrap this in a per-env loop). Kept single-env to keep the
        numpy core clean.
        """
        import torch
        q_np = q_sim_t.detach().cpu().numpy()
        qd_np = qdot_sim_t.detach().cpu().numpy()
        tau_np = self.compute(q_np, qd_np, dt=dt)
        return torch.as_tensor(tau_np, dtype=q_sim_t.dtype, device=q_sim_t.device)
