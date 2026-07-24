"""Port-Hamiltonian system wrapper around the G1 multibody tree.

State  : x = [q; p] ∈ R^{2 NB},  p = M(q) qdot.
Energy : H(q, p) = 1/2 p^T M(q)^{-1} p + U(q).
Form   : xdot = (J - R) ∇H(x) + G u
         J = [[0, I],[-I, 0]]   (skew),
         R = blkdiag(0, D)      (PSD joint damping),
         G = [[0],[B]]          B picks controlled DOFs (HAND12 by default).

U(q) is the gravitational potential of the absorbed-link mass distribution,
evaluated via a Featherstone-style forward walk on the tree.

∇_q H is computed by finite differences here (cheap, O(NB) solves). The
RNEA-based analytic gradient is a deferred item (see Phase 3 memory).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .multibody_tree import MultiBodyTree, crba, _X_J


def _unskew(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def _Ep_from_X(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (E, r) from a Plücker motion transform X_B_from_A.

    Conventions: X = Xrot(E) @ Xtrans(r), so E is the rotation A->B and r is
    the origin of frame B expressed in A.
    """
    E = X[:3, :3]
    skew_r = -E.T @ X[3:, :3]
    r = _unskew(skew_r)
    return E, r


@dataclass
class PHSystem:
    tree: MultiBodyTree
    control_joint_names: Sequence[str]
    damping: np.ndarray | None = None       # (NB,) joint damping, default 0
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Cached, derived fields
    mass: np.ndarray = field(init=False)      # (NB,)
    com: np.ndarray = field(init=False)       # (NB, 3) in body frame
    D: np.ndarray = field(init=False)         # (NB, NB) diagonal
    control_indices: List[int] = field(init=False)
    B: np.ndarray = field(init=False)         # (NB, nu)

    def __post_init__(self):
        NB = self.tree.nb
        self.mass = np.zeros(NB)
        self.com = np.zeros((NB, 3))
        for i, Ii in enumerate(self.tree.I):
            m = float(np.trace(Ii[3:, 3:]) / 3.0)
            self.mass[i] = m
            if m > 1e-12:
                self.com[i] = _unskew(Ii[0:3, 3:6]) / m

        d = np.zeros(NB) if self.damping is None else np.asarray(self.damping, float).reshape(-1)
        if d.size != NB:
            raise ValueError(f"damping size {d.size} != NB {NB}")
        self.D = np.diag(d)

        name_to_idx = {n: i for i, n in enumerate(self.tree.joint_names)}
        missing = [n for n in self.control_joint_names if n not in name_to_idx]
        if missing:
            raise ValueError(f"control joints missing from tree: {missing}")
        self.control_indices = [name_to_idx[n] for n in self.control_joint_names]
        nu = len(self.control_indices)
        B = np.zeros((NB, nu))
        for k, i in enumerate(self.control_indices):
            B[i, k] = 1.0
        self.B = B

    # ------------------------------------------------------------------ FK
    def forward_kinematics(self, q) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Per-body (E_body_from_world, r_body_origin_in_world)."""
        NB = self.tree.nb
        q = np.asarray(q, float).reshape(-1)
        X_bw: List[np.ndarray] = [None] * NB  # type: ignore
        poses: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(NB):
            Xi = _X_J(self.tree, i, float(q[i])) @ self.tree.Xtree[i]
            p = self.tree.parent[i]
            Xbw = Xi if p < 0 else Xi @ X_bw[p]
            X_bw[i] = Xbw
            poses.append(_Ep_from_X(Xbw))
        return poses

    # ----------------------------------------- energies
    def U(self, q) -> float:
        g = np.asarray(self.gravity, float)
        total = 0.0
        for i, (E, r) in enumerate(self.forward_kinematics(q)):
            m = self.mass[i]
            if m <= 0.0:
                continue
            com_world = E.T @ self.com[i] + r
            total -= m * (g @ com_world)
        return float(total)

    def M(self, q) -> np.ndarray:
        return crba(self.tree, q)

    def H(self, q, p) -> float:
        M = self.M(q)
        v = np.linalg.solve(M, p)
        return float(0.5 * p @ v + self.U(q))

    # -------------------------------------------------- gradients of H
    def grad_H_p(self, q, p) -> np.ndarray:
        """∇_p H = M(q)^{-1} p = qdot."""
        return np.linalg.solve(self.M(q), np.asarray(p, float))

    def grad_H_q(self, q, p, eps: float = 1e-6) -> np.ndarray:
        """∇_q H by central finite differences.

        Analytic form needs ∂M^{-1}/∂q (equivalently RNEA Coriolis + gravity
        torques). RNEA is deferred; this is adequate for validation and
        moderate-rate simulation.
        """
        q = np.asarray(q, float).reshape(-1).copy()
        NB = q.size
        g = np.zeros(NB)
        for i in range(NB):
            qp = q.copy(); qp[i] += eps
            qm = q.copy(); qm[i] -= eps
            g[i] = (self.H(qp, p) - self.H(qm, p)) / (2.0 * eps)
        return g

    # ----------------------------------------------- structural matrices
    def J_struct(self) -> np.ndarray:
        NB = self.tree.nb
        Z = np.zeros((NB, NB))
        I = np.eye(NB)
        return np.block([[Z, I], [-I, Z]])

    def R_struct(self) -> np.ndarray:
        NB = self.tree.nb
        Z = np.zeros((NB, NB))
        return np.block([[Z, Z], [Z, self.D]])

    def G_struct(self) -> np.ndarray:
        NB = self.tree.nb
        Z = np.zeros((NB, self.B.shape[1]))
        return np.vstack([Z, self.B])

    # -------------------------------------------------------- dynamics
    def dynamics(self, q, p, u) -> Tuple[np.ndarray, np.ndarray]:
        """Return (qdot, pdot) = (J-R) ∇H + G u."""
        q = np.asarray(q, float).reshape(-1)
        p = np.asarray(p, float).reshape(-1)
        u = np.asarray(u, float).reshape(-1)
        qdot = self.grad_H_p(q, p)                # from ∇_p H
        dH_dq = self.grad_H_q(q, p)
        pdot = -dH_dq - self.D @ qdot + self.B @ u
        return qdot, pdot

    # ---------------------------------------------------------- power
    def power_balance(self, q, p, u) -> dict:
        """Instantaneous energy bookkeeping (should hold up to FD noise)."""
        qdot = self.grad_H_p(q, p)
        P_in = float((self.B @ u) @ qdot)         # control port power
        P_diss = float(qdot @ self.D @ qdot)      # dissipated (>=0)
        dH_dt = P_in - P_diss
        return dict(P_in=P_in, P_diss=P_diss, dH_dt=dH_dt)


