"""Spatial algebra (Featherstone) for the G1 PH layer.

Conventions (Featherstone 2008):
    - Motion vector m = [omega; v_lin]  (angular-first, 6-vector).
    - Force vector f = [tau; f_lin].
    - Motion transform X_BA maps motion expressed in frame A to frame B:
          m_B = X_BA @ m_A
    - Force dual: f_A = X_BA.T @ f_B  (since X* = X^{-T}).
    - Inertia transform: I_A = X_BA.T @ I_B @ X_BA.

URDF convention: <origin rpy="r p y" xyz="x y z"/> places child frame
at xyz in parent coords, with rotation E = Rz(yaw) @ Ry(pitch) @ Rx(roll)
mapping child basis to parent basis (v_parent = E @ v_child).
Matches `fk_hands.py::rpy2R_urdf`.
"""
from __future__ import annotations

import numpy as np


def skew(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -v[2],  v[1]],
                     [v[2],  0.0, -v[0]],
                     [-v[1], v[0],  0.0]], dtype=float)


def rpy2R(rpy) -> np.ndarray:
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), rpy=[roll,pitch,yaw]."""
    rpy = np.asarray(rpy, dtype=float).reshape(3)
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def Xrot(E: np.ndarray) -> np.ndarray:
    """Motion transform for pure rotation.

    Takes motion in 'old' coords to motion in 'new' coords, where basis
    transform is v_new = E @ v_old (so E rotates old->new basis coords).
    """
    X = np.zeros((6, 6))
    X[:3, :3] = E
    X[3:, 3:] = E
    return X


def Xtrans(r) -> np.ndarray:
    """Motion transform for pure translation: new origin at position r in old coords."""
    r = np.asarray(r, dtype=float).reshape(3)
    X = np.eye(6)
    X[3:, :3] = -skew(r)
    return X


def X_child_from_parent(E_parent_from_child: np.ndarray, p_parent) -> np.ndarray:
    """URDF-style joint transform.

    Given: child frame sits at position p_parent in parent coords, with
    rotation E_parent_from_child (so v_parent = E @ v_child, as in URDF).
    Returns motion transform mapping parent coords -> child coords.
    """
    return Xrot(E_parent_from_child.T) @ Xtrans(p_parent)


def spatial_inertia(m: float, c, I_c: np.ndarray) -> np.ndarray:
    """Spatial inertia (6x6) about link-frame origin.

    Inputs: mass m, COM c (3-vec in link frame), inertia tensor I_c (3x3)
    about COM, expressed in link orientation.
    """
    c = np.asarray(c, dtype=float).reshape(3)
    C = skew(c)
    I6 = np.zeros((6, 6))
    I6[:3, :3] = I_c + m * (C @ C.T)
    I6[:3, 3:] = m * C
    I6[3:, :3] = m * C.T
    I6[3:, 3:] = m * np.eye(3)
    return I6


def X_J_revolute(axis, q: float) -> np.ndarray:
    """Motion transform for revolute joint: pre-joint -> post-joint frame.

    Post frame = pre frame rotated by angle q about `axis` (unit vector,
    expressed in joint/pre frame). Components in post = R(axis, -q) applied
    to components in pre.
    """
    u = np.asarray(axis, dtype=float).reshape(3)
    n = np.linalg.norm(u)
    if n < 1e-12:
        return np.eye(6)
    u = u / n
    c, s = np.cos(q), np.sin(q)
    K = skew(u)
    E_post_from_pre = np.eye(3) - s * K + (1.0 - c) * (K @ K)
    return Xrot(E_post_from_pre)


def S_revolute(axis) -> np.ndarray:
    """Motion subspace for a revolute joint about `axis` (in joint/body frame)."""
    u = np.asarray(axis, dtype=float).reshape(3)
    n = np.linalg.norm(u)
    if n < 1e-12:
        raise ValueError("Zero-norm revolute axis.")
    u = u / n
    S = np.zeros(6)
    S[:3] = u
    return S


def S_prismatic(axis) -> np.ndarray:
    u = np.asarray(axis, dtype=float).reshape(3)
    n = np.linalg.norm(u)
    if n < 1e-12:
        raise ValueError("Zero-norm prismatic axis.")
    u = u / n
    S = np.zeros(6)
    S[3:] = u
    return S


def X_J_prismatic(axis, q: float) -> np.ndarray:
    u = np.asarray(axis, dtype=float).reshape(3)
    n = np.linalg.norm(u)
    if n < 1e-12:
        return np.eye(6)
    return Xtrans((q / n) * u)
