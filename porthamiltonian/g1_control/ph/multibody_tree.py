"""Featherstone multibody tree + CRBA for the G1 PH layer.

Builds a MultiBodyTree from a URDFModel by:
  1. Walking link topology from root_link, absorbing fixed joints into
     adjacent movable bodies (accumulating their Xtree and spatial inertia).
  2. For each movable joint: parent body index, Xtree (parent-body -> joint
     pre-frame), motion subspace S, spatial inertia I (in body frame at q=0).

CRBA (Featherstone 2008, Table 6.2) computes the joint-space mass matrix
M(q) at configuration q. Fixed-base (root = world) for this phase; the
floating-base extension is deferred to a later phase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .spatial import (
    S_prismatic,
    S_revolute,
    X_child_from_parent,
    X_J_prismatic,
    X_J_revolute,
    rpy2R,
    spatial_inertia,
)
from .urdf_parser import URDFModel


@dataclass
class MultiBodyTree:
    joint_names: List[str]           # movable joint names (body order)
    joint_types: List[str]           # per-body joint type (no "fixed")
    joint_axes: List[np.ndarray]     # (3,) per body, in joint/pre frame
    parent: List[int]                # -1 = world; otherwise body index
    Xtree: List[np.ndarray]          # (6,6) parent-body -> joint pre-frame
    I: List[np.ndarray]              # (6,6) spatial inertia in body frame
    S: List[np.ndarray]              # (6,) motion subspace
    body_link_names: List[List[str]] # URDF links absorbed into each body

    @property
    def nb(self) -> int:
        return len(self.joint_names)

    def total_mass(self) -> float:
        # Mass block of spatial inertia = m * I3; trace of lower-right = 3m.
        return float(sum(np.trace(Ii[3:, 3:]) / 3.0 for Ii in self.I))


# -----------------------------------------------------------------------------

def build_tree(urdf: URDFModel) -> MultiBodyTree:
    """Fixed-base MultiBodyTree from URDFModel.

    The root link (expected `pelvis` for G1) is treated as the inertial
    world. All links attached rigidly to root (via fixed joints, without
    any movable joint intervening) are ignored here — they don't move and
    don't contribute to joint-space dynamics.
    """
    movable = urdf.movable_joints()
    movable_child_to_body: Dict[str, int] = {j.child: i for i, j in enumerate(movable)}

    # anchor[link] = (body_idx_or_-1, X_link_from_anchorBody)
    # where X_link_from_anchorBody is the motion transform taking motion
    # in the anchor body's frame to motion in this link's frame.
    anchor: Dict[str, Tuple[int, np.ndarray]] = {urdf.root_link: (-1, np.eye(6))}

    queue = [urdf.root_link]
    while queue:
        parent_link = queue.pop(0)
        for child_link in urdf.children_of.get(parent_link, []):
            # locate the joint parent_link -> child_link
            j = next(
                jj for jj in urdf.joints
                if jj.parent == parent_link and jj.child == child_link
            )
            E = rpy2R(j.origin_rpy)  # parent_from_child rotation
            X_child_from_plink = X_child_from_parent(E, j.origin_xyz)

            if child_link in movable_child_to_body:
                # new body — its own frame is the anchor for itself at q=0.
                anchor[child_link] = (movable_child_to_body[child_link], np.eye(6))
            else:
                # fixed joint (or movable whose child link isn't listed — shouldn't happen):
                # accumulate transform from same anchor body to this link frame.
                a_idx, X_plink_from_anchor = anchor[parent_link]
                X_child_from_anchor = X_child_from_plink @ X_plink_from_anchor
                anchor[child_link] = (a_idx, X_child_from_anchor)

            queue.append(child_link)

    # Per-body fields
    parent_idx: List[int] = []
    Xtree: List[np.ndarray] = []
    S_list: List[np.ndarray] = []
    types: List[str] = []
    axes: List[np.ndarray] = []
    names: List[str] = []
    body_I = [np.zeros((6, 6)) for _ in movable]
    body_link_names: List[List[str]] = [[] for _ in movable]

    for i, j in enumerate(movable):
        # parent body is the anchor of this joint's parent link.
        p_anchor, X_plink_from_pbody = anchor[j.parent]
        parent_idx.append(p_anchor)

        E = rpy2R(j.origin_rpy)
        X_jframe_from_plink = X_child_from_parent(E, j.origin_xyz)
        Xtree_i = X_jframe_from_plink @ X_plink_from_pbody  # pbody -> joint pre-frame
        Xtree.append(Xtree_i)

        types.append(j.type)
        axes.append(j.axis.copy())
        names.append(j.name)

        if j.type in ("revolute", "continuous"):
            S_list.append(S_revolute(j.axis))
        elif j.type == "prismatic":
            S_list.append(S_prismatic(j.axis))
        else:
            raise NotImplementedError(f"Unsupported joint type: {j.type}")

    # Compose spatial inertia of each body from all links anchored to it.
    for link_name, link in urdf.links.items():
        if not link.has_inertial or link.mass <= 0.0:
            continue
        a_idx, X_L_from_anchor = anchor[link_name]
        if a_idx < 0:
            # rigidly attached to world/base; skipped (fixed base).
            continue
        I_L = spatial_inertia(link.mass, link.com, link.inertia)
        body_I[a_idx] += X_L_from_anchor.T @ I_L @ X_L_from_anchor
        body_link_names[a_idx].append(link_name)

    return MultiBodyTree(
        joint_names=names,
        joint_types=types,
        joint_axes=axes,
        parent=parent_idx,
        Xtree=Xtree,
        I=body_I,
        S=S_list,
        body_link_names=body_link_names,
    )


# -----------------------------------------------------------------------------

def _X_J(tree: MultiBodyTree, i: int, qi: float) -> np.ndarray:
    t = tree.joint_types[i]
    if t in ("revolute", "continuous"):
        return X_J_revolute(tree.joint_axes[i], qi)
    if t == "prismatic":
        return X_J_prismatic(tree.joint_axes[i], qi)
    raise NotImplementedError(t)


def crba(tree: MultiBodyTree, q) -> np.ndarray:
    """Composite Rigid Body Algorithm: joint-space mass matrix M(q).

    Returns H of shape (NB, NB), symmetric positive definite for a
    well-posed tree.
    """
    NB = tree.nb
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != NB:
        raise ValueError(f"q size {q.size} != NB {NB}")

    X_up = [_X_J(tree, i, float(q[i])) @ tree.Xtree[i] for i in range(NB)]

    IC = [Ii.copy() for Ii in tree.I]
    for i in reversed(range(NB)):
        p = tree.parent[i]
        if p >= 0:
            IC[p] = IC[p] + X_up[i].T @ IC[i] @ X_up[i]

    H = np.zeros((NB, NB))
    for i in range(NB):
        F = IC[i] @ tree.S[i]
        H[i, i] = float(tree.S[i] @ F)
        j = i
        while tree.parent[j] >= 0:
            F = X_up[j].T @ F
            j = tree.parent[j]
            H[i, j] = float(F @ tree.S[j])
            H[j, i] = H[i, j]
    return H
