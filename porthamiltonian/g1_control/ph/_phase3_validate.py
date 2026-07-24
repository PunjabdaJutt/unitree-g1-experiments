"""Phase 3 sanity check: parse URDF, build tree, compute M(0), verify PD + mass.

Run (from repo root, under Isaac Lab python):
    python scripts/g1_control/ph/_phase3_validate.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.g1_control.ph import build_tree, crba, parse_urdf


URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "robot" / "robots" / "g1_description"
    / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
)


def main() -> int:
    print(f"URDF: {URDF_PATH}")
    urdf = parse_urdf(URDF_PATH)
    print(f"  root_link = {urdf.root_link}")
    print(f"  links     = {len(urdf.links)}")
    print(f"  joints    = {len(urdf.joints)} "
          f"(movable={len(urdf.movable_joints())})")

    tree = build_tree(urdf)
    NB = tree.nb
    print(f"MultiBodyTree: NB = {NB}")

    total_mass_urdf = sum(L.mass for L in urdf.links.values() if L.has_inertial)
    total_mass_tree = tree.total_mass()
    print(f"  total mass (URDF links)      = {total_mass_urdf:.4f} kg")
    print(f"  total mass (tree bodies)     = {total_mass_tree:.4f} kg "
          f"(should equal URDF minus root-anchored)")

    q0 = np.zeros(NB)
    M = crba(tree, q0)
    print(f"M(0) shape: {M.shape}")
    sym_err = float(np.max(np.abs(M - M.T)))
    print(f"  symmetry ||M - M.T||_inf     = {sym_err:.3e}")
    eigs = np.linalg.eigvalsh(0.5 * (M + M.T))
    print(f"  eig range                    = [{eigs.min():.6e}, {eigs.max():.6e}]")
    pd = bool(eigs.min() > 0)
    print(f"  positive definite            = {pd}")

    # Show a few diagonal entries for inspection
    print("  sample diag(M) (first 6):", np.round(np.diag(M)[:6], 6).tolist())
    return 0 if (pd and sym_err < 1e-8) else 1


if __name__ == "__main__":
    raise SystemExit(main())
