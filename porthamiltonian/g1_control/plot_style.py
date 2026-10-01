# plot_style.py
"""Shared plotting conventions for lift_fingerPH.py and analyze_controller.py.

Both files independently re-typed the same color/line-style dict, the same
force/position segment-name sets, the same mode-shading helpers, and near-
identical saturation-duty math. This module is the single source of truth
for all of that -- pure numpy + matplotlib, no Isaac/torch/URDF dependency,
so it's safe to import from either the live sim script or the offline
analysis script.

Convention (see either caller's docstring for the full rationale): solid
blue = actual/delivered, dashed orange = reference/target/ideal, dotted
green = a component breakdown (not a target). Force-mode segments (S3-S5,
contact regulation) shade orange; position-mode segments shade gray.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.patches import Patch

# Canonical figure resolution. lift_fingerPH.py's live diagnostics previously
# saved at dpi=95 and analyze_controller.py's offline diagnostics at
# dpi=110; both now use this single value for visual consistency.
DPI = 110

# ----------------------------------------------------------------------------
# Line/color conventions
# ----------------------------------------------------------------------------
ACTUAL_STYLE     = dict(color="tab:blue",   lw=1.4, ls="-")
REF_STYLE        = dict(color="tab:orange", lw=1.4, ls="--")
ERROR_STYLE      = dict(color="tab:purple", lw=1.2, ls="-")
DELIVERED_STYLE  = dict(color="tab:blue",   lw=1.4, ls="-")
IDEAL_STYLE      = dict(color="tab:orange", lw=1.3, ls="--")
IMPEDANCE_STYLE  = dict(color="tab:green",  lw=1.1, ls=":")
FORCE_COLOR = "tab:orange"
POSITION_COLOR = "tab:gray"

# ----------------------------------------------------------------------------
# Segment / control-mode taxonomy (lift_fingerPH.py's S1..S10 segment names)
# ----------------------------------------------------------------------------
POSITION_MODE_SEGMENTS = {
    "S1-pregrasp", "S2-close-approach", "S6-release", "S7-drop",
    "S8-idle", "S9-idle", "S10-idle",
}
FORCE_MODE_SEGMENTS = {"S3-grasp-PH", "S4-lift-PH", "S5-moveair-PH"}
SEG_ORDER = ["S1-pregrasp", "S2-close-approach", "S3-grasp-PH", "S4-lift-PH",
             "S5-moveair-PH", "S6-release", "S7-drop", "S8-idle", "S9-idle", "S10-idle"]


# ----------------------------------------------------------------------------
# Mode shading
# ----------------------------------------------------------------------------
def mode_index_spans(mode_arr: Sequence[bool]) -> List[Tuple[int, int]]:
    """Contiguous (i0,i1) index spans where mode_arr is True (i1 exclusive)."""
    spans, in_span, i0 = [], False, None
    for i, m in enumerate(mode_arr):
        if m and not in_span:
            in_span, i0 = True, i
        elif not m and in_span:
            in_span = False; spans.append((i0, i))
    if in_span:
        spans.append((i0, len(mode_arr)))
    return spans


def shade_mode(ax, t_arr: np.ndarray, mode_arr: Sequence[bool]) -> None:
    """Gray background = position mode; orange = force mode, over `t_arr`."""
    ax.axvspan(t_arr[0], t_arr[-1], color=POSITION_COLOR, alpha=0.07, zorder=0)
    for i0, i1 in mode_index_spans(mode_arr):
        ax.axvspan(t_arr[i0], t_arr[i1 - 1], color=FORCE_COLOR, alpha=0.18, zorder=0)


def mode_legend_patches() -> List[Patch]:
    return [Patch(facecolor=FORCE_COLOR, alpha=0.3, label="force mode (contact regulation)"),
            Patch(facecolor=POSITION_COLOR, alpha=0.2, label="position mode (free motion)")]


# ----------------------------------------------------------------------------
# Torque-saturation duty cycle
# ----------------------------------------------------------------------------
def torque_rails(tau: np.ndarray, finger_tau_clamp: float = 0.0) -> np.ndarray:
    """(k,) per-joint torque rail: `finger_tau_clamp` if > 0, else the
    observed max|tau| per joint (auto-detect fallback -- valid whenever tau
    never exceeds its own rail by construction, e.g. post-clip logged
    torques or a rail defined as the observed max itself)."""
    if finger_tau_clamp > 0:
        return np.full(tau.shape[1], float(finger_tau_clamp))
    return np.maximum(np.max(np.abs(tau), axis=0), 1e-6)


def saturation_mask(tau: np.ndarray, rails: np.ndarray, tol: float = 0.02) -> np.ndarray:
    """(n,k) bool: True where |tau| is within `tol` fraction of its joint's rail."""
    return np.abs(tau) >= (1.0 - tol) * rails[None, :]


def saturation_duty(sat_mask: np.ndarray, sel: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-joint fraction of samples saturated, optionally restricted to `sel`
    (a boolean row-selector). NaN-filled if `sel` selects zero rows."""
    s = sat_mask[sel] if sel is not None else sat_mask
    return s.mean(axis=0) if len(s) else np.full(sat_mask.shape[1], np.nan)
