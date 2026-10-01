#!/usr/bin/env python3
"""
plot_arm_position_single_run.py
================================

Single-run replacement for the old 5-runs-overlaid "Figure 14" (right-arm
raw position timeline). Same convention as fig03b_armB_position_timeline in
analyze_real_logs.py (segbands highlighting grasp_hold + side_down), but
ONE run only (main_185109) and reflowed to single-column width, matching
fig13_handB_single_run_2panel's single-run treatment on the hand side.

Raw position only (not tracking error -- that's the separate fig:hwarmtracking
figure elsewhere in the paper), all 7 right-arm joints, one subplot each,
consolidated into a single figure.

Usage: python Real_logs/plot_arm_position_single_run.py [out_dir]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_CSV = "arm_positions_20260918_185109.csv"

RIGHT_ARM = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
             "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
RIGHT_ARM_SHORT = [j.replace("right_", "") for j in RIGHT_ARM]
SEG_COLORS = {"grasp_hold": "#f3d5b5", "side_down": "#a4c9d9"}
DEFAULT_SEG_COLOR = "#f0f0f0"
W1 = 3.50  # IEEE single-column width [in]


def paper_style():
    serif = [f for f in ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
             if f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}]
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": serif or ["DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.4, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def segbands(ax, t, seg, seg_colors, default_color):
    sw = np.where(seg[1:] != seg[:-1])[0] + 1
    b = [0] + list(sw) + [len(t)]
    for i in range(len(b) - 1):
        a, e = b[i], b[i + 1]
        c = seg_colors.get(seg[a], default_color)
        ax.axvspan(t[a], t[min(e, len(t) - 1)], color=c, alpha=.55, lw=0, zorder=0)
    for k in sw:
        ax.axvline(t[k], color="k", lw=.35, alpha=.35, zorder=1)


OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "real_logs_analysis")
PDF, PNG = (os.path.join(OUT, s) for s in ("figures_pdf", "figures_png"))
for d in (PDF, PNG):
    os.makedirs(d, exist_ok=True)

paper_style()
df = pd.read_csv(os.path.join(HERE, RUN_CSV))
t = df["t_s"].to_numpy(float)
seg = df["segment"].to_numpy()

fig, axs = plt.subplots(4, 2, figsize=(W1, 5.4), sharex=False)
fig.subplots_adjust(hspace=0.75, wspace=0.35)
for k, (j, js) in enumerate(zip(RIGHT_ARM, RIGHT_ARM_SHORT)):
    ax = axs.flat[k]
    segbands(ax, t, seg, SEG_COLORS, DEFAULT_SEG_COLOR)
    ax.plot(t, df[j].to_numpy(float), color="#003049", lw=.8)
    ax.set_title(js, fontsize=7, pad=2)
    ax.set_xlabel("Time [s]", fontsize=6)
    ax.set_ylabel("q [rad]", fontsize=6)
fig.delaxes(axs.flat[7])

# legend proxy for the two shading colors, placed in the empty 8th grid cell
from matplotlib.patches import Patch
legend_ax = fig.add_axes([0.58, 0.06, 0.35, 0.16])
legend_ax.axis("off")
legend_ax.legend(handles=[
    Patch(facecolor=SEG_COLORS["grasp_hold"], alpha=.55, label="grasp_hold"),
    Patch(facecolor=SEG_COLORS["side_down"], alpha=.55, label="side_down"),
    Patch(facecolor=DEFAULT_SEG_COLOR, alpha=.55, label="other segments"),
], loc="center", fontsize=6, frameon=False)

fig.suptitle("Right-arm joint position, full run (main_185109)", fontsize=8)
fig.savefig(os.path.join(PNG, "fig_armB_position_single_run.png"), dpi=200)
fig.suptitle("")
fig.savefig(os.path.join(PDF, "fig_armB_position_single_run.pdf"))
plt.close(fig)

print("Wrote fig_armB_position_single_run (pdf+png) to", OUT)
