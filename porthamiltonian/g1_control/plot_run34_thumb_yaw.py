#!/usr/bin/env python3
"""
plot_run34_thumb_yaw.py
========================

Paper-quality figure for Section VI: right-hand thumb_yaw failing to reach a
stable force-mode equilibrium in the sim grasp (run34). Two stacked panels
(position, torque), single joint only, single-column width -- same
conventions/style as Real_logs/plot_hand_single_run_2panel.py's fig13/fig14.

Background: run34 (logs/robot_ph_run34_q_tau_29.csv, lift_fingerPH.py output,
25mm-wide cylinder target) shows the grasp succeeding overall (object height
rises and stays elevated through S4-S5; contact_active fires through both
segments), and the four non-thumb right-hand fingers reach a genuine contact
equilibrium (torque rises then relaxes during S5). thumb_yaw does not: it
sits pinned at its 1.3 rad mechanical limit for nearly the whole S3-S5 span
while its torque climbs monotonically with no sign of settling, peaking at
the very last S5 sample. This figure exists to show that specific behavior,
not a "fingers don't close" claim (they do -- see the position panel of
every OTHER finger, not shown here).

Usage: python scripts/g1_control/plot_run34_thumb_yaw.py [out_dir]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(REPO_ROOT, "logs", "robot_ph_run34_q_tau_29.csv")

JOINT = "R_thumb_proximal_yaw_joint"
THUMB_YAW_LIMIT_HI_RAD = 1.3  # HAND12_LIMITS_HI[0], ph/robot_indices.py -- mechanical max
FORCE_SEGMENTS = {"S3-grasp-PH", "S4-lift-PH", "S5-moveair-PH"}
W1 = 3.50  # IEEE single-column width [in], matches fig13/fig14


def paper_style():
    serif = [f for f in ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
             if f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}]
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": serif or ["DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 6,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.4, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def segbands(ax, t, seg, force_segs):
    sw = np.where(seg[1:] != seg[:-1])[0] + 1
    b = [0] + list(sw) + [len(t)]
    for i in range(len(b) - 1):
        a, e = b[i], b[i + 1]
        c = "#f3d5b5" if seg[a] in force_segs else "#f0f0f0"
        ax.axvspan(t[a], t[min(e, len(t) - 1)], color=c, alpha=.55, lw=0, zorder=0)
    for k in sw:
        ax.axvline(t[k], color="k", lw=.35, alpha=.35, zorder=1)


OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "logs_analysis")
PDF, PNG = (os.path.join(OUT, s) for s in ("figures_pdf", "figures_png"))
for d in (PDF, PNG):
    os.makedirs(d, exist_ok=True)

paper_style()
df = pd.read_csv(CSV)
t = df["time_s"].to_numpy(float)
seg = df["seg_name"].to_numpy()
q = df[f"{JOINT}_pos"].to_numpy(float)
tau = df[f"{JOINT}_tau"].to_numpy(float)

peak_idx = int(np.argmax(np.abs(tau)))
peak_tau = float(tau[peak_idx])
peak_t = float(t[peak_idx])
peak_seg = str(seg[peak_idx])

fig, axs = plt.subplots(2, 1, figsize=(W1, 3.6), sharex=True)

ax = axs[0]
segbands(ax, t, seg, FORCE_SEGMENTS)
ax.plot(t, q, color="#003049", lw=1.0, label="thumb_yaw (measured)")
ax.axhline(THUMB_YAW_LIMIT_HI_RAD, color="#a4161a", ls="--", lw=.9,
           label=f"mechanical limit ({THUMB_YAW_LIMIT_HI_RAD:g} rad)")
ax.set_ylabel("q [rad]")
ax.legend(fontsize=6, ncol=1, loc="lower right", framealpha=0.9)

ax = axs[1]
segbands(ax, t, seg, FORCE_SEGMENTS)
ax.plot(t, tau, color="#003049", lw=1.0, label="thumb_yaw (measured)")
ax.annotate(f"peak = {peak_tau:.2f} Nm\n({peak_seg}, t={peak_t:.1f}s)",
            xy=(peak_t, peak_tau), xytext=(-70, -14), textcoords="offset points",
            fontsize=6, ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color="k", lw=.6))
ax.set_ylabel(r"$\tau$ [Nm]")
ax.set_xlabel("Time [s]")

fig.suptitle("Simulation: right-hand thumb_yaw position vs. torque (run34)", fontsize=8.5)
fig.savefig(os.path.join(PNG, "fig_run34_thumb_yaw.png"), dpi=200)
fig.suptitle("")
fig.savefig(os.path.join(PDF, "fig_run34_thumb_yaw.pdf"))
plt.close(fig)

print(f"Wrote fig_run34_thumb_yaw (pdf+png) to {OUT}")
print(f"peak |tau| = {peak_tau:.6f} Nm at t={peak_t:.3f}s, segment={peak_seg}, "
      f"q at that instant = {q[peak_idx]:.6f} rad")
