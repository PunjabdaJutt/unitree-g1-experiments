#!/usr/bin/env python3
"""
plot_hand_single_run_2panel.py
===============================

Simplified paper-figure replacement: ONE run (main_185109), TWO panels
(position, measured contact force), the five healthy right-hand joints
(thumb_pitch, index, middle, ring, pinky -- thumb_yaw excluded, faulty
sensor/actuator, see REAL_LOGS_ARCHITECTURE.md), sized for single-column
width (IEEE W1 = 3.5in). No tau_d panel -- position and force only.

main_185109 chosen because it's already the segment-timing reference used
elsewhere in this repo's figures and the only grasp_ph.py run with full
tau_d logging (irrelevant here since tau_d is dropped, but keeps this
figure consistent with the others that DO use it).

Same conventions as analyze_real_logs.py / plot_hand_no_thumb_yaw.py:
paper_style(), segbands(), titled PNG + untitled PDF.

Usage: python Real_logs/plot_hand_single_run_2panel.py [out_dir]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_CSV = "hand_state_20260918_185109.csv"

# thumb_yaw intentionally excluded (faulty hardware).
JOINTS = ["R_thumb_proximal_pitch_joint", "R_index_proximal_joint", "R_middle_proximal_joint",
          "R_ring_proximal_joint", "R_pinky_proximal_joint"]
SHORT = [j.replace("R_", "").replace("_proximal", "").replace("_joint", "") for j in JOINTS]
COLORS = ["#a4161a", "#2a9d8f", "#e07a5f", "#8d99ae", "#6a4c93"]
FORCE_SEG = {"grasp_hold"}
W1 = 3.50  # IEEE single-column width [in]


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


def segbands(ax, t, seg, highlight):
    sw = np.where(seg[1:] != seg[:-1])[0] + 1
    b = [0] + list(sw) + [len(t)]
    for i in range(len(b) - 1):
        a, e = b[i], b[i + 1]
        c = "#f3d5b5" if seg[a] in highlight else "#f0f0f0"
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

fig, axs = plt.subplots(2, 1, figsize=(W1, 3.9), sharex=True)

ax = axs[0]
segbands(ax, t, seg, FORCE_SEG)
for j, s, c in zip(JOINTS, SHORT, COLORS):
    ax.plot(t, df[f"{j}_pos_rad"], color=c, lw=.9, label=s)
ax.set_ylabel("q [rad]")
ax.legend(fontsize=6, ncol=3, loc="upper left", framealpha=0.9, handlelength=1.4, columnspacing=0.8)

ax = axs[1]
segbands(ax, t, seg, FORCE_SEG)
for j, s, c in zip(JOINTS, SHORT, COLORS):
    ax.plot(t, df[f"{j}_force_g"], color=c, lw=.9, label=s)
ax.set_ylabel("Force [g]")
ax.set_xlabel("Time [s]")
ax.legend(fontsize=6, ncol=3, loc="upper left", framealpha=0.9, handlelength=1.4, columnspacing=0.8)

fig.suptitle("Right-hand state through the full grasp (main_185109)", fontsize=8.5)
fig.savefig(os.path.join(PNG, "fig13_handB_single_run_2panel.png"), dpi=200)
fig.suptitle("")
fig.savefig(os.path.join(PDF, "fig13_handB_single_run_2panel.pdf"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Companion figure: thumb_yaw's force reading on its own axis. Excluded from
# fig13 above (and from fig04b/fig06-08 elsewhere) because it's known-faulty
# hardware -- this figure exists specifically to SHOW that fault, not hide it.
# Reference lines are g1_hw_ph_finger_control.py's own thresholds: the
# modbus_force_limit_g hard Modbus stop (600g default) and FORCE_FAULT_ABS_G
# sensor-fault gate (1000g) -- see REAL_LOGS_ARCHITECTURE.md.
# ---------------------------------------------------------------------------
THUMB_YAW = "R_thumb_proximal_yaw_joint"
MODBUS_FORCE_LIMIT_G = 600.0
FORCE_FAULT_ABS_G = 1000.0

fig, ax = plt.subplots(figsize=(W1, 2.6))
segbands(ax, t, seg, FORCE_SEG)
fy = df[f"{THUMB_YAW}_force_g"].to_numpy(float)
ax.plot(t, fy, color="#003049", lw=.8, label="thumb_yaw (measured)")
ax.axhline(MODBUS_FORCE_LIMIT_G, color="#e07a5f", ls="--", lw=.9,
           label=f"modbus_force_limit_g ({MODBUS_FORCE_LIMIT_G:g} g)")
ax.axhline(-MODBUS_FORCE_LIMIT_G, color="#e07a5f", ls="--", lw=.9)
ax.axhline(FORCE_FAULT_ABS_G, color="#a4161a", ls=":", lw=.9,
           label=f"FORCE_FAULT_ABS_G ({FORCE_FAULT_ABS_G:g} g)")
ax.axhline(-FORCE_FAULT_ABS_G, color="#a4161a", ls=":", lw=.9)
peak = float(np.max(np.abs(fy)))
ax.text(.02, .04, f"peak |force| = {peak:.0f} g", transform=ax.transAxes,
        fontsize=6.5, ha="left", va="bottom", style="italic")
ax.set_ylabel("Force [g]")
ax.set_xlabel("Time [s]")
ax.legend(fontsize=6, ncol=1, loc="upper right", framealpha=0.9)
fig.suptitle("thumb_yaw force reading -- known-faulty sensor/actuator (main_185109)", fontsize=8)
fig.savefig(os.path.join(PNG, "fig14_thumb_yaw_force_fault.png"), dpi=200)
fig.suptitle("")
fig.savefig(os.path.join(PDF, "fig14_thumb_yaw_force_fault.pdf"))
plt.close(fig)

print("Wrote fig13_handB_single_run_2panel (pdf+png) to", OUT)
print(f"Wrote fig14_thumb_yaw_force_fault (pdf+png), peak |force| = {peak:.0f} g, to", OUT)
