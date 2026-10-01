#!/usr/bin/env python3
"""
Category B right-hand figures with thumb_yaw REMOVED ENTIRELY (known-faulty sensor/actuator,
see analyze_real_logs.py) so the remaining five right-hand joints -- thumb_pitch, index,
middle, ring, pinky -- can be read at their own scale (thumb_yaw's +/-5000 g spikes otherwise
flatten every other force trace).

Same conventions as analyze_real_logs.py / scripts/g1_control/analysis.py: paper_style(),
segbands(), save_fig() (titled PNG + untitled PDF), save_table() (CSV). thumb_yaw is dropped
at load time, so it never reaches a plot or a table.

Outputs (into real_logs_analysis/, alongside the existing figures):
  fig04b_handB_timeline_no_thumb_yaw        main_185109, q / force / tau_d   (same as fig04)
  fig06_handB_<run>_timeline_no_thumb_yaw   one per Successfull_grasps run, q / force
  fig07_handB_runs_overlay_no_thumb_yaw     5 joints x 5 runs, q and force overlaid
  fig08_handB_settled_no_thumb_yaw          settled grasp_hold mean +/- std across runs
  tables/handB_no_thumb_yaw_settled.csv

Usage:  python Real_logs/plot_hand_no_thumb_yaw.py [out_dir] [a4|ieee]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# thumb_yaw is intentionally NOT in this list.
JOINTS = ["R_thumb_proximal_pitch_joint", "R_index_proximal_joint", "R_middle_proximal_joint",
          "R_ring_proximal_joint", "R_pinky_proximal_joint"]
SHORT = [j.replace("R_", "").replace("_proximal", "").replace("_joint", "") for j in JOINTS]
C_BLUE, C_RED, C_TEAL, C_ORANGE, C_GRAY = "#003049", "#a4161a", "#2a9d8f", "#e07a5f", "#8d99ae"
# same colours the existing fig04 gives these joints (thumb_yaw's C_BLUE slot is simply unused)
COLORS = [C_RED, C_TEAL, C_ORANGE, C_GRAY, "#6a4c93"]
NOTE = "thumb_yaw excluded (faulty hardware)"

RUNS = [
    ("main_185109", "hand_state_20260918_185109.csv"),
    ("sg_173308", "Successfull_grasps/hand_state_20260918_173308.csv"),
    ("sg_173640", "Successfull_grasps/hand_state_20260918_173640.csv"),
    ("sg_173920", "Successfull_grasps/hand_state_20260918_173920.csv"),
    ("sg_174410", "Successfull_grasps/hand_state_20260918_174410.csv"),
]
RUN_COLORS = {"main_185109": C_BLUE, "sg_173308": C_RED, "sg_173640": C_TEAL,
              "sg_173920": C_ORANGE, "sg_174410": "#6a4c93"}
FORCE_SEG = {"grasp_hold"}


def paper_style():
    serif = [f for f in ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
             if f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}]
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": serif or ["DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.4, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "real_logs_analysis")
_PAGE = (sys.argv[2].lower() if len(sys.argv) > 2 else "a4")
if _PAGE not in ("a4", "ieee"):
    sys.exit(f"ERROR: unknown page preset '{_PAGE}'. Use 'a4' or 'ieee'.")
W1, W2 = (3.05, 6.30) if _PAGE == "a4" else (3.50, 7.16)
PDF, PNG, TAB = (os.path.join(OUT, s) for s in ("figures_pdf", "figures_png", "tables"))
for d in (PDF, PNG, TAB):
    os.makedirs(d, exist_ok=True)
CAPTIONS = {}


def save_fig(fig, stem, title, caption):
    fig.suptitle(title, fontsize=9)
    fig.savefig(os.path.join(PNG, stem + ".png"), dpi=200)
    fig.suptitle("")
    fig.savefig(os.path.join(PDF, stem + ".pdf"))
    plt.close(fig)
    CAPTIONS[stem] = caption


def segbands(ax, t, seg, highlight):
    sw = np.where(seg[1:] != seg[:-1])[0] + 1
    b = [0] + list(sw) + [len(t)]
    for i in range(len(b) - 1):
        a, e = b[i], b[i + 1]
        c = "#f3d5b5" if seg[a] in highlight else "#f0f0f0"
        ax.axvspan(t[a], t[min(e, len(t) - 1)], color=c, alpha=.55, lw=0, zorder=0)
    for k in sw:
        ax.axvline(t[k], color="k", lw=.35, alpha=.35, zorder=1)


def load(fname):
    df = pd.read_csv(os.path.join(HERE, fname))
    keep = ["t_s", "segment"]
    for j in JOINTS:
        keep += [c for c in (f"{j}_pos_rad", f"{j}_force_g", f"{j}_tau_des_Nm") if c in df.columns]
    return df[keep]  # every thumb_yaw column (and the whole left hand) is dropped here


paper_style()
DATA = {name: load(f) for name, f in RUNS}


def panel(ax, df, suffix, ylabel, legend_loc="upper right"):
    t, seg = df["t_s"].to_numpy(float), df["segment"].to_numpy()
    segbands(ax, t, seg, FORCE_SEG)
    for j, s, c in zip(JOINTS, SHORT, COLORS):
        ax.plot(t, df[f"{j}_{suffix}"], color=c, lw=.8, label=s)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=5.5, ncol=2, loc=legend_loc)


# ---- fig04b: main run, same three panels as fig04, no thumb_yaw
df = DATA["main_185109"]
fig, axs = plt.subplots(3, 1, figsize=(W2, 5.2), sharex=True)
panel(axs[0], df, "pos_rad", "q [rad]")
panel(axs[1], df, "force_g", "Force [g]")
panel(axs[2], df, "tau_des_Nm", "$\\tau_d$ [Nm]", legend_loc="center right")
axs[2].set_xlabel("Time [s]")
save_fig(fig, "fig04b_handB_timeline_no_thumb_yaw",
         f"Category B: right-hand state through the full grasp (main_185109) -- {NOTE}",
         "Same data as fig04 with thumb_yaw removed, so the force axis is scaled to the five "
         "healthy joints (thumb_pitch, index, middle, ring, pinky). Shaded band is grasp_hold, "
         "the only FORCE-mode segment. thumb_yaw is excluded because its sensor/actuator is "
         "faulty hardware on this robot.")

# ---- fig06: one per Successfull_grasps run (these logs have no tau_des column: q + force)
for name, _ in RUNS[1:]:
    df = DATA[name]
    fig, axs = plt.subplots(2, 1, figsize=(W2, 3.9), sharex=True)
    panel(axs[0], df, "pos_rad", "q [rad]")
    panel(axs[1], df, "force_g", "Force [g]")
    axs[1].set_xlabel("Time [s]")
    save_fig(fig, f"fig06_handB_{name}_timeline_no_thumb_yaw",
             f"Category B: right-hand state through the full grasp ({name}) -- {NOTE}",
             f"Right-hand joint position and measured contact force for the successful grasp "
             f"{name}. Shaded band is grasp_hold (FORCE mode). {NOTE}. This older log schema "
             f"has no tau_des column, so no force-target panel is shown.")

# ---- fig07: all runs overlaid, one column per joint
fig, axs = plt.subplots(2, len(JOINTS), figsize=(W2 * 1.9, 4.2), sharex=True)
ref = DATA["main_185109"]
for k, (j, s) in enumerate(zip(JOINTS, SHORT)):
    for row, (suffix, yl) in enumerate([("pos_rad", "q [rad]"), ("force_g", "Force [g]")]):
        ax = axs[row, k]
        segbands(ax, ref["t_s"].to_numpy(float), ref["segment"].to_numpy(), FORCE_SEG)
        for name, d in DATA.items():
            ax.plot(d["t_s"], d[f"{j}_{suffix}"], color=RUN_COLORS[name], lw=.7,
                    label=name if (row == 0 and k == 0) else None)
        if row == 0:
            ax.set_title(s, fontsize=7.5, pad=3)
        if k == 0:
            ax.set_ylabel(yl)
        if row == 1:
            ax.set_xlabel("Time [s]")
axs[0, 0].legend(fontsize=5, loc="upper left")
save_fig(fig, "fig07_handB_runs_overlay_no_thumb_yaw",
         f"Category B: right-hand joints across all 5 grasp runs -- {NOTE}",
         "Measured position (top) and contact force (bottom) of the five healthy right-hand "
         "joints, all five grasp_ph.py runs overlaid. Shaded bands use main_185109's segment "
         "timing; the other runs' segment boundaries land at similar but not identical times. "
         f"{NOTE}.")

# ---- fig08 + table: settled grasp_hold state, mean +/- std across runs
rows = []
for name, d in DATA.items():
    g = d[d["segment"] == "grasp_hold"]
    n = len(g)
    g = g.iloc[n // 2:]  # second half of grasp_hold = settled
    for j, s in zip(JOINTS, SHORT):
        rows.append(dict(run=name, joint=s, pos_rad=g[f"{j}_pos_rad"].mean(),
                         force_g=g[f"{j}_force_g"].mean()))
settled = pd.DataFrame(rows)
settled.to_csv(os.path.join(TAB, "handB_no_thumb_yaw_settled.csv"), index=False)
mean = settled.groupby("joint")[["pos_rad", "force_g"]].mean().reindex(SHORT)
std = settled.groupby("joint")[["pos_rad", "force_g"]].std().reindex(SHORT)
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0))
x = np.arange(len(SHORT))
for ax, col, yl in zip(axs, ["pos_rad", "force_g"], ["q [rad]", "Force [g]"]):
    ax.bar(x, mean[col], yerr=std[col], color=COLORS, capsize=2, width=.6)
    for name in DATA:
        sub = settled[settled["run"] == name].set_index("joint").reindex(SHORT)
        ax.plot(x, sub[col], "o", ms=2.5, color="k", alpha=.55, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=25, ha="right", fontsize=6.5)
    ax.set_ylabel(yl)
save_fig(fig, "fig08_handB_settled_no_thumb_yaw",
         f"Category B: settled grasp_hold state, mean +/- std across 5 runs -- {NOTE}",
         "Mean of the second half of grasp_hold per run, then mean (bar) and standard "
         "deviation (whisker) across the five runs; black dots are the individual runs. "
         f"{NOTE}.")

print("\n".join(f"{k}: written" for k in CAPTIONS))
print(mean.round(3).to_string())
print(std.round(3).to_string())
