"""G1 hardware logs: analysis figures (right arm bench test + grasp run).

Folder layout this script expects (everything relative to where this file lives):

    G1_hardware_analysis/
        make_figs.py          <- this script
        data/                 <- the CSV logs from the robot
        figures/              <- created here: PNG + PDF of every figure

Run from PowerShell, inside the G1_hardware_analysis folder:

    python make_figs.py
    python make_figs.py --data "D:\other\logs" --out "D:\other\figures"
    python make_figs.py --grasp-run 20260918_185109

By default the newest file of each kind in data/ is used. Output is also a small
table, results_summary.csv, with every number the figures quote.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no screen needed; figures go straight to files
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="Make the G1 hardware-log figures.")
ap.add_argument("--data", default=os.path.join(HERE, "data"), help="folder with the CSV logs (default: ./data)")
ap.add_argument("--out", default=os.path.join(HERE, "figures"), help="folder for the figures (default: ./figures)")
ap.add_argument("--bench-ph", default=None, help="timestamp of the PH bench run, e.g. 20260918_183908")
ap.add_argument("--bench-pd", default=None, help="timestamp of the position-PD bench run, e.g. 20260918_184048")
ap.add_argument("--grasp-run", default=None, help="timestamp of the grasp run, e.g. 20260918_185109")
args = ap.parse_args()
DATA, OUT = os.path.abspath(args.data), os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)


def find_log(prefix, stamp):
    """Return data/<prefix><stamp>.csv, or the newest data/<prefix>*.csv if no stamp is given."""
    if stamp:
        path = os.path.join(DATA, f"{prefix}{stamp}.csv")
        if not os.path.isfile(path):
            sys.exit(f"[ERROR] not found: {path}")
        return path
    hits = sorted(glob.glob(os.path.join(DATA, f"{prefix}*.csv")))
    if not hits:
        sys.exit(f"[ERROR] no {prefix}*.csv in {DATA}. Put the robot logs in that folder or pass --data.")
    return hits[-1]


def clock(path):
    """'arm_ph_ph_20260918_183908.csv' -> '18 Sep 2026, 18:39'."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\d{2}", os.path.basename(path))
    if not m:
        return os.path.basename(path)
    y, mo, d, hh, mm = m.groups()
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"{int(d)} {months[int(mo) - 1]} {y}, {hh}:{mm}"


SUMMARY = []  # rows for results_summary.csv


def note(figure, quantity, value, unit=""):
    SUMMARY.append(dict(figure=figure, quantity=quantity, value=round(float(value), 4), unit=unit))


# ---- style -----------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, INK2, MUTED, GRID = "#8f8e89", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BAND = "#eeede8"
PH_C, PD_C = BLUE, GRAY

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def suptitle(fig, title, sub, pad_in=0.12):
    """Title + subtitle stacked ABOVE the axes area, spaced in inches so they never collide."""
    fig.tight_layout()
    h = fig.get_figheight()
    n = sub.count("\n") + 1
    y_sub = 1 + pad_in / h
    y_tit = 1 + (pad_in + n * 0.19 + 0.08) / h
    fig.text(0.0, y_sub, sub, fontsize=9.5, color=INK2, ha="left", va="bottom", linespacing=1.35)
    fig.text(0.0, y_tit, title, fontsize=13, fontweight="bold", color=INK, ha="left", va="bottom")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("  wrote", os.path.join(OUT, name) + ".png / .pdf")


def shade(ax, d, seg, label=None, color=BAND):
    s = d[d.segment == seg]
    if len(s):
        ax.axvspan(s.t_s.min(), s.t_s.max(), color=color, lw=0, zorder=0)
        if label:
            ax.text((s.t_s.min() + s.t_s.max()) / 2, 1.0, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=INK2)


# ---- data ------------------------------------------------------------------
F_PH = find_log("arm_ph_ph_", args.bench_ph)
F_PD = find_log("arm_ph_position_", args.bench_pd)
F_ARM = find_log("arm_positions_", args.grasp_run)
F_HAND = find_log("hand_state_", args.grasp_run or re.search(r"(\d{8}_\d{6})", F_ARM).group(1))
print("Data folder :", DATA)
for lab, f in (("bench PH", F_PH), ("bench PD", F_PD), ("grasp arm", F_ARM), ("grasp hand", F_HAND)):
    print(f"  {lab:10s}: {os.path.basename(f)}")
print("Figures to  :", OUT)

ph = pd.read_csv(F_PH)
pd_ = pd.read_csv(F_PD)
ph = ph[ph.segment != "done"]; pd_ = pd_[pd_.segment != "done"]
arm = pd.read_csv(F_ARM)
hand = pd.read_csv(F_HAND)

JOINTS = [("right_shoulder_pitch", "Shoulder pitch", 15.0),
          ("right_elbow", "Elbow", 10.0),
          ("right_wrist_pitch", "Wrist pitch", 3.0)]
SAFETY_KP, PD_KP = 3.0, 60.0


def steady(d):
    h = d[d.segment == "hold_torque"]
    return h[h.t_s > h.t_s.min() + 1.0]  # after the switch-over transient


# ============================================================================
# Fig 1: the whole bench test, both runs on top of each other
# ============================================================================
fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)
seg_labels = {"hold_in": "ramp in", "move_to_pose": "move to pose", "hold_torque": "hold (law differs)",
              "return": "return", "release": "ramp out"}
for ax, (j, name, _) in zip(axes, JOINTS):
    for seg, lab in seg_labels.items():
        s = ph[ph.segment == seg]
        if seg == "hold_torque":
            ax.axvspan(s.t_s.min(), s.t_s.max(), color=BAND, lw=0, zorder=0)
        if ax is axes[0]:
            ax.text((s.t_s.min() + s.t_s.max()) / 2, 1.02, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=INK2)
        ax.axvline(s.t_s.min(), color=GRID, lw=0.8, zorder=0)
    ax.plot(ph.t_s, ph[j + "_qdes_rad"], color=INK, lw=1.0, ls=(0, (4, 3)), label="commanded target")
    ax.plot(pd_.t_s, pd_[j], color=PD_C, lw=2, label="position-PD baseline (kp 60)")
    ax.plot(ph.t_s, ph[j], color=PH_C, lw=2, label="PH torque law (in hold)")
    ax.set_ylabel(f"{name} [rad]")
    # annotate steady offsets in hold
    sp, spd = steady(ph), steady(pd_)
    e_ph, e_pd = sp[j + "_qerr_rad"].mean(), spd[j + "_qerr_rad"].mean()
    note("fig1", f"{name}: steady hold error, PH", abs(e_ph), "rad")
    note("fig1", f"{name}: steady hold error, PD", abs(e_pd), "rad")
    ytxt = {"right_shoulder_pitch": 0.80, "right_elbow": None, "right_wrist_pitch": 0.80}[j]
    if ytxt is None:  # elbow: write between the two held lines
        yd = (sp[j].mean() + spd[j].mean()) / 2
        ax.text(8.5, yd, f"PH off by {abs(e_ph):.3f} rad\nPD off by {abs(e_pd):.3f} rad",
                fontsize=8, color=INK, ha="center", va="center")
    else:
        ax.text(8.5, ytxt, f"PH off by {abs(e_ph):.3f} rad\nPD off by {abs(e_pd):.3f} rad",
                transform=ax.get_xaxis_transform(), fontsize=8, color=INK, ha="center", va="center")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
axes[-1].set_xlabel("time [s]")
suptitle(fig, "Arm bench test: neither law reaches the target pose", pad_in=0.45, sub=
         f"Right arm, same choreography run twice (PH {clock(F_PH)}; PD {clock(F_PD)}). Only the grey hold band uses a different law.")
save(fig, "fig1_arm_bench_overview")

# ============================================================================
# Fig 2: zoom on the hold, every logged sample, plus how much the joint moves
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6), gridspec_kw=dict(height_ratios=[1.4, 1]))
for col, (j, name, _) in enumerate(JOINTS[:2]):
    ax, axv = axes[0, col], axes[1, col]
    for d, c, lab in ((pd_, PD_C, "PD baseline"), (ph, PH_C, "PH law")):
        w = d[(d.t_s > 4.8) & (d.t_s < 11.2)]
        ax.plot(w.t_s, w[j], color=c, lw=1.4, marker="o", ms=3.2, mec="white", mew=0.5, label=lab)
        step = np.abs(np.diff(w[j].values))
        axv.semilogy(w.t_s.values[1:], np.maximum(step, 1e-6), color=c, lw=1.4, marker="o", ms=3.2,
                     mec="white", mew=0.5)
    for a in (ax, axv):
        shade(a, ph, "hold_torque")
    ax.plot(w.t_s, w[j + "_qdes_rad"], color=INK, lw=1.0, ls=(0, (4, 3)), label="target")
    ax.set_title(name)
    ax.set_ylabel("angle [rad]")
    axv.set_ylabel("change per sample [rad]")
    axv.set_xlabel("time [s]")
    axv.axhline(1e-4, color=MUTED, lw=0.8, ls=":")
    axv.set_ylim(5e-7, 0.2)
    h0 = ph[ph.segment == "hold_torque"].t_s.min()
    axv.annotate("joint stops moving\n~0.6 s into the hold", xy=(h0 + 0.7, 2e-5), xytext=(h0 + 1.6, 3e-3),
                 fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
axes[0, 0].legend(loc="upper right")
axes[1, 1].text(4.75, 1.3e-4, "0.1 mrad", fontsize=7.5, color=MUTED, ha="left", va="bottom")
suptitle(fig, "The 5 s hold is a 0.6 s transient, then a frozen equilibrium",
         "Every logged sample shown (10 Hz). After the switch, the joints stop within about six samples in both runs:\n"
         "what the hold reports is a static offset held by friction, not tracking.")
save(fig, "fig2_hold_zoom")

# ============================================================================
# Fig 3: torque needed to hold the pose versus what the model predicts
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(9.0, 4.4))
ratios = []
for ax, (j, name, kp) in zip(axes, JOINTS):
    sp, spd = steady(ph), steady(pd_)
    g = abs(sp[j + "_g_Nm"].mean())
    tau = sp[j + "_tau_cmd_Nm"].mean()
    e = sp[j + "_qerr_rad"].mean()
    fb = abs(tau) - g                      # impedance feedback part of tau
    floor = abs(SAFETY_KP * e)             # firmware PD floor, same direction
    pd_hold = abs(PD_KP * spd[j + "_qerr_rad"].mean())
    x = [0, 1]
    ax.bar(0, pd_hold, width=0.6, color=PD_C, edgecolor="white", lw=1.5)
    ax.bar(1, g, width=0.6, color=PH_C, edgecolor="white", lw=1.5)
    ax.bar(1, fb, bottom=g, width=0.6, color=ORANGE, edgecolor="white", lw=1.5)
    ax.bar(1, floor, bottom=g + fb, width=0.6, color=AQUA, edgecolor="white", lw=1.5)
    ax.axhline(g, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax.text(1.5, g, f"model g(q)\n{g:.2f} Nm", fontsize=8, color=INK, va="center", ha="left")
    ax.text(0, pd_hold, f"{pd_hold:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ph_tot = g + fb + floor
    ratios += [ph_tot / g, pd_hold / g]
    note("fig3", f"{name}: model gravity torque g(q)", g, "Nm")
    note("fig3", f"{name}: holding torque, PH run (tau + floor)", ph_tot, "Nm")
    note("fig3", f"{name}: holding torque, PD run (60 x error)", pd_hold, "Nm")
    ax.text(1, ph_tot, f"{ph_tot:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, ["PD\nbaseline", "PH\nlaw"])
    ax.set_xlim(-0.5, 2.3)
    ax.set_title(name)
    ax.grid(axis="x", visible=False)
    ax.text(0.5, -0.24, f"needs {ph_tot / g:.1f}× to {pd_hold / g:.1f}× the model",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=INK2)
axes[0].set_ylabel("torque holding the pose [Nm]  (magnitude)")
handles = [Patch(color=PD_C, label="PD baseline: 60 · error"),
           Patch(color=PH_C, label="PH: gravity feed-forward g(q)"),
           Patch(color=ORANGE, label="PH: impedance feedback Kp · error"),
           Patch(color=AQUA, label="PH: firmware safety floor 3 · error")]
fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.13))
suptitle(fig, "The gravity model under-predicts the torque the arm needs",
         f"Torque that actually held the arm still during the hold, from each run. Both runs, measured independently, need {min(ratios):.1f} to {max(ratios):.1f} times\n"
         f"Pinocchio's g(q). The larger PH error is what a lower stiffness ({JOINTS[0][2] + SAFETY_KP:.0f} vs {PD_KP:.0f} Nm/rad on shoulder pitch) does with the same missing torque.")
save(fig, "fig3_holding_torque_vs_model")

# ============================================================================
# Fig 4: proof that the firmware applies tau
# ============================================================================
fig, ax = plt.subplots(figsize=(6.4, 3.9))
names, obs, ifign = [], [], []
for j, name, kp in JOINTS:
    sp = steady(ph)
    e = abs(sp[j + "_qerr_rad"].mean())
    need = abs(sp[j + "_tau_cmd_Nm"].mean()) + SAFETY_KP * e
    names.append(name); obs.append(e); ifign.append(need / SAFETY_KP)
    note("fig4", f"{name}: PH error measured", e, "rad")
    note("fig4", f"{name}: PH error if tau were ignored", need / SAFETY_KP, "rad")
y = np.arange(len(names))
ax.barh(y + 0.19, ifign, height=0.36, color=GRAY, edgecolor="white", lw=1.5,
        label="error if the firmware ignored tau (only the kp 3 floor left)")
ax.barh(y - 0.19, obs, height=0.36, color=PH_C, edgecolor="white", lw=1.5, label="error actually measured in PH mode")
for yi, a, b in zip(y, obs, ifign):
    ax.text(a + 0.02, yi - 0.19, f"{a:.2f} rad", va="center", fontsize=8.5, color=INK)
    ax.text(b + 0.02, yi + 0.19, f"{b:.2f} rad", va="center", fontsize=8.5, color=INK)
ax.axvline(0.6, color=ORANGE, lw=1.4)
ax.set_ylim(2.55, -0.85)
ax.text(0.62, -0.6, "watchdog abort 0.6 rad", color=INK2, fontsize=8, va="center")
ax.set_yticks(y, names)
ax.set_xlabel("steady hold error [rad]")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(max(ifign), 0.6) * 1.18)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=1)
suptitle(fig, "The arm firmware does apply the torque command",
         "Without tau the weak floor would let shoulder and elbow sag past the watchdog.\n"
         f"The measured error is {min(np.array(ifign) / np.array(obs)):.0f} to {max(np.array(ifign) / np.array(obs)):.0f} times smaller, so tau reaches the motors.")
save(fig, "fig4_tau_is_applied")

# ============================================================================
# Fig 5: the waist moves although it is commanded to hold
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
for ax, (d, title) in zip(axes, ((ph, "Bench test, PH run"), (arm, "Grasp run (IK + position-PD)"))):
    for j, c, lab in (("waist_pitch", ORANGE, "waist pitch"), ("waist_yaw", BLUE, "waist yaw"),
                      ("waist_roll", AQUA, "waist roll")):
        ax.plot(d.t_s, d[j] - d[j].iloc[0], color=c, lw=2)
        ax.plot(d.t_s, d[j + "_qdes_rad"] - d[j].iloc[0], color=c, lw=0.9, ls=(0, (4, 3)))
    ax.set_title(title); ax.set_xlabel("time [s]")
    ax.set_xlim(0, d.t_s.iloc[-1])
axes[0].set_ylabel("change from start [rad]")
fig.legend(handles=[Line2D([], [], color=ORANGE, lw=2, label="waist pitch"),
                    Line2D([], [], color=AQUA, lw=2, label="waist roll"),
                    Line2D([], [], color=BLUE, lw=2, label="waist yaw"),
                    Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 3)), label="commanded (flat)")],
           loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
lean = max((ph.waist_pitch - ph.waist_pitch.iloc[0]).max(), (arm.waist_pitch - arm.waist_pitch.iloc[0]).max())
note("fig5", "largest forward lean of waist pitch", lean, "rad")
suptitle(fig, "The waist is not static",
         f"Waist pitch leans up to {lean:.2f} rad forward while the arm reaches out, although its command never changes.\n"
         "Proposition 1 assumes these joints are fixed, and g(q) depends on them.")
save(fig, "fig5_waist_moves")

# ============================================================================
# Fig 6: grasp run arm tracking, and why side_down must be excluded
# ============================================================================
segs = arm.segment.drop_duplicates().tolist()
fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True)
for ax, (j, name, _) in zip(axes, JOINTS[:2]):
    for i, s in enumerate(segs):
        ss = arm[arm.segment == s]
        if s == "side_down":
            ax.axvspan(ss.t_s.min(), ss.t_s.max(), color="#fbe3d8", lw=0, zorder=0)
        elif s in ("grasp_hold", "lift_hold"):
            ax.axvspan(ss.t_s.min(), ss.t_s.max(), color=BAND, lw=0, zorder=0)
        ax.axvline(ss.t_s.min(), color=GRID, lw=0.8, zorder=0)
        if ax is axes[0]:
            ax.text((ss.t_s.min() + ss.t_s.max()) / 2, 1.02, s.replace("_", " "), rotation=40,
                    transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=7.5, color=INK2)
    ax.plot(arm.t_s, arm[j + "_qdes_rad"], color=INK, lw=1.0, ls=(0, (4, 3)), label="commanded (IK)")
    ax.plot(arm.t_s, arm[j], color=PD_C, lw=2, label="measured")
    ax.set_ylabel(f"{name} [rad]")
axes[0].legend(loc="lower left")
sd = arm[arm.segment == "side_down"]
axes[1].text(33.3, 0.97, "Pink band (side down): the arm is handed back to Unitree's controller.\n"
             "It returns home while the logged target keeps going,\nso the error there is not a tracking error.",
             fontsize=8, color=INK2, va="top", ha="right")
axes[-1].set_xlabel("time [s]")
valid = arm[arm.segment != "side_down"]
e_sp = valid["right_shoulder_pitch_qerr_rad"].abs()
e_gh = arm.loc[arm.segment == "grasp_hold", "right_shoulder_pitch_qerr_rad"].abs().median()
note("fig6", "shoulder pitch error in grasp hold (median)", e_gh, "rad")
note("fig6", "largest shoulder pitch error, side_down excluded", e_sp.max(), "rad")
note("fig6", "largest shoulder pitch error inside side_down", arm["right_shoulder_pitch_qerr_rad"].abs().max(), "rad")
suptitle(fig, f"Grasp run: a steady {e_gh:.2f}–{e_sp.max():.2f} rad shoulder offset, and a false alarm at the end", pad_in=0.3, sub=
         f"Same sag as in the bench test. Excluding side down, the largest shoulder-pitch error is {e_sp.max():.2f} rad "
         f"(it reaches {arm['right_shoulder_pitch_qerr_rad'].abs().max():.2f} rad inside side down).")
save(fig, "fig6_grasp_arm_tracking")

# ============================================================================
# Fig 7: hand forces through the grasp
# ============================================================================
G2N = 9.80665e-3
FING = [("R_thumb_proximal_pitch_joint", "thumb pitch", BLUE), ("R_index_proximal_joint", "index", ORANGE),
        ("R_middle_proximal_joint", "middle", AQUA), ("R_ring_proximal_joint", "ring", YELLOW),
        ("R_pinky_proximal_joint", "pinky", MAGENTA)]
fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.2, 0.9]))
for ax in axes:
    for s in ("grasp_hold", "lift_hold"):
        shade(ax, hand, s, label=s.replace("_", " ") if ax is axes[0] else None)
    for s in ("grasp", "lift", "lower", "release"):
        ss = hand[hand.segment == s]
        ax.axvline(ss.t_s.min(), color=GRID, lw=0.8, zorder=0)
xl = hand.t_s.iloc[-1]
for j, lab, c in FING:
    axes[0].plot(hand.t_s, hand[j + "_pos_rad"], color=c, lw=1.8)
    axes[1].plot(hand.t_s, hand[j + "_force_g"] * G2N, color=c, lw=1.8)
axes[0].legend(handles=[Line2D([], [], color=c, lw=2, label=lab) for _, lab, c in FING],
               loc="upper left", ncol=5, bbox_to_anchor=(0.0, 1.0))
axes[0].set_ylim(-0.2, 2.1)
axes[1].annotate("ring reads about −0.7 N with no contact\n(sensor offset)", xy=(8, -0.64), xytext=(1.0, 1.7),
                 fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
axes[0].set_ylabel("finger angle [rad]")
axes[1].set_ylabel("contact force [N]")
tyaw = hand["R_thumb_proximal_yaw_joint_force_g"] * G2N
axes[2].plot(hand.t_s, tyaw, color=GRAY, lw=1.4)
for lim, ls in ((600 * G2N, (0, (4, 3))), (1000 * G2N, ":")):
    for sg in (1, -1):
        axes[2].axhline(sg * lim, color=ORANGE, lw=1.0, ls=ls)
axes[2].text(0.3, -42, "dashed: ±5.9 N Modbus force limit    dotted: ±9.8 N fault threshold",
             fontsize=7.5, color=INK2, va="bottom")
axes[2].set_ylabel("thumb yaw force [N]")
axes[2].set_xlabel("time [s]")
axes[2].set_title("Thumb yaw (faulty sensor, excluded from the grip analysis)", fontsize=9.5)
for a in axes: a.set_xlim(0, xl)
tau_d = hand.loc[hand.segment == "grasp_hold", "R_index_proximal_joint_tau_des_Nm"].max()
held = hand[hand.segment.isin(["grasp_hold", "lift_hold"])]
f_means = [held[j + "_force_g"].mean() * G2N for j, lab, c in FING if lab != "thumb pitch"]
for j, lab, c in FING:
    note("fig7", f"{lab}: mean force in grasp hold", hand.loc[hand.segment == "grasp_hold", j + "_force_g"].mean() * G2N, "N")
    note("fig7", f"{lab}: mean force in lift hold", hand.loc[hand.segment == "lift_hold", j + "_force_g"].mean() * G2N, "N")
note("fig7", "logged force-mode target tau_d", tau_d, "Nm")
note("fig7", "thumb yaw peak |force|", tyaw.abs().max(), "N")
suptitle(fig, "Hand: the grip is set in grasp hold and survives the lift", pad_in=0.3, sub=
         f"Right hand, grasp run of {clock(F_HAND)}. The four fingers hold {min(f_means):.1f}–{max(f_means):.1f} N on average from grasp hold through lift hold.\n"
         f"Logged force-mode target τd = {tau_d:.2f} Nm on every right-hand joint; the thesis text says 2 Nm (1 Nm thumb yaw).")
save(fig, "fig7_hand_grip")

pd.DataFrame(SUMMARY).to_csv(os.path.join(OUT, "results_summary.csv"), index=False)
print("  wrote", os.path.join(OUT, "results_summary.csv"))
print("Done: 7 figures.")
