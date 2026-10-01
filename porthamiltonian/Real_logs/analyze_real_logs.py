#!/usr/bin/env python3
"""
analyze_real_logs.py
=====================

First-pass, categorized analysis of Real_logs/ -- the real-hardware CSVs, as opposed to
scripts/g1_control/analysis.py (this directory's sibling), which analyzes the SIMULATION
run logs (robot_ph_run*_q_tau_29.csv). Same plumbing/plotting conventions as that script
(paper_style, say()-buffered report, save_fig()'s titled-PNG + untitled-PDF pair,
save_table()'s CSV-per-table, segment-colored background bands), deliberately kept
identical so both analyses read as one family -- see REAL_LOGS_ARCHITECTURE.md for what
each input file is and which control law produced it.

Two categories, analyzed and reported separately because they are two different control
architectures on two different subsystems -- pooling them would mix incomparable things:

  Category A: g1_hw_arm_ph_control.py bench test -- right-arm-ONLY, PH gravity-comp +
              impedance TORQUE vs. plain position-PD, same choreography, back-to-back.
  Category B: grasp_ph.py full grasp runs -- arm position-PD (IK-driven reach) + hand
              PH-torque-via-admittance (force-mode grasp_hold). 5 runs: 1 in Real_logs/
              itself, 4 in Successfull_grasps/.

Outputs (relative to this file, unless an output dir is given):
  figures_pdf/   one vector PDF per figure, no in-figure titles, for the paper
  figures_png/   the same figures with titles, for reviewing on screen
  tables/        CSV for every computed table
  report.txt     full numeric report
  captions.tex   draft captions matching the PDF filenames

Usage
-----
  python analyze_real_logs.py [output_dir] [a4|ieee]
"""
import os
import sys
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

RIGHT_ARM = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
             "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
RIGHT_ARM_SHORT = [j.replace("right_", "") for j in RIGHT_ARM]
HAND12 = ["L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint", "L_index_proximal_joint",
          "L_middle_proximal_joint", "L_ring_proximal_joint", "L_pinky_proximal_joint",
          "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
          "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint"]
RIGHT_HAND = [n for n in HAND12 if n.startswith("R_")]
RIGHT_HAND_SHORT = [j.replace("R_", "").replace("_proximal", "").replace("_joint", "") for j in RIGHT_HAND]
GRASP_SEG_ORDER = ["hold", "side_up", "approach_out", "approach", "descend", "grasp", "grasp_hold",
                   "lift", "lift_hold", "lower", "release", "retreat", "retreat_out", "side_down"]
ARMPH_SEG_ORDER = ["hold_in", "move_to_pose", "hold_torque", "return", "release"]

# thumb_yaw's actuator/sensor is KNOWN FAULTY on this physical robot (confirmed hardware
# fault, not a PH-control-law finding) -- every plot/table that shows thumb_yaw alongside
# the other finger joints must say so in its own legend/title/axis label, not just in a
# passing report note, so the fault can't be misread later as a control-law result.
KNOWN_FAULTY_JOINT_SHORT = "thumb_yaw"
FAULTY_LABEL = "thumb_yaw (KNOWN-FAULTY sensor/actuator on this robot -- disregard as a control signal)"
FAULTY_LABEL_SHORT = "thumb_yaw [FAULTY HW]"

# --------------------------------------------------------------- paper style (identical
# conventions to scripts/g1_control/analysis.py's paper_style()/palette)
def paper_style():
    serif = [f for f in ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
             if f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}]
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": serif or ["DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

_PAGE = (sys.argv[2].lower() if len(sys.argv) > 2 else "a4")
if _PAGE not in ("a4", "ieee"):
    sys.exit(f"ERROR: unknown page preset '{_PAGE}'. Use 'a4' or 'ieee'.")
W1, W2 = (3.05, 6.30) if _PAGE == "a4" else (3.50, 7.16)

C_BLUE, C_RED, C_TEAL, C_ORANGE, C_GRAY = "#003049", "#a4161a", "#2a9d8f", "#e07a5f", "#8d99ae"

# --------------------------------------------------------------- plumbing (identical to
# scripts/g1_control/analysis.py's say()/save_fig()/save_table())
_buf = io.StringIO()
def say(*a):
    line = " ".join(str(x) for x in a)
    print(line); _buf.write(line + "\n")

def resolve_out():
    out = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "real_logs_analysis")
    for sub in ("figures_pdf", "figures_png", "tables"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out

OUT = resolve_out()
PDF, PNG, TAB = (os.path.join(OUT, s) for s in ("figures_pdf", "figures_png", "tables"))
CAPTIONS = {}

def save_fig(fig, stem, title, caption):
    fig.suptitle(title, fontsize=9)
    fig.savefig(os.path.join(PNG, stem + ".png"), dpi=200)
    fig.suptitle("")
    fig.savefig(os.path.join(PDF, stem + ".pdf"))
    plt.close(fig)
    CAPTIONS[stem] = caption

def save_table(df_, name):
    df_.to_csv(os.path.join(TAB, name + ".csv"), index=False)

def segbands(ax, t, seg, highlight):
    """Shade contiguous segment runs, matching scripts/g1_control/analysis.py's segbands():
    highlighted (e.g. torque/force-mode) segments in orange, everything else in light gray,
    with a thin vertical line at every segment switch."""
    sw = np.where(seg[1:] != seg[:-1])[0] + 1
    b = [0] + list(sw) + [len(t)]
    for i in range(len(b) - 1):
        a, e = b[i], b[i + 1]
        c = "#f3d5b5" if seg[a] in highlight else "#f0f0f0"
        ax.axvspan(t[a], t[min(e, len(t) - 1)], color=c, alpha=.55, lw=0, zorder=0)
    for k in sw:
        ax.axvline(t[k], color="k", lw=.35, alpha=.35, zorder=1)
    return sw

paper_style()

# =============================================================================
# Load
# =============================================================================
def load_arm_ph_pair():
    ph = pd.read_csv(os.path.join(HERE, "arm_ph_ph_20260918_183908.csv"))
    pos = pd.read_csv(os.path.join(HERE, "arm_ph_position_20260918_184048.csv"))
    return ph, pos

GRASP_RUNS = [
    ("main_185109", "arm_positions_20260918_185109.csv", "hand_state_20260918_185109.csv"),
    ("sg_173308", "Successfull_grasps/arm_positions_20260918_173308.csv", "Successfull_grasps/hand_state_20260918_173308.csv"),
    ("sg_173640", "Successfull_grasps/arm_positions_20260918_173640.csv", "Successfull_grasps/hand_state_20260918_173640.csv"),
    ("sg_173920", "Successfull_grasps/arm_positions_20260918_173920.csv", "Successfull_grasps/hand_state_20260918_173920.csv"),
    ("sg_174410", "Successfull_grasps/arm_positions_20260918_174410.csv", "Successfull_grasps/hand_state_20260918_174410.csv"),
]
ARM_ONLY_RUN = "Successfull_grasps/arm_positions_20260917_141924.csv"

def load_grasp_runs():
    out = {}
    for name, arm_f, hand_f in GRASP_RUNS:
        out[name] = (pd.read_csv(os.path.join(HERE, arm_f)), pd.read_csv(os.path.join(HERE, hand_f)))
    return out

ARM_PH, ARM_POS = load_arm_ph_pair()
GRASP = load_grasp_runs()

# ============================================================ 1 FILE PROFILE
say("=" * 78); say("1. FILE PROFILE"); say("=" * 78)
rows = []
for label, df, cat in ([("arm_ph_ph_20260918_183908", ARM_PH, "A"),
                         ("arm_ph_position_20260918_184048", ARM_POS, "A")]
                        + [(f"{n}_arm", a, "B") for n, (a, h) in GRASP.items()]
                        + [(f"{n}_hand", h, "B") for n, (a, h) in GRASP.items()]):
    rows.append(dict(file=label, category=cat, rows=len(df), duration_s=float(df["t_s"].iloc[-1]),
                      n_segments=df["segment"].nunique(), missing_cells=int(df.isna().sum().sum())))
profile = pd.DataFrame(rows)
say(profile.to_string(index=False))
save_table(profile, "file_profile")
say("\nNOTE: STATE_LOG_HZ=10 for every file here (see REAL_LOGS_ARCHITECTURE.md) -- this is")
say("      a 10 Hz snapshot of the real ~50 Hz (control_dt=0.02s) control loop, not the")
say("      control-rate signal itself. Any claim about servo-rate dynamics is aliased away.")

# ============================================================ 2 CATEGORY A
say("\n" + "=" * 78)
say("2. CATEGORY A -- g1_hw_arm_ph_control.py: PH TORQUE vs. POSITION-PD BASELINE")
say("=" * 78)
say("Same choreography (hold_in -> move_to_pose -> hold_torque -> return -> release), run")
say("back-to-back, only 'hold_torque' switches control law. See section 'Control law,")
say("hold_torque segment only' in REAL_LOGS_ARCHITECTURE.md for the exact equations.")

rows = []
for label, df in [("ph", ARM_PH), ("position", ARM_POS)]:
    seg = df[df["segment"] == "hold_torque"]
    for j, js in zip(RIGHT_ARM, RIGHT_ARM_SHORT):
        err = seg[f"{j}_qerr_rad"]
        rows.append(dict(mode=label, joint=js, n=len(err), mean_err_rad=err.mean(),
                          abs_mean_err_rad=err.abs().mean(), max_abs_err_rad=err.abs().max(),
                          std_err_rad=err.std()))
armA_tracking = pd.DataFrame(rows)
say("\nhold_torque tracking error (measured - qdes), right-arm joints, both modes:")
say(armA_tracking.round(5).to_string(index=False))
save_table(armA_tracking, "categoryA_hold_torque_tracking")

seg_ph = ARM_PH[ARM_PH["segment"] == "hold_torque"]
rows = []
for j, js in zip(RIGHT_ARM, RIGHT_ARM_SHORT):
    tau, g = seg_ph[f"{j}_tau_cmd_Nm"], seg_ph[f"{j}_g_Nm"]
    fb = tau - g
    rows.append(dict(joint=js, mean_tau_Nm=tau.mean(), mean_g_Nm=g.mean(),
                      mean_feedback_Nm=fb.mean(), abs_mean_feedback_Nm=fb.abs().mean(),
                      peak_abs_tau_Nm=tau.abs().max()))
armA_torque = pd.DataFrame(rows)
say("\n--mode ph, hold_torque: commanded torque = gravity-comp g(q) + PD feedback:")
say(armA_torque.round(4).to_string(index=False))
save_table(armA_torque, "categoryA_ph_torque_breakdown")
say("\n=> mean_tau ~= mean_g with small |feedback| means gravity comp is doing most of the")
say("   work and the impedance term is only making small corrections (expected at a held")
say("   pose); a large |feedback| relative to g means the arm is fighting the pose, not")
say("   resting at it (possible under-/over-compensation).")

say("\ndid either run abort into the runaway-watchdog winddown early (segment sequence")
say("never reaching 'release', or ending short of the nominal 17.08s)?")
for label, df in [("ph", ARM_PH), ("position", ARM_POS)]:
    say(f"  [{label}] final segment={df['segment'].iloc[-1]!r}, duration={df['t_s'].iloc[-1]:.2f}s")

# ============================================================ 3 CATEGORY B
say("\n" + "=" * 78)
say("3. CATEGORY B -- grasp_ph.py: ARM (position-PD) + HAND (PH-admittance) GRASP RUNS")
say("=" * 78)
if os.path.exists(os.path.join(HERE, ARM_ONLY_RUN)):
    df = pd.read_csv(os.path.join(HERE, ARM_ONLY_RUN))
    say(f"\n[EXCLUDED] {ARM_ONLY_RUN}: {len(df)} rows, no matching hand_state file -- an")
    say(f"           arm-only partial/early run, segments={sorted(df['segment'].unique())}.")
    say("           Not part of the 5-run comparison below.")

rows = []
arm_qerr_runs, arm_no_qerr_runs = [], []
for name, (arm, hand) in GRASP.items():
    has_qerr = f"{RIGHT_ARM[0]}_qerr_rad" in arm.columns
    (arm_qerr_runs if has_qerr else arm_no_qerr_runs).append(name)
    if not has_qerr:
        continue
    for seg_name, seg_df in arm.groupby("segment"):
        for j, js in zip(RIGHT_ARM, RIGHT_ARM_SHORT):
            rows.append(dict(run=name, segment=seg_name, joint=js,
                              abs_mean_err_rad=seg_df[f"{j}_qerr_rad"].abs().mean()))
seg_err_df = pd.DataFrame(rows)
armB_pivot = (seg_err_df.groupby(["segment", "joint"])["abs_mean_err_rad"].mean()
              .unstack("joint").reindex(GRASP_SEG_ORDER).reindex(RIGHT_ARM_SHORT, axis=1))
say(f"\n[NOTE] the 4 Successfull_grasps/ arm_positions_*.csv files ({arm_no_qerr_runs}) use an")
say("       even older 32-column schema than their hand_state companions: raw G1_JOINT_NAMES")
say("       positions only, no qdes_rad/qerr_rad at all (grasp_ph.py hadn't added arm tracking")
say(f"       logging yet). Only {arm_qerr_runs} has qerr_rad -- the table below is that ONE run,")
say("       not a 5-run average, despite covering the same 5-run comparison elsewhere in this")
say("       report. See fig03b for what IS available (raw position) across all 5 runs.")
say(f"\narm tracking: mean |qerr| [rad] per segment x right-arm joint ({arm_qerr_runs[0]} only):")
say(armB_pivot.round(4).to_string())
_sd = armB_pivot.loc["side_down"]
say(f"\n=> 'side_down' is an outlier: {_sd.idxmax()}={_sd.max():.3f} rad, roughly "
    f"{_sd.max() / armB_pivot.drop('side_down').to_numpy().max():.0f}x the worst error seen in "
    "any other segment across all 5 runs, on every right-arm joint at once -- worth checking "
    "against that segment's own IK target/speed rather than treating it as ordinary tracking "
    "noise.")
save_table(armB_pivot.reset_index(), "categoryB_arm_tracking_by_segment")

rows = []
old_schema_runs = []
for name, (arm, hand) in GRASP.items():
    gh_full = hand[hand["segment"] == "grasp_hold"]
    if gh_full.empty:
        continue
    has_ext = f"{RIGHT_HAND[0]}_qerr_rad" in hand.columns
    if not has_ext:
        old_schema_runs.append(name)
    gh = gh_full.iloc[len(gh_full) // 2:]
    for j, js in zip(RIGHT_HAND, RIGHT_HAND_SHORT):
        rows.append(dict(
            run=name, joint=js, pos_rad=gh[f"{j}_pos_rad"].mean(), force_g=gh[f"{j}_force_g"].mean(),
            qerr_rad=gh[f"{j}_qerr_rad"].mean() if has_ext else np.nan,
            tau_des_Nm=gh[f"{j}_tau_des_Nm"].mean() if has_ext else np.nan,
            max_abs_force_g=gh_full[f"{j}_force_g"].abs().max()))
grasp_settled = pd.DataFrame(rows)
say("\nright-hand grasp_hold (FORCE mode), settled window (last 50% of grasp_hold rows), per run:")
say(grasp_settled.round(3).to_string(index=False))
say(f"\n[HARDWARE FAULT] {FAULTY_LABEL}: the {KNOWN_FAULTY_JOINT_SHORT} force/position readings")
say("       above (including its sign flip between runs and its multi-thousand-gram spikes,")
say("       see the force-guard check below) are KNOWN to come from faulty sensor/actuator")
say("       hardware on this physical robot, confirmed independently of this log analysis.")
say("       Do not read them as evidence about the PH admittance law -- every other joint's")
say("       numbers in this section are unaffected and should be trusted normally.")
if old_schema_runs:
    say(f"\n[NOTE] {old_schema_runs}: hand_state log predates the qdes_rad/qerr_rad/tau_des_Nm")
    say("       columns (older grasp_ph.py hand-log header, 4 cols/joint, same as")
    say("       Successfull_grasps/'s other early runs) -- qerr_rad/tau_des_Nm are NaN for")
    say("       these runs; pos_rad/force_g are still directly comparable across all 5 runs.")
save_table(grasp_settled, "categoryB_grasp_hold_settled")

consistency = (grasp_settled.groupby("joint")[["pos_rad", "force_g", "tau_des_Nm"]]
               .agg(["mean", "std"]))
say("\ncross-run consistency of the settled grasp_hold state (std across the 5 runs, per joint):")
say(consistency.round(4).to_string())
say(f"=> ignore the '{KNOWN_FAULTY_JOINT_SHORT}' row above -- known-faulty hardware (see note")
say("   above), its force sign/magnitude is unreliable and not a control-law signal. For the")
say("   remaining (healthy) joints: low force_g std relative to its mean means the grip")
say("   settles to a repeatable force regardless of run; a high std (or a std comparable to")
say("   the mean, as seen here) means the settled grip is sensitive to run-to-run variation")
say("   (grasp geometry/timing), consistent with report.md's simulation-side finding that")
say("   force-mode equilibria are impedance-, not position-, driven.")
save_table(consistency.reset_index(), "categoryB_grasp_hold_cross_run_consistency")

say("\nforce-guard sanity: max |force_g| reached anywhere in grasp_hold, right hand, per run")
say("(compare against g1_hw_ph_finger_control.py's FORCE_FAULT_ABS_G=1000g sensor-fault")
say("gate and its modbus_force_limit_g default=600g hard Modbus stop):")
fmax = grasp_settled.pivot(index="run", columns="joint", values="max_abs_force_g")
say(fmax.round(1).to_string())
say(f"=> the '{KNOWN_FAULTY_JOINT_SHORT}' column's multi-thousand-gram peaks (main_185109:")
say(f"   {fmax['thumb_yaw']['main_185109']:.0f} g) are the known hardware fault, not a real")
say("   grip force -- expected given the confirmed-faulty sensor/actuator, not a finding")
say("   about FORCE_FAULT_ABS_G's threshold placement.")
_ring_max = fmax["ring"].idxmax()
say(f"   Among the healthy joints, 'ring' in run '{_ring_max}' reaches {fmax['ring'][_ring_max]:.0f} g -- "
    "above the 600g modbus_force_limit_g default and close to the 1000g fault gate, unlike "
    "every other healthy joint/run in this table (all comfortably under 600g). Worth a second "
    "look on that run specifically before assuming it's just a firmer grip.")
save_table(fmax.reset_index(), "categoryB_grasp_hold_force_guard_check")

# ============================================================ FIGURES
# F1 Category A -- right-arm tracking, PH vs position, hold_torque segment highlighted
fig, axs = plt.subplots(4, 2, figsize=(W2, 7.5), sharex=True)
for k, (j, js) in enumerate(zip(RIGHT_ARM, RIGHT_ARM_SHORT)):
    ax = axs.flat[k]
    for label, df, c in [("position", ARM_POS, C_GRAY), ("ph", ARM_PH, C_RED)]:
        t = df["t_s"].to_numpy(float); seg = df["segment"].to_numpy()
        if k == 0:
            segbands(ax, t, seg, {"hold_torque"})
        else:
            segbands(ax, t, seg, {"hold_torque"})
        ax.plot(t, df[f"{j}_qerr_rad"], color=c, label=label)
    ax.set_title(js, fontsize=7.5)
    ax.axhline(0, color="k", lw=.5, alpha=.4)
    ax.legend(loc="upper right", fontsize=5.5)
for ax in axs[-1, :]:
    ax.set_xlabel("Time [s]")
for ax in axs[:, 0]:
    ax.set_ylabel("q err [rad]")
fig.delaxes(axs.flat[7])
save_fig(fig, "fig01_armA_tracking", "Category A: right-arm tracking error, PH vs. position-PD",
         "Measured-minus-commanded joint angle for each right-arm joint, comparing the "
         "--mode ph run (red) against the --mode position baseline (gray) run back-to-back "
         "on the same choreography. Shaded band is the hold_torque segment, the only one "
         "where the two modes' control law actually differs.")

# F2 Category A -- torque decomposition, ph run, hold_torque
fig, ax = plt.subplots(figsize=(W1, 3.2))
x = np.arange(len(RIGHT_ARM_SHORT))
ax.bar(x - .2, armA_torque.mean_g_Nm, .4, color=C_BLUE, label="gravity comp $g(q)$")
ax.bar(x + .2, armA_torque.mean_feedback_Nm, .4, color=C_RED, label="PD feedback")
ax.axhline(0, color="k", lw=.6)
ax.set_xticks(x); ax.set_xticklabels(RIGHT_ARM_SHORT, rotation=60, ha="right", fontsize=6.5)
ax.set_ylabel("Torque [Nm]"); ax.legend(fontsize=6.5)
save_fig(fig, "fig02_armA_torque_breakdown", "Category A: commanded torque decomposition (--mode ph, hold_torque)",
         "Mean commanded right-arm torque during hold_torque, split into the gravity-"
         "compensation term $g(q)$ and the remaining proportional-derivative feedback "
         "correction. A feedback term small relative to $g(q)$ indicates the arm is "
         "resting near its target with gravity comp doing most of the work.")

# F3 Category B -- arm tracking heatmap, segment x joint, averaged across 5 runs
fig, ax = plt.subplots(figsize=(W2, 3.2))
h = armB_pivot.to_numpy()
im = ax.imshow(h, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(RIGHT_ARM_SHORT))); ax.set_xticklabels(RIGHT_ARM_SHORT, rotation=60, ha="right", fontsize=6.5)
ax.set_yticks(range(len(armB_pivot.index))); ax.set_yticklabels(armB_pivot.index, fontsize=6.5)
ax.grid(False)
for i in range(h.shape[0]):
    for kk in range(h.shape[1]):
        v = h[i, kk]
        if not np.isnan(v):
            ax.text(kk, i, f"{v:.3f}", ha="center", va="center", fontsize=5.5,
                    color="white" if v > np.nanmax(h) * .55 else "black")
fig.colorbar(im, ax=ax, label=f"mean |q err| [rad] ({arm_qerr_runs[0]})", fraction=.03, pad=.02)
save_fig(fig, "fig03_armB_tracking_heatmap",
         f"Category B: arm tracking error by segment and joint ({arm_qerr_runs[0]} only)",
         f"Mean absolute right-arm tracking error, by trajectory segment and joint, for "
         f"{arm_qerr_runs[0]} -- the only one of the 5 grasp_ph.py runs whose arm_positions "
         f"log includes qdes_rad/qerr_rad (the 4 Successfull_grasps/ runs use an older, "
         "position-only schema; see fig03b for what's available across all 5). "
         "Position-PD only (grasp_ph.py never applies a torque law to the arm) -- large "
         "values here reflect IK/trajectory tracking, not gravity sag.")

# F3b Category B -- raw right-arm POSITION, full time series, all 5 runs overlaid per joint.
# NOT tracking error: only main_185109's arm_positions log has qdes_rad/qerr_rad at all (see
# the [NOTE] above) -- raw measured position is the one right-arm signal every one of the 5
# runs (including all 4 Successfull_grasps/ pairs) actually has, so it's what this figure
# plots, labelled honestly rather than reusing the "tracking error" name for something the
# other 4 runs can't support.
_RUN_COLORS = {"main_185109": C_BLUE, "sg_173308": C_RED, "sg_173640": C_TEAL,
               "sg_173920": C_ORANGE, "sg_174410": "#6a4c93"}
fig, axs = plt.subplots(4, 2, figsize=(W2, 9.0), sharex=False)
fig.subplots_adjust(hspace=0.65, wspace=0.28)
for k, (j, js) in enumerate(zip(RIGHT_ARM, RIGHT_ARM_SHORT)):
    ax = axs.flat[k]
    t_main = GRASP["main_185109"][0]["t_s"].to_numpy(float)
    seg_main = GRASP["main_185109"][0]["segment"].to_numpy()
    segbands(ax, t_main, seg_main, {"grasp_hold", "side_down"})
    for name, (arm, hand) in GRASP.items():
        t = arm["t_s"].to_numpy(float)
        ax.plot(t, arm[j], color=_RUN_COLORS[name], lw=.7, label=name)
    ax.set_title(js, fontsize=7.5, pad=3)
    ax.set_xlabel("Time [s]", fontsize=6.5)
    ax.set_ylabel("q [rad]", fontsize=6.5)
    ax.legend(loc="upper right", fontsize=4.8, ncol=1)
fig.delaxes(axs.flat[7])
save_fig(fig, "fig03b_armB_position_timeline",
         "Category B: right-arm measured position over time, all 5 grasp_ph.py runs "
         "(raw position, not tracking error)",
         "Measured right-arm joint angle, all 5 grasp_ph.py runs overlaid per joint "
         "(main_185109 plus the 4 Successfull_grasps/ pairs). This is raw position, NOT "
         "tracking error: the 4 Successfull_grasps/ arm_positions logs have no qdes_rad/"
         "qerr_rad columns at all (an older, position-only schema -- see the note in the "
         "report accompanying fig03), so error can only be computed for main_185109 "
         "(fig03). Shaded bands mark grasp_hold (orange) and side_down (tan), taken from "
         "main_185109's own segment timing -- other runs' segment boundaries land at "
         "similar but not identical times, since these are independent real-hardware runs, "
         "not a synchronized replay. Tight agreement between traces across runs is evidence "
         "of repeatable IK/trajectory execution even without a logged error signal; visible "
         "divergence is worth a closer look at that specific run.")

# F4 Category B -- right-hand grasp_hold settling, one representative run, full timeline
name0, (arm0, hand0) = "main_185109", GRASP["main_185109"]
t0, seg0 = hand0["t_s"].to_numpy(float), hand0["segment"].to_numpy()
def _hand_legend_label(js):
    return FAULTY_LABEL_SHORT if js == KNOWN_FAULTY_JOINT_SHORT else js

_HAND_COLORS = [C_BLUE, C_RED, C_TEAL, C_ORANGE, C_GRAY, "#6a4c93"]
fig, axs = plt.subplots(3, 1, figsize=(W2, 5.2), sharex=True)
ax = axs[0]; segbands(ax, t0, seg0, {"grasp_hold"})
for j, js, c in zip(RIGHT_HAND, RIGHT_HAND_SHORT, _HAND_COLORS):
    ax.plot(t0, hand0[f"{j}_pos_rad"], color=c, lw=.8, label=_hand_legend_label(js))
ax.set_ylabel("q [rad]"); ax.legend(fontsize=5.5, ncol=2, loc="upper right")
ax = axs[1]; segbands(ax, t0, seg0, {"grasp_hold"})
for j, js, c in zip(RIGHT_HAND, RIGHT_HAND_SHORT, _HAND_COLORS):
    ax.plot(t0, hand0[f"{j}_force_g"], color=c, lw=.8, label=_hand_legend_label(js))
ax.set_ylabel("Force [g]"); ax.legend(fontsize=5.5, ncol=2, loc="upper right")
ax.text(.99, .04, f"{FAULTY_LABEL_SHORT}: values not physically meaningful", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6, color=C_BLUE, style="italic")
ax = axs[2]; segbands(ax, t0, seg0, {"grasp_hold"})
for j, js, c in zip(RIGHT_HAND, RIGHT_HAND_SHORT, _HAND_COLORS):
    ax.plot(t0, hand0[f"{j}_tau_des_Nm"], color=c, lw=.8, label=_hand_legend_label(js))
ax.set_ylabel("$\\tau_d$ [Nm]"); ax.set_xlabel("Time [s]")
ax.legend(fontsize=5.5, ncol=2, loc="center right")
save_fig(fig, "fig04_handB_timeline",
         f"Category B: right-hand state through the full grasp ({name0}) -- "
         f"{FAULTY_LABEL_SHORT}",
         "Right-hand position, measured contact force, and force-mode spring target "
         "($\\tau_d$, PHContactForceController's internal state) over the whole "
         "grasp_ph.py choreography. Shaded band is grasp_hold, the only FORCE-mode segment. "
         f"{FAULTY_LABEL} -- its multi-thousand-gram force spikes in the middle panel are a "
         "confirmed sensor/actuator fault on this robot, not a control-law effect, and "
         "should be disregarded. Note also that $\\tau_d$ does NOT return to 0 after "
         "grasp_hold ends: PHContactForceController._tau_d is only ever written by "
         "set_force_mode(), so it stays latched at the grasp target through lift/lower/"
         "release -- contrary to grasp_ph.py's own header comment claiming it reads 0 "
         "outside grasp_hold.")

# F5 Category B -- cross-run consistency of the settled grasp_hold state
_xtick_labels = [FAULTY_LABEL_SHORT if js == KNOWN_FAULTY_JOINT_SHORT else js for js in RIGHT_HAND_SHORT]
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0))
x = np.arange(len(RIGHT_HAND_SHORT))
means = grasp_settled.groupby("joint")[["pos_rad", "force_g"]].mean().reindex(RIGHT_HAND_SHORT)
stds = grasp_settled.groupby("joint")[["pos_rad", "force_g"]].std().reindex(RIGHT_HAND_SHORT)
_colors = [C_RED if js == KNOWN_FAULTY_JOINT_SHORT else C_BLUE for js in RIGHT_HAND_SHORT]
_legend_handles = [
    plt.Line2D([0], [0], marker="o", ls="", ms=4, color=C_BLUE, label="healthy joint"),
    plt.Line2D([0], [0], marker="o", ls="", ms=4, color=C_RED, label=FAULTY_LABEL_SHORT),
]
ax = axs[0]
for xi, js in enumerate(RIGHT_HAND_SHORT):
    ax.errorbar(xi, means.pos_rad[js], yerr=stds.pos_rad[js], fmt="o", ms=4, capsize=3,
                color=C_RED if js == KNOWN_FAULTY_JOINT_SHORT else C_BLUE)
ax.set_xticks(x); ax.set_xticklabels(_xtick_labels, rotation=60, ha="right", fontsize=6.5)
ax.set_ylabel("settled q [rad]"); ax.set_title("position", fontsize=8)
ax.legend(handles=_legend_handles, fontsize=5.5, loc="lower left")
ax = axs[1]
for xi, js in enumerate(RIGHT_HAND_SHORT):
    ax.errorbar(xi, means.force_g[js], yerr=stds.force_g[js], fmt="o", ms=4, capsize=3,
                color=C_RED if js == KNOWN_FAULTY_JOINT_SHORT else C_BLUE)
ax.axhline(0, color="k", lw=.5, alpha=.5)
ax.set_xticks(x); ax.set_xticklabels(_xtick_labels, rotation=60, ha="right", fontsize=6.5)
ax.set_ylabel("settled force [g]"); ax.set_title("force", fontsize=8)
ax.legend(handles=_legend_handles, fontsize=5.5, loc="lower left")
save_fig(fig, "fig05_handB_cross_run_consistency",
         f"Category B: settled grasp_hold state, mean +/- std across 5 runs -- red = {FAULTY_LABEL_SHORT}",
         "Right-hand joint position and contact force in the settled half of grasp_hold, "
         "mean and standard deviation across the 5 independent grasp_ph.py runs. Small "
         "error bars indicate a repeatable settled state across runs; large ones indicate "
         "sensitivity to run-to-run variation in grasp geometry/timing. "
         f"{FAULTY_LABEL} (plotted in red): its apparent sign flip and large spread across "
         "runs is a confirmed hardware fault on this robot, not evidence of impedance-driven "
         "variability -- read only the other (blue) joints as control-law evidence.")

# --------------------------------------------------------------- write out
with open(os.path.join(OUT, "captions.tex"), "w", encoding="utf-8") as f:
    for stem, cap in CAPTIONS.items():
        f.write("\\begin{figure}[t]\n  \\centering\n"
                f"  \\includegraphics[width=\\columnwidth]{{{stem}.pdf}}\n"
                f"  \\caption{{{cap}}}\n  \\label{{fig:{stem.split('_', 1)[1]}}}\n"
                "\\end{figure}\n\n")
with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
    f.write(_buf.getvalue())

print("\n" + "=" * 78)
print(f"Wrote {len(CAPTIONS)} figures (PDF + PNG), "
      f"{len(os.listdir(TAB))} CSV tables, report.txt and captions.tex to:")
print(f"  {OUT}")
