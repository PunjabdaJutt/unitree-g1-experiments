# analyze_controller.py
"""Post-hoc control-theory diagnostics for the HAND12 IDA-PBC controller.

Reads a `lift_fingerPH.py` CSV log and rebuilds the *real* PH model (URDF ->
MultiBodyTree -> PHSystem, pure numpy, no Isaac/torch dependency at import
time) to produce four diagnostic figures grounded in the actual closed-loop
law implemented by `ph.ph_impedance.PHContactForceController`:

  position mode (S1,S2,S6,S7,S8,S9,S10):
      u = g_c(q) - Kp*e - Kd*qdot_c - Ki*z,      e = q_c - q_des
  force mode    (S3,S4,S5):
      u = g_c(q) - grad_Vd(q_c) - Kd*qdot_c,     grad_Vd = Kf*(tau_spring-tau_d)

Figures (saved next to the input log, prefixed with its timestamp):
  1. diag_phase_portraits.png  q_c vs qdot_c per joint -- limit-cycle / chatter check.
  2. diag_energy_passivity.png Mechanical H(q,p), whole-run power balance
                               (dH/dt vs sum(tau*qdot)), and a per-segment
                               *local* shaped energy Hd (reset at every
                               segment boundary, since set_force_mode() /
                               set_position_mode() re-anchor the shaping
                               potential there too) -- the Lyapunov/passivity
                               validity check for IDA-PBC (DISC Ch.2 Sec.4.2).
  3. diag_chatter_psd.png      Relative PSD of tau and qdot over the
                               S3..S5 force-mode window -- quantifies the
                               per-step chatter frequency content so
                               `--min_vel_iters` sweeps can be compared.
  4. diag_control_authority.png Per-joint torque-saturation duty cycle (how
                               much of each mode the joint spends pinned at the
                               torque rail) plus a model-free CHATTER INDEX
                               (lag-1 autocorrelation of tau and qdot over the
                               S3..S5 force-mode window). A gain-recovery
                               (regression) panel was deliberately dropped here:
                               with the joint railed most of the run there is no
                               linear regime to identify Kp/Kd from, and a naive
                               fit on railed + phase-lagged qdot returns
                               meaningless (sign-flipped) gains. Answers "is the
                               loop even in its linear regime, or is it running
                               bang-bang?" (the top panel is the first thing to
                               read).

A text report (same prefix, `_diag_report.txt`) is written alongside the
figures with the numeric summary.

Usage (env_isaaclab):
    python scripts/g1_control/analyze_controller.py
    python scripts/g1_control/analyze_controller.py --log logs/robot_ph_..._q_tau_29.csv \\
        --ph_kp 4 --ph_kd 1.4 --ph_ki 1 --ph_i_clamp 0.35 --finger_tau_clamp 0

Gain defaults below mirror the most recently documented run
(--ph_kp 4 --ph_kd 1.4 --ph_ki 1 --ph_i_clamp 0.35 --finger_tau_clamp 0);
override them if the log you're analyzing used different CLI gains --
they are NOT stored in the CSV itself.

-------------------------------------------------------------------------
PH-THEORY MAPPING (van der Schaft, DISC lecture notes "Port-Hamiltonian
Systems Theory: An Introductory Overview" -- Ch.1 lumped PH systems, Ch.2
control of PH systems / IDA-PBC; Ch.3 is distributed-parameter and does
not apply here, the hand is a finite-dimensional lumped tree)
-------------------------------------------------------------------------
This whole module is a numerical check of the DISC Ch.1 identity

    xdot = (J - R) grad(H)(x) + G u,   y = G^T grad(H)(x)
    dH/dt = u^T y  -  grad(H)^T R grad(H)   <=  u^T y      (passivity)

evaluated on the REAL model (`PHSystem`: J_struct/R_struct/G_struct, H=T+U
via CRBA) rather than trusting the design equations on paper.

  Figure 1 (phase portraits): IDA-PBC (DISC Ch.2) places the minimum of the
    shaped energy H_d at (q*, 0) -- zero momentum IS the energy minimum. A
    passive, correctly-damped closed loop is therefore a clean inward
    spiral to that point. A limit cycle / scribble is a live violation of
    the Ch.2 convergence guarantee.

  Figure 2 (H, power balance, shaped H_d): panel 1 is H(q,p) itself; panel 2
    plots the two sides of the Ch.1 power-balance identity against each
    other (dH/dt vs u^T y = sum tau*qdot; here R=0 so they should coincide
    exactly) -- large spikes are the sampled implementation injecting
    energy the continuous-time theory forbids; panel 3 is H_d = H + V_d,
    the actual IDA-PBC Lyapunov function from Ch.2 (V_d = 1/2 Kp e^2 in
    position mode, the virtual-spring shaping potential in force mode),
    reset at every set_force_mode()/set_position_mode() call because those
    re-anchor V_d to a new target.

  Figure 3 (chatter PSD): Ch.1 passivity is a continuous-time, low-frequency
    statement about smooth dissipation. Energy pinned at the sampling
    Nyquist frequency (sign-flip every step) is a discretization artifact
    the continuous PH model never predicts -- it is where the real
    implementation departs from the theory, not a property of the theory.

  Figure 4 (control authority / chatter index): IDA-PBC's Ch.2 Lyapunov
    guarantee assumes the designed input u = g_c - grad(V_d) - Kd*qdot is
    the input actually delivered. When u saturates at the torque rail most
    of the run, that law is NOT being applied -- the closed loop is a
    bang-bang relay, not the shaped PH system, which is the formal reason
    gain tuning has no effect until the rail/chatter issue is fixed first.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.g1_control.ph import (
    parse_urdf, build_tree, PHSystem,
    IK17_JOINT_NAMES, IK29_JOINT_NAMES, HAND12_NAMES, HAND12_MIMIC_JOINTS,
    fk_chain, default_urdf,
)
from scripts.g1_control.plot_style import (
    DPI, POSITION_MODE_SEGMENTS, FORCE_MODE_SEGMENTS,
    ACTUAL_STYLE, REF_STYLE,
    shade_mode, mode_legend_patches,
    torque_rails, saturation_mask, saturation_duty,
)

SHORT = [n.replace("_proximal", "").replace("_joint", "") for n in HAND12_NAMES]

_WRIST_NAMES = [n for n in IK17_JOINT_NAMES if "wrist" in n]

# Joint-group keywords accepted by --joints/--extra_joints (mirrors Jelle
# Vinke's plot.py group tokens, but resolved against our own joint-name
# constants; wrists/left_wrist/right_wrist are our own addition).
GROUP_MAP = {
    "fingers": HAND12_NAMES,
    "left_hand": HAND12_NAMES[:6],
    "right_hand": HAND12_NAMES[6:],
    "wrists": _WRIST_NAMES,
    "left_wrist": [n for n in _WRIST_NAMES if n.startswith("left_")],
    "right_wrist": [n for n in _WRIST_NAMES if n.startswith("right_")],
    "all": IK29_JOINT_NAMES,
}


# ----------------------------------------------------------------------------
# CSV loading
# ----------------------------------------------------------------------------

def load_csv(path: str):
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r]
    idx = {name: i for i, name in enumerate(header)}
    data = np.array(rows, dtype=object)
    return header, idx, data


def col(data, idx, name, dtype=float):
    return data[:, idx[name]].astype(dtype)


def stacked12(data, idx, suffix):
    """(n,12) array, HAND12 order, from per-joint '{name}{suffix}' columns."""
    return np.stack([col(data, idx, n + suffix) for n in HAND12_NAMES], axis=1)


def latest_log(log_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(log_dir, "robot_ph_*_q_tau_29.csv")))
    if not files:
        raise FileNotFoundError(f"No robot_ph_*_q_tau_29.csv logs found in {log_dir}")
    return files[-1]


def segment_bounds(seg_names: np.ndarray):
    """[(start, stop, name), ...] contiguous run-length blocks of seg_name."""
    change = np.where(seg_names[1:] != seg_names[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(seg_names)]))
    return list(zip(starts.tolist(), stops.tolist(), seg_names[starts].tolist()))


# ----------------------------------------------------------------------------
# Real PH model reconstruction
# ----------------------------------------------------------------------------

def build_ph_system(urdf_path: str) -> PHSystem:
    urdf = parse_urdf(urdf_path)
    tree = build_tree(urdf)
    return PHSystem(tree, control_joint_names=HAND12_NAMES)


def reconstruct_qtree(ph_sys: PHSystem, t, pos17, poshand12, qdothand12):
    """Full tree-ordered q(t), qdot(t) from logged sim positions.

    Only the 17 IK joints + 12 HAND12 + their 12 kinematically-mimicked
    joints are populated; every other tree DOF (legs -- not driven or
    logged by lift_fingerPH.py, a fixed-base tabletop task) is held at 0.
    That shifts U(q) by an unknown but CONSTANT offset (the fixed
    lower-body pose never changes during the run), which is irrelevant to
    every diagnostic here -- they all look at delta-H / dH/dt, never the
    absolute value of H.
    """
    name_to_tree = {n: i for i, n in enumerate(ph_sys.tree.joint_names)}
    n = len(t)
    NB = ph_sys.tree.nb
    Q = np.zeros((n, NB))
    Qdot = np.zeros((n, NB))
    dt = float(t[1] - t[0])

    for k, jn in enumerate(IK17_JOINT_NAMES):
        if jn in name_to_tree:
            ti = name_to_tree[jn]
            Q[:, ti] = pos17[:, k]
            Qdot[:, ti] = np.gradient(pos17[:, k], dt)  # not logged directly

    for k, jn in enumerate(HAND12_NAMES):
        if jn in name_to_tree:
            ti = name_to_tree[jn]
            Q[:, ti] = poshand12[:, k]
            Qdot[:, ti] = qdothand12[:, k]  # raw sim readback -- chatter included

    hand12_col = {n: k for k, n in enumerate(HAND12_NAMES)}
    for mimic_name, (parent_name, mult, off) in HAND12_MIMIC_JOINTS.items():
        if mimic_name in name_to_tree and parent_name in hand12_col:
            ti = name_to_tree[mimic_name]
            pk = hand12_col[parent_name]
            Q[:, ti] = mult * poshand12[:, pk] + off
            Qdot[:, ti] = mult * qdothand12[:, pk]

    return Q, Qdot


def compute_H(ph_sys: PHSystem, Q, Qdot):
    """Mechanical Hamiltonian H(q,p)=1/2 p^T M(q)^-1 p + U(q) at every logged
    instant, from the real reconstructed model (used by the energy/passivity
    figure)."""
    n = Q.shape[0]
    H = np.zeros(n)
    for i in range(n):
        q, qdot = Q[i], Qdot[i]
        M = ph_sys.M(q)
        p = M @ qdot
        v = np.linalg.solve(M, p)
        H[i] = 0.5 * p @ v + ph_sys.U(q)
        if i % 200 == 0:
            print(f"  [energy reconstruction] row {i}/{n}")
    return H


# ----------------------------------------------------------------------------
# Figure 1 -- phase portraits
# ----------------------------------------------------------------------------

def fig_phase_portraits(out_path, t, q_c, qdot_c, seg_bounds):
    fig, axes = plt.subplots(4, 3, figsize=(15, 13))
    axes = axes.flatten()
    for k, (n, s) in enumerate(zip(HAND12_NAMES, SHORT)):
        ax = axes[k]
        sc = ax.scatter(q_c[:, k], qdot_c[:, k], c=t, cmap="viridis", s=4, lw=0)
        for s0, _, sname in seg_bounds:
            if sname in FORCE_MODE_SEGMENTS:
                ax.scatter(q_c[s0, k], qdot_c[s0, k], marker="*", s=90,
                           facecolor="orange", edgecolor="k", zorder=5)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(s, fontsize=9)
        if k >= 9:  # bottom row only
            ax.set_xlabel("q_c [rad]", fontsize=7)
        if k % 3 == 0:  # left column only
            ax.set_ylabel("qdot_c [rad/s]", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.06, right=0.89, top=0.92, bottom=0.09,
                        hspace=0.5, wspace=0.25)
    cax = fig.add_axes((0.905, 0.30, 0.014, 0.40))
    fig.colorbar(sc, cax=cax, label="time [s]")
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green",
               markersize=7, label="state sample (fill color = time, see colorbar)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="orange",
               markeredgecolor="k", markersize=13, label="force-mode segment entry"),
        Line2D([0], [0], color="k", lw=0.6, label="qdot = 0 (equilibrium line)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Phase portraits (q_c vs qdot_c): a clean spiral into a fixed "
                 "point is healthy; a dense scribble/cloud is chatter or a limit "
                 "cycle.", fontsize=11)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------------
# Figure 2 -- energy / passivity
# ----------------------------------------------------------------------------

def fig_energy_passivity(out_path, t, H, tau_all29, qdot_all29, seg_bounds,
                          q_c, qdes_c, qdot_c, gradVd, Kp):
    dt = float(t[1] - t[0])
    dHdt_meas = np.gradient(H, dt)
    P_in = np.sum(tau_all29 * qdot_all29, axis=1)  # sum(tau_j * qdot_j), D=0 in this PHSystem

    Hd_local = np.full(len(t), np.nan)
    for s0, s1, sname in seg_bounds:
        H0 = H[s0]
        if sname in FORCE_MODE_SEGMENTS:
            Vd = np.zeros(s1 - s0)
            for k in range(1, s1 - s0):
                dq = q_c[s0 + k] - q_c[s0 + k - 1]
                Vd[k] = Vd[k - 1] + 0.5 * np.dot(gradVd[s0 + k] + gradVd[s0 + k - 1], dq)
            Hd_local[s0:s1] = (H[s0:s1] - H0) + Vd
        else:
            e = q_c[s0:s1] - qdes_c[s0:s1]
            Vd_pos = 0.5 * np.sum(Kp * e ** 2, axis=1)
            Hd_local[s0:s1] = (H[s0:s1] - H0) + (Vd_pos - Vd_pos[0])

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(t, H, color="tab:blue", lw=1.0,
                 label=r"$H(q,p)=\frac{1}{2}\,p^{T}M(q)^{-1}p + U(q)$")
    axes[0].set_ylabel("H(q,p)  [J]  (offset by fixed leg pose)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Mechanical Hamiltonian H(q,p): upward drift = energy being "
                       "pumped in (chatter), not just tracking error.", fontsize=9)

    axes[1].plot(t, dHdt_meas, color="tab:red", lw=0.8, label="dH/dt (measured, finite-diff)")
    axes[1].plot(t, P_in, color="tab:green", lw=0.8, alpha=0.8, label=r"$\sum_j \tau_j \dot q_j$ (predicted, D=0)")
    axes[1].axhline(0, color="k", lw=0.4)
    axes[1].set_ylabel("power [W]")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title(r"Power balance: $dH/dt$ vs $\sum_j \tau_j \dot q_j$ (should "
                       "coincide, D=0). Spikes = chatter injecting/removing energy "
                       "every step.", fontsize=9)

    mode_arr = np.zeros(len(t), dtype=bool)
    for s0, s1, sname in seg_bounds:
        if sname in FORCE_MODE_SEGMENTS:
            mode_arr[s0:s1] = True
    shade_mode(axes[2], t, mode_arr)
    axes[2].plot(t, Hd_local, color="tab:purple", lw=1.0,
                 label=r"$H_d^{local}(t)-H_d^{local}(t_{seg\,start})$")
    axes[2].axhline(0, color="k", lw=0.4)
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel(r"$H_d^{local}(t) - H_d^{local}(t_{seg\ start})$  [J]")
    hd_handles = [
        Line2D([0], [0], color="tab:purple", lw=1.0,
               label=r"$H_d^{local}$ (shaped closed-loop energy)"),
    ] + mode_legend_patches()
    axes[2].legend(handles=hd_handles, fontsize=8, loc="upper right")
    axes[2].set_title("Per-segment shaped energy H_d (reset at each segment): should "
                       "settle to a bounded value within every dwell (IDA-PBC "
                       "Lyapunov check).", fontsize=9)

    for s0, s1, sname in seg_bounds:
        for ax in axes:
            ax.axvline(t[s0], color="k", lw=0.4, ls=":")
        axes[0].text(t[s0], axes[0].get_ylim()[1], sname, fontsize=6,
                     rotation=90, va="top", ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return Hd_local, dHdt_meas, P_in


# ----------------------------------------------------------------------------
# Figure 3 -- chatter PSD
# ----------------------------------------------------------------------------

def _relative_psd_db(x, dt):
    n = len(x)
    win = np.hanning(n)
    xf = (x - np.mean(x)) * win
    spec = np.fft.rfft(xf)
    psd = np.abs(spec) ** 2
    freq = np.fft.rfftfreq(n, d=dt)
    psd_db = 10.0 * np.log10(psd / (np.max(psd) + 1e-30) + 1e-12)
    return freq, psd_db


def fig_chatter_psd(out_path, t, tau_c, qdot_c, seg_bounds):
    force_idx = np.concatenate([
        np.arange(s0, s1) for s0, s1, sname in seg_bounds if sname in FORCE_MODE_SEGMENTS
    ]) if any(sname in FORCE_MODE_SEGMENTS for _, _, sname in seg_bounds) else np.arange(len(t))
    dt = float(t[1] - t[0])
    nyquist = 0.5 / dt

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    for k, (n, s) in enumerate(zip(HAND12_NAMES, SHORT)):
        ax = axes[k]
        f_tau, p_tau = _relative_psd_db(tau_c[force_idx, k], dt)
        f_qd, p_qd = _relative_psd_db(qdot_c[force_idx, k], dt)
        ax.plot(f_tau, p_tau, color="tab:blue", lw=0.9, label=r"$\tau$ (torque)")
        ax.plot(f_qd, p_qd, color="tab:red", lw=0.9, label=r"$\dot q$ (velocity)", alpha=0.8)
        ax.axvline(nyquist, color="k", lw=0.6, ls="--", label=f"Nyquist ({nyquist:.0f} Hz)")
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("Hz", fontsize=7)
        ax.set_ylabel("rel. PSD [dB]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylim(-60, 5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=True)
    fig.suptitle(f"Relative PSD over the S3..S5 force-mode window (control "
                 f"{1.0/dt:.0f} Hz, Nyquist {nyquist:.0f} Hz = dashed line). A peak "
                 f"at Nyquist = sign-flip every step = chatter; compare across "
                 f"--min_vel_iters runs.", fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------------
# Figure 4 -- control authority (saturation duty cycle) + chatter index
# ----------------------------------------------------------------------------

def detect_saturation(tau_c, finger_tau_clamp: float):
    """Per-joint saturation rail + boolean saturated-sample mask.

    The torque clamp is NOT stored in the CSV, so the rail is auto-detected
    from the data: if a joint spends a meaningful fraction of the run pinned
    at a single |tau| value, that value is the active clamp/effort limit.
    A user-supplied --finger_tau_clamp (if >0) overrides the detection.
    Returns (rails (12,), sat_mask (n,12), duty (12,)). Thin wrapper around
    plot_style's shared rail/mask/duty math (also used by lift_fingerPH.py's
    live control-authority figure)."""
    rails = torque_rails(tau_c, finger_tau_clamp)
    sat = saturation_mask(tau_c, rails)
    duty = saturation_duty(sat)
    return rails, sat, duty


def mode_duty(sat, seg_bounds):
    """Saturation duty cycle per joint, split by control mode."""
    pos = np.zeros(len(sat), dtype=bool); force = np.zeros(len(sat), dtype=bool)
    for s0, s1, sname in seg_bounds:
        if sname in POSITION_MODE_SEGMENTS:
            pos[s0:s1] = True
        elif sname in FORCE_MODE_SEGMENTS:
            force[s0:s1] = True
    duty_pos = saturation_duty(sat, sel=pos) if pos.any() else np.full(sat.shape[1], np.nan)
    duty_force = saturation_duty(sat, sel=force) if force.any() else np.full(sat.shape[1], np.nan)
    return pos, force, duty_pos, duty_force


def _lag1_autocorr(x):
    x = np.asarray(x, float)
    if len(x) < 3 or np.std(x) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def chatter_index(tau_c, qdot_c, seg_bounds):
    """Per-joint lag-1 autocorrelation of tau and qdot, over the whole run
    and over the S3..S5 force-mode window.

    Model-free chatter metric: an autocorrelation near -1 means the signal
    flips sign every single control step -- energy pinned at the Nyquist
    frequency, i.e. bang-bang / limit-cycle chatter. Near 0 = uncorrelated
    (smooth or noise-free). This is the number to watch when sweeping
    --min_vel_iters: a fix pushes tau_ac1 up toward 0."""
    force_idx = np.concatenate([
        np.arange(s0, s1) for s0, s1, sname in seg_bounds if sname in FORCE_MODE_SEGMENTS
    ]) if any(sname in FORCE_MODE_SEGMENTS for _, _, sname in seg_bounds) else np.arange(len(tau_c))
    tau_ac_all = np.array([_lag1_autocorr(tau_c[:, k]) for k in range(12)])
    qdot_ac_all = np.array([_lag1_autocorr(qdot_c[:, k]) for k in range(12)])
    tau_ac_force = np.array([_lag1_autocorr(tau_c[force_idx, k]) for k in range(12)])
    qdot_ac_force = np.array([_lag1_autocorr(qdot_c[force_idx, k]) for k in range(12)])
    return tau_ac_all, qdot_ac_all, tau_ac_force, qdot_ac_force


def fig_control_authority(out_path, duty_all, duty_pos, duty_force, rails,
                           tau_ac_force, qdot_ac_force, tau_ac_all):
    x = np.arange(12)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    w = 0.27
    axes[0].bar(x - w, 100 * duty_pos, width=w, color="tab:blue", label="position mode")
    axes[0].bar(x, 100 * duty_force, width=w, color="tab:orange", label="force mode")
    axes[0].bar(x + w, 100 * duty_all, width=w, color="tab:gray", label="whole run")
    axes[0].axhline(20, color="tab:red", lw=1.0, ls="--",
                    label="20% tuning-relevance floor")
    axes[0].set_xticks(x); axes[0].set_xticklabels(SHORT, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("time at torque rail  [%]")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[0].set_title(f"CONTROL AUTHORITY: % of time at the torque rail "
                       f"(~{np.nanmedian(rails):.2f} Nm). High = bang-bang, so gain "
                       f"tuning has no effect until the rail rises or chatter drops. "
                       f"Read this panel first.", fontsize=9)

    axes[1].axhspan(-1.0, -0.3, color="tab:red", alpha=0.07, label="chatter zone (< -0.3)")
    axes[1].bar(x - w / 2, tau_ac_force, width=w, color="tab:blue",
                label=r"$\tau$  lag-1 autocorr (force mode)")
    axes[1].bar(x + w / 2, qdot_ac_force, width=w, color="tab:red",
                label=r"$\dot q$  lag-1 autocorr (force mode)")
    axes[1].axhline(-0.3, color="tab:red", lw=1.0, ls="--", label="chatter threshold (-0.3)")
    axes[1].axhline(0.0, color="k", lw=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(SHORT, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("lag-1 autocorrelation")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].legend(fontsize=8, loc="upper right", ncol=2)
    axes[1].set_title("CHATTER INDEX (model-free): lag-1 autocorrelation of tau and "
                       "qdot over S3..S5. Near -1 = sign-flip every step (bang-bang); "
                       "near 0 = smooth. Track this when sweeping --min_vel_iters.",
                       fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------------
# General-purpose diagnostics ported from Jelle Vinke's plots/*.py scripts
# (see plan file for the piece-by-piece mapping). Unlike figures 1-4 above,
# none of these need the reconstructed PH model (H, Q, Qdot) -- they read
# directly from the logged CSV columns, so they're cheap and independent of
# --ph_kp/--ph_kd/etc.
# ----------------------------------------------------------------------------

# ---- Piece 1: avgTorque.py -- text-only average-torque-in-window report ----

def avg_torque_report(t, data, idx, t0: float, t1: float, joint: str | None = None) -> str:
    """Mean torque per joint (or a single joint) over t in [t0,t1], mirroring
    Jelle Vinke's avgTorque.py. Returns formatted text (also meant to be
    appended to the run's _diag_report.txt)."""
    m = (t >= t0) & (t <= t1)
    lines = [f"AVERAGE TORQUE over t in [{t0:.3f}, {t1:.3f}] s ({int(m.sum())} samples)"]
    if not m.any():
        lines.append("  (window contains no samples)")
        return "\n".join(lines) + "\n"
    names = [joint] if joint else IK29_JOINT_NAMES
    means = []
    for n in names:
        col_name = n + "_tau"
        if col_name not in idx:
            lines.append(f"  {n:32s}  (no '{col_name}' column)")
            continue
        v = float(np.mean(col(data, idx, col_name)[m]))
        means.append(v)
        lines.append(f"  {n:32s} {v:8.4f} Nm")
    if joint is None and means:
        lines.append(f"  {'OVERALL MEAN':32s} {float(np.mean(means)):8.4f} Nm")
    return "\n".join(lines) + "\n"


# ---- Piece 2: multi_plot.py / multi_plot_avg.py -- overlay runs for one joint ----

def fig_multi_run_torque_pos(out_prefix: str, csv_paths: list[str], labels: list[str] | None,
                              joint: str, t0: float | None, t1: float | None,
                              avg_t0: float, avg_t1: float, plot_pos: bool) -> str:
    """Overlay `{joint}_tau` (and optionally `_pos`/`_qdes`) across multiple
    CSV logs -- compare tuning runs, or one of ours against one of Jelle
    Vinke's own `*_q_tau_29.csv` logs (same column-suffix convention, so
    this works unchanged against either). Returns the per-run average-torque
    summary as text."""
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in csv_paths]
    if len(labels) != len(csv_paths):
        raise ValueError("--compare_labels must have the same length as --compare_csvs")

    runs = []
    for p in csv_paths:
        _, idx_p, data_p = load_csv(p)
        t_p = col(data_p, idx_p, "time_s")
        tau_col = joint + "_tau"
        if tau_col not in idx_p:
            raise ValueError(f"{p}: missing column '{tau_col}'")
        tau_p = col(data_p, idx_p, tau_col)
        pos_p = qdes_p = None
        if plot_pos:
            pos_col, qdes_col = joint + "_pos", joint + "_qdes"
            if pos_col not in idx_p or qdes_col not in idx_p:
                print(f"  [warn] {p}: missing '{pos_col}'/'{qdes_col}', skipping position overlay for this run")
            else:
                pos_p, qdes_p = col(data_p, idx_p, pos_col), col(data_p, idx_p, qdes_col)
        runs.append((t_p, tau_p, pos_p, qdes_p))

    report_lines = [f"AVERAGE TORQUE for '{joint}' over t in [{avg_t0:.3f}, {avg_t1:.3f}] s (per run)"]

    fig_t, ax_t = plt.subplots(figsize=(11, 6))
    for (t_p, tau_p, _, _), label in zip(runs, labels):
        m_plot = np.ones_like(t_p, dtype=bool)
        if t0 is not None: m_plot &= (t_p >= t0)
        if t1 is not None: m_plot &= (t_p <= t1)
        ax_t.plot(t_p[m_plot], tau_p[m_plot], label=label)
        m_avg = (t_p >= avg_t0) & (t_p <= avg_t1)
        avg = float(np.mean(tau_p[m_avg])) if m_avg.any() else float("nan")
        report_lines.append(f"  {label:24s} {avg:8.4f} Nm")
    ax_t.set_xlabel("Time [s]"); ax_t.set_ylabel("Torque [Nm]")
    ax_t.set_title(f"{joint} torque over time (multiple runs)")
    ax_t.grid(True); ax_t.legend(loc="upper right", fontsize=8)
    fig_t.tight_layout()
    tau_path = f"{out_prefix}_compare_tau.png"
    fig_t.savefig(tau_path, dpi=DPI)
    plt.close(fig_t)
    print(f"Saved: {tau_path}")

    if plot_pos:
        fig_p, ax_p = plt.subplots(figsize=(11, 6))
        for (t_p, _, pos_p, qdes_p), label in zip(runs, labels):
            if pos_p is None:
                continue
            m_plot = np.ones_like(t_p, dtype=bool)
            if t0 is not None: m_plot &= (t_p >= t0)
            if t1 is not None: m_plot &= (t_p <= t1)
            ax_p.plot(t_p[m_plot], pos_p[m_plot], label=f"{label} (pos)")
            ax_p.plot(t_p[m_plot], qdes_p[m_plot], "--", label=f"{label} (qdes)")
        ax_p.set_xlabel("Time [s]"); ax_p.set_ylabel("Position [rad]")
        ax_p.set_title(f"{joint} position over time (actual vs desired)")
        ax_p.grid(True); ax_p.legend(loc="upper right", fontsize=8)
        fig_p.tight_layout()
        pos_path = f"{out_prefix}_compare_pos.png"
        fig_p.savefig(pos_path, dpi=DPI)
        plt.close(fig_p)
        print(f"Saved: {pos_path}")

    return "\n".join(report_lines) + "\n"


# ---- Piece 3: plot.py / plot2.py figs 1-2 -- torque & position by joint group ----

def resolve_joint_group(tokens: list[str]) -> list[str]:
    """Expand a list of joint-name/group-keyword tokens (`all`, `fingers`,
    `left_hand`, `right_hand`, or an exact IK29_JOINT_NAMES entry) into an
    ordered, de-duplicated joint-name list."""
    all_names = set(IK29_JOINT_NAMES)
    out, seen = [], set()
    for tok in tokens:
        if tok in GROUP_MAP:
            names = GROUP_MAP[tok]
        elif tok in all_names:
            names = [tok]
        else:
            raise ValueError(f"Unknown joint or group '{tok}'. Groups: {sorted(GROUP_MAP)}; "
                              f"or an exact joint name from IK29_JOINT_NAMES.")
        for n in names:
            if n not in seen:
                seen.add(n); out.append(n)
    return out


def fig_group_torque(out_path, t, data, idx, joints: list[str], mode_arr, no_shade: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    if not no_shade:
        shade_mode(ax, t, mode_arr)
    for jn in joints:
        col_name = jn + "_tau"
        if col_name not in idx:
            print(f"  [warn] missing column '{col_name}', skipping {jn}")
            continue
        ax.plot(t, col(data, idx, col_name), label=jn, lw=1.1)
    handles, _ = ax.get_legend_handles_labels()
    if not no_shade:
        handles += mode_legend_patches()
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Torque [Nm]")
    ax.set_title("Joint torque over time")
    ax.grid(True)
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_group_pos(out_path, t, data, idx, joints: list[str], mode_arr, no_shade: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    if not no_shade:
        shade_mode(ax, t, mode_arr)
    for jn in joints:
        pos_col, qdes_col = jn + "_pos", jn + "_qdes"
        if pos_col not in idx or qdes_col not in idx:
            print(f"  [warn] missing '{pos_col}'/'{qdes_col}', skipping {jn}")
            continue
        line, = ax.plot(t, col(data, idx, pos_col), label=f"{jn} (pos)", lw=1.1)
        ax.plot(t, col(data, idx, qdes_col), "--", color=line.get_color(),
                label=f"{jn} (qdes)", lw=1.0)
    handles, _ = ax.get_legend_handles_labels()
    if not no_shade:
        handles += mode_legend_patches()
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Position [rad]")
    ax.set_title("Joint positions (actual vs desired)")
    ax.grid(True)
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---- Shared 3-row (x/y/z) plotting helper for Pieces 4/6/7 ----

def _plot_xyz_rows(axes, t, series, mode_arr=None, no_shade: bool = False) -> None:
    """series: list of (data(n,3), label, plot_kwargs). Draws x/y/z rows onto
    `axes` (length 3), optionally mode-shaded, with a shared legend on row 0."""
    row_labels = ["x", "y", "z"]
    for k in range(3):
        ax = axes[k]
        if mode_arr is not None and not no_shade:
            shade_mode(ax, t, mode_arr)
        for data_k, label, kw in series:
            ax.plot(t, data_k[:, k], label=label, **kw)
        ax.set_ylabel(f"{row_labels[k]} [m]")
        ax.grid(True, alpha=0.4)
    handles = [Line2D([0], [0], label=label, **kw) for (_, label, kw) in series]
    if mode_arr is not None and not no_shade:
        handles += mode_legend_patches()
    axes[0].legend(handles=handles, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")


# ---- Piece 4: plot.py / plot2.py fig 3 -- wrist EE Cartesian tracking ----

def reconstruct_ee_xyz(urdf, pos17: np.ndarray, qdes17: np.ndarray, side: str):
    """FK actual & desired wrist EE position (n,3 each) in the pelvis frame,
    reconstructed from the already-logged 17-DOF arm/waist joint columns.
    `side` in {'left','right'}. Uses the exact fk_chain(urdf, "pelvis",
    f"{side}_wrist_yaw_link", ...) call ik_two_hands_dls already solves
    against (ph/kinematics.py), so this composes directly with the
    fingertip/object-position figures without any frame conversion. FK on
    `_pos` recovers the real simulated EE trajectory; FK on `_qdes` recovers
    the desired Cartesian trajectory to within the logged IK's residual."""
    tip = f"{side}_wrist_yaw_link"
    n = pos17.shape[0]
    act = np.zeros((n, 3)); des = np.zeros((n, 3))
    for i in range(n):
        act[i] = fk_chain(urdf, "pelvis", tip, dict(zip(IK17_JOINT_NAMES, pos17[i])))[:3, 3]
        des[i] = fk_chain(urdf, "pelvis", tip, dict(zip(IK17_JOINT_NAMES, qdes17[i])))[:3, 3]
    return act, des


def fig_ee_tracking(out_path, t, ee_act, ee_des, mode_arr, side: str, no_shade: bool = False) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    _plot_xyz_rows(
        axes, t,
        [(ee_act, "actual", dict(ACTUAL_STYLE)),
         (ee_des, "desired (FK of q_des)", dict(REF_STYLE))],
        mode_arr, no_shade,
    )
    fig.suptitle(f"{side.capitalize()} wrist end-effector position in pelvis frame "
                 f"(FK reconstruction from logged joint angles): actual vs desired")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---- Piece 6: plotFollowBox.py "box" figure -- object camera vs ground truth ----

def fig_object_tracking(out_path, t, cube_cam, cube_gt, mode_arr, no_shade: bool = False) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    _plot_xyz_rows(
        axes, t,
        [(cube_cam, "camera estimate", dict(color="tab:blue", lw=1.3)),
         (cube_gt, "ground truth", dict(color="tab:orange", lw=1.3, ls="--"))],
        mode_arr, no_shade,
    )
    fig.suptitle("Cube center in pelvis frame: camera estimate vs ground truth")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---- Piece 7: plotFollowBox2.py "combined" figure -- object + EE overlay ----

def fig_object_ee_combined(out_path, t, cube_cam, ee_act, ee_des, mode_arr, side: str,
                            no_shade: bool = False) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    _plot_xyz_rows(
        axes, t,
        [(cube_cam, "cube (camera)", dict(color="tab:green", lw=1.3)),
         (ee_act, f"{side} EE actual", dict(color="tab:blue", lw=1.3)),
         (ee_des, f"{side} EE desired", dict(color="tab:orange", lw=1.3, ls="--"))],
        mode_arr, no_shade,
    )
    fig.suptitle(f"Cube (camera) and {side} EE positions in pelvis frame")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=str, default=None,
                    help="Path to a robot_ph_*_q_tau_29.csv log. Default: newest in --log_dir.")
    ap.add_argument("--log_dir", type=str, default=os.path.join(REPO_ROOT, "logs"))
    ap.add_argument("--ph_urdf_path", type=str,
                    default="unitree_g1_inspire_assets/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")
    ap.add_argument("--ph_kp", type=float, default=4.0)
    ap.add_argument("--ph_kd", type=float, default=1.4)
    ap.add_argument("--ph_ki", type=float, default=1.0,
                    help="Not used in the gain fit (integral state z isn't logged); "
                         "printed for reference only, its contribution is bounded by --ph_i_clamp.")
    ap.add_argument("--ph_i_clamp", type=float, default=0.35)
    ap.add_argument("--finger_tau_clamp", type=float, default=0.0,
                    help="Match the run's --finger_tau_clamp so saturated samples "
                         "can be excluded from the gain fit. 0 = no clamp (default "
                         "matches the most recently documented run).")

    # -- Opt-in Jelle-Vinke-style diagnostics (all off by default; the four
    #    figures + report above always run regardless of these flags) --
    ap.add_argument("--avg_torque_t0", type=float, default=None,
                    help="Piece 1: with --avg_torque_t1, append a mean-torque-per-joint "
                         "report (over this time window) to the _diag_report.txt.")
    ap.add_argument("--avg_torque_t1", type=float, default=None)
    ap.add_argument("--avg_torque_joint", type=str, default=None,
                    help="Restrict the avg-torque report to one joint (default: all 29).")

    ap.add_argument("--compare_csvs", type=str, nargs="+", default=None,
                    help="Piece 2: overlay --compare_joint's torque (and optionally "
                         "position) across these CSV logs -- works against any "
                         "*_q_tau_29.csv-shaped log, including Jelle Vinke's own.")
    ap.add_argument("--compare_labels", type=str, nargs="+", default=None,
                    help="Legend labels for --compare_csvs (default: basenames).")
    ap.add_argument("--compare_joint", type=str, default=None,
                    help="Required with --compare_csvs: the joint to overlay.")
    ap.add_argument("--compare_t0", type=float, default=None, help="Plot window start.")
    ap.add_argument("--compare_t1", type=float, default=None, help="Plot window end.")
    ap.add_argument("--compare_avg_t0", type=float, default=11.0,
                    help="Averaging-window start for the printed per-run average.")
    ap.add_argument("--compare_avg_t1", type=float, default=12.0,
                    help="Averaging-window end for the printed per-run average.")
    ap.add_argument("--compare_pos", action="store_true",
                    help="Also plot actual-vs-desired position overlays across runs.")
    ap.add_argument("--compare_out_prefix", type=str, default=None,
                    help="Output path prefix (default: derived from --compare_csvs[0]).")

    ap.add_argument("--joints", type=str, nargs="+", default=None,
                    help="Piece 3: joint names and/or group keywords "
                         "(all/fingers/left_hand/right_hand/wrists/left_wrist/right_wrist) -- "
                         "plots torque and position (actual vs desired) for the selected "
                         "joints, ALL ON ONE combined figure.")
    ap.add_argument("--joints_tag", type=str, default=None,
                    help="Output-filename tag for --joints (default: the tokens joined by '_').")

    ap.add_argument("--fingers", action="store_true",
                    help="Baked-in per-hand finger diagnostics: 4 figures total -- "
                         "{left,right} hand x {torque,position}, each overlaying its 6 "
                         "HAND12 fingers on one axes. Position = actual vs q_des (the "
                         "reference already logged by the sim for THIS run, not a second "
                         "CSV). No per-joint CLI calls needed. Always reads --log (or the "
                         "newest run if --log is omitted).")
    ap.add_argument("--extra_joints", type=str, nargs="+", default=None,
                    help="With --fingers: additional joint names/group keywords folded "
                         "into the matching hand's figure by name prefix (left_/L_ or "
                         "right_/R_), e.g. --extra_joints wrists, or "
                         "--extra_joints left_wrist_yaw_joint right_wrist_yaw_joint.")
    ap.add_argument("--fingers_out_dir", type=str, default=None,
                    help="Output directory for --fingers (default: same directory as --log).")

    ap.add_argument("--ee", type=str, default=None, choices=["left", "right", "both"],
                    help="Piece 4: plot wrist end-effector Cartesian tracking (actual vs "
                         "desired), reconstructed via forward kinematics on the logged "
                         "17-DOF arm/waist joint columns -- no new CSV columns needed.")

    ap.add_argument("--object_track", action="store_true",
                    help="Piece 6: plot the grasped object's camera estimate vs ground "
                         "truth. Requires cube_cam_*/cube_gt_* CSV columns (added by the "
                         "updated lift_fingerPH.py logger).")
    ap.add_argument("--object_ee_combined", type=str, default=None, choices=["left", "right"],
                    help="Piece 7: combined object (camera) + wrist-EE overlay for one "
                         "side. Requires the same cube_cam_*/cube_gt_* columns as "
                         "--object_track.")

    ap.add_argument("--no_mode_shade", action="store_true",
                    help="Disable force/position-mode background shading on the new "
                         "(non-PH-theory) figures above, for closer visual parity with "
                         "Jelle Vinke's original plots/*.py output.")
    args = ap.parse_args()

    log_path = args.log or latest_log(args.log_dir)
    print(f"Analyzing log: {log_path}")
    header, idx, data = load_csv(log_path)
    stamp = os.path.basename(log_path).replace("_q_tau_29.csv", "")
    out_dir = os.path.dirname(log_path)

    t = col(data, idx, "time_s")
    seg = np.array([str(s) for s in data[:, idx["seg_name"]]])
    bounds = segment_bounds(seg)

    pos17 = np.stack([col(data, idx, n + "_pos") for n in IK17_JOINT_NAMES], axis=1)
    q_c = stacked12(data, idx, "_pos")
    qdes_c = stacked12(data, idx, "_qdes")
    tau_all29 = np.stack([col(data, idx, n + "_tau") for n in IK17_JOINT_NAMES], axis=1)
    tau_all29 = np.concatenate([tau_all29, stacked12(data, idx, "_tau")], axis=1)
    qdot_c = stacked12(data, idx, "_qdot")
    gradVd = stacked12(data, idx, "_gradVd")
    tau_c = stacked12(data, idx, "_tau")

    urdf_path = args.ph_urdf_path
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.join(REPO_ROOT, urdf_path)
    print("Building real PH model from URDF (pure numpy, no Isaac dependency)...")
    ph_sys = build_ph_system(urdf_path)
    print(f"  tree NB={ph_sys.tree.nb}, HAND12 tree indices={ph_sys.control_indices}")

    Q, Qdot = reconstruct_qtree(ph_sys, t, pos17, q_c, qdot_c)
    qdot_all29 = np.concatenate(
        [np.gradient(pos17, t[1] - t[0], axis=0), qdot_c], axis=1
    )

    print("Reconstructing H(q,p) per timestep (~seconds)...")
    H = compute_H(ph_sys, Q, Qdot)

    fig_phase_portraits(os.path.join(out_dir, f"{stamp}_diag_phase_portraits.png"),
                         t, q_c, qdot_c, bounds)

    Hd_local, dHdt_meas, P_in = fig_energy_passivity(
        os.path.join(out_dir, f"{stamp}_diag_energy_passivity.png"),
        t, H, tau_all29, qdot_all29, bounds, q_c, qdes_c, qdot_c, gradVd,
        Kp=np.full(12, args.ph_kp),
    )

    fig_chatter_psd(os.path.join(out_dir, f"{stamp}_diag_chatter_psd.png"),
                    t, tau_c, qdot_c, bounds)

    rails, sat_mask, duty_all = detect_saturation(tau_c, args.finger_tau_clamp)
    _pos, _force, duty_pos, duty_force = mode_duty(sat_mask, bounds)
    tau_ac_all, qdot_ac_all, tau_ac_force, qdot_ac_force = chatter_index(
        tau_c, qdot_c, bounds)
    fig_control_authority(
        os.path.join(out_dir, f"{stamp}_diag_control_authority.png"),
        duty_all, duty_pos, duty_force, rails,
        tau_ac_force, qdot_ac_force, tau_ac_all,
    )

    report_path = os.path.join(out_dir, f"{stamp}_diag_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Diagnostics for {log_path}\n")
        f.write(f"Commanded gains: Kp={args.ph_kp} Kd={args.ph_kd} Ki={args.ph_ki} "
                f"i_clamp={args.ph_i_clamp} finger_tau_clamp={args.finger_tau_clamp}\n")
        f.write(f"Detected torque rail (per joint, Nm): "
                f"{np.array2string(rails, precision=2)}\n\n")
        f.write("SATURATION DUTY CYCLE (fraction of time pinned at the torque rail)\n")
        f.write(f"{'joint':16s} {'whole':>7s} {'pos':>7s} {'force':>7s}\n")
        for k, s in enumerate(SHORT):
            f.write(f"{s:16s} {duty_all[k]*100:6.1f}% {duty_pos[k]*100:6.1f}% "
                    f"{duty_force[k]*100:6.1f}%\n")
        f.write("\nCHATTER INDEX (lag-1 autocorrelation; near -1 = sign-flips every "
                "step = bang-bang)\n")
        f.write(f"{'joint':16s} {'tau_ac(all)':>12s} {'qdot_ac(all)':>13s} "
                f"{'tau_ac(force)':>14s} {'qdot_ac(force)':>15s}\n")
        for k, s in enumerate(SHORT):
            f.write(f"{s:16s} {tau_ac_all[k]:12.3f} {qdot_ac_all[k]:13.3f} "
                    f"{tau_ac_force[k]:14.3f} {qdot_ac_force[k]:15.3f}\n")
        f.write(f"\nH(q,p) range: [{H.min():.4f}, {H.max():.4f}] J "
                f"(offset by fixed leg pose, only the range/trend is meaningful)\n")
        f.write(f"max|dH/dt - sum(tau*qdot)| residual: "
                f"{np.max(np.abs(dHdt_meas - P_in)):.4f} W\n")
    print(f"Saved: {report_path}")
    print("\nDone. Four PNGs + a text report were written next to the input log.")

    # -------------------------------------------------------------------
    # Opt-in Jelle-Vinke-style figures (Pieces 1-4, 6-7). None of these need
    # the PH-model reconstruction above -- they read straight from the
    # already-loaded `data`/`idx`/`t`/`seg`, so they're cheap even when the
    # four figures above are the slow part of the run.
    # -------------------------------------------------------------------
    mode_arr_all = np.isin(seg, list(FORCE_MODE_SEGMENTS))
    extra_report_sections = []

    if args.avg_torque_t0 is not None or args.avg_torque_t1 is not None:
        if args.avg_torque_t0 is None or args.avg_torque_t1 is None:
            raise ValueError("--avg_torque_t0 and --avg_torque_t1 must be given together")
        section = avg_torque_report(t, data, idx, args.avg_torque_t0, args.avg_torque_t1,
                                     args.avg_torque_joint)
        print(section)
        extra_report_sections.append(section)

    if args.compare_csvs:
        if not args.compare_joint:
            raise ValueError("--compare_csvs requires --compare_joint")
        out_prefix = args.compare_out_prefix or os.path.join(
            out_dir, os.path.basename(args.compare_csvs[0]).replace("_q_tau_29.csv", ""))
        section = fig_multi_run_torque_pos(
            out_prefix, args.compare_csvs, args.compare_labels, args.compare_joint,
            args.compare_t0, args.compare_t1, args.compare_avg_t0, args.compare_avg_t1,
            args.compare_pos)
        print(section)
        extra_report_sections.append(section)

    if args.joints:
        joints = resolve_joint_group(args.joints)
        tag = args.joints_tag or "_".join(args.joints)
        fig_group_torque(os.path.join(out_dir, f"{stamp}_diag_torque_{tag}.png"),
                          t, data, idx, joints, mode_arr_all, args.no_mode_shade)
        fig_group_pos(os.path.join(out_dir, f"{stamp}_diag_joint_pos_{tag}.png"),
                      t, data, idx, joints, mode_arr_all, args.no_mode_shade)

    if args.fingers:
        # Baked-in joint set: the 12 HAND12 fingers, split into their two
        # hands, no per-joint CLI calls. Reference = this run's OWN logged
        # q_des (Piece 3's single-log actual-vs-desired plotter,
        # fig_group_torque/fig_group_pos) -- not a second/external CSV.
        # Exactly 4 figures out: {L,R} x {torque,pos}, each hand's 6 fingers
        # overlaid on one axes (matches _HAND_COLS's L=0:6/R=6:12 split used
        # throughout lift_fingerPH.py's own live figures).
        left_joints = list(HAND12_NAMES[:6])
        right_joints = list(HAND12_NAMES[6:])

        def _side_of(name: str) -> str | None:
            if name.startswith("left_") or name.startswith("L_"):
                return "L"
            if name.startswith("right_") or name.startswith("R_"):
                return "R"
            return None

        if args.extra_joints:
            for n in resolve_joint_group(args.extra_joints):
                side = _side_of(n)
                if side == "L" and n not in left_joints:
                    left_joints.append(n)
                elif side == "R" and n not in right_joints:
                    right_joints.append(n)
                elif side is None:
                    print(f"  [warn] --extra_joints '{n}' has no left_/L_ or right_/R_ "
                          f"prefix -- skipped (can't tell which hand's figure it belongs on)")

        fingers_out_dir = args.fingers_out_dir or out_dir
        os.makedirs(fingers_out_dir, exist_ok=True)
        fig_group_torque(os.path.join(fingers_out_dir, f"{stamp}_diag_torque_L.png"),
                          t, data, idx, left_joints, mode_arr_all, args.no_mode_shade)
        fig_group_torque(os.path.join(fingers_out_dir, f"{stamp}_diag_torque_R.png"),
                          t, data, idx, right_joints, mode_arr_all, args.no_mode_shade)
        fig_group_pos(os.path.join(fingers_out_dir, f"{stamp}_diag_pos_L.png"),
                      t, data, idx, left_joints, mode_arr_all, args.no_mode_shade)
        fig_group_pos(os.path.join(fingers_out_dir, f"{stamp}_diag_pos_R.png"),
                      t, data, idx, right_joints, mode_arr_all, args.no_mode_shade)
        print(f"Saved 4 per-hand figures (L/R x torque/pos) to {fingers_out_dir}")

    need_fk = bool(args.ee) or bool(args.object_ee_combined)
    urdf_for_fk = qdes17 = None
    if need_fk:
        urdf_for_fk = default_urdf(urdf_path)
        qdes17 = np.stack([col(data, idx, n + "_qdes") for n in IK17_JOINT_NAMES], axis=1)

    if args.ee:
        for side in (["left", "right"] if args.ee == "both" else [args.ee]):
            ee_act, ee_des = reconstruct_ee_xyz(urdf_for_fk, pos17, qdes17, side)
            fig_ee_tracking(os.path.join(out_dir, f"{stamp}_diag_ee_pos_{side}.png"),
                             t, ee_act, ee_des, mode_arr_all, side, args.no_mode_shade)

    cube_cam = cube_gt = None
    if args.object_track or args.object_ee_combined:
        needed = ["cube_cam_x", "cube_cam_y", "cube_cam_z", "cube_gt_x", "cube_gt_y", "cube_gt_z"]
        missing = [c for c in needed if c not in idx]
        if missing:
            raise ValueError(
                f"--object_track/--object_ee_combined need columns {missing} in the CSV "
                f"-- re-run lift_fingerPH.py with the updated logger to get them.")
        cube_cam = np.stack([col(data, idx, c) for c in ("cube_cam_x", "cube_cam_y", "cube_cam_z")], axis=1)
        cube_gt = np.stack([col(data, idx, c) for c in ("cube_gt_x", "cube_gt_y", "cube_gt_z")], axis=1)

    if args.object_track:
        fig_object_tracking(os.path.join(out_dir, f"{stamp}_diag_object_track.png"),
                             t, cube_cam, cube_gt, mode_arr_all, args.no_mode_shade)

    if args.object_ee_combined:
        side = args.object_ee_combined
        ee_act, ee_des = reconstruct_ee_xyz(urdf_for_fk, pos17, qdes17, side)
        fig_object_ee_combined(os.path.join(out_dir, f"{stamp}_diag_object_ee_{side}.png"),
                                t, cube_cam, ee_act, ee_des, mode_arr_all, side, args.no_mode_shade)

    if extra_report_sections:
        with open(report_path, "a") as f:
            f.write("\n\n" + "\n\n".join(extra_report_sections))
        print(f"Appended {len(extra_report_sections)} extra section(s) to: {report_path}")


if __name__ == "__main__":
    main()
