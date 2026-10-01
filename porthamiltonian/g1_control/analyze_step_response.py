# analyze_step_response.py
"""Offline step-response diagnostic: why the HAND12 IDA-PBC law looks "linear".

This is a STANDALONE diagnostic -- it does not modify or extend the live plots
in `lift_fingerPH.py` nor the post-hoc figures in `analyze_controller.py`. It
answers a different question than either of those: "if I hand this controller
a clean step reference, does it produce the textbook underdamped response
(rise / overshoot / decaying ring / settle), or not -- and why?"

Root cause (see memory `finger-torque-bangbang-saturation` and the cited
thesis, Chan-Zheng 2023, "Tuning rules for energy-based control methods for
mechanical systems"):

  During S3-grasp-PH, `lift_fingerPH.py` sets `ph_ctrl.Kt_grasp`/`Bt_grasp`
  from `--Kt_close`/`--Bt_close`, but ALSO sets `ph_ctrl.x_des = None`. In
  `PHContactForceController.compute()` (ph/ph_impedance.py), the task-space
  term that multiplies Kt/Bt only fires when `x_des is not None` -- so
  `--Kt_close`/`--Bt_close` never reach the applied torque during S3. The
  gains that DO govern S3 are the force-mode virtual spring-damper:

      u = g_c(q) - Kf*(K_spr*(q_c - q0_spr) - tau_d) - Kd*qdot_c

  i.e. a linear mass-spring-damper with stiffness K = Kf*K_spr and damping
  Kd = --ph_kd, acting on the finger's PhysX-armature-dominated effective
  inertia (~0.01, see robot/unitree.py; the raw URDF link inertia is
  ~1e-7 and irrelevant). With the documented run values (Kf=1, K_spr=2.5,
  ph_kd=1.4) this loop is roughly 4x *overdamped*
  (Theorem 3.2/4.4 of the cited thesis: no-overshoot iff Kd^2 >= 4*K*M;
  here 1.4^2=1.96 vs 4*2.5*0.01=0.1) -- which is why the response looks like
  a flat, ringless ramp regardless of --grasp_torque/--Bt_close/--Kt_close.

This script reproduces that overdamped response next to a "retuned" gain set
solved from the thesis's damping-ratio rule (same stiffness K, damping Kd
picked to hit a target zeta), so the classic underdamped step shape can
actually be seen -- and the printed numbers can be checked against the
closed-form formulas.

Usage (env_isaaclab, no Isaac Sim needed -- pure numpy, runs in seconds):
    python scripts/g1_control/analyze_step_response.py
    python scripts/g1_control/analyze_step_response.py --target_zeta 0.5 --joints \
        R_index_proximal_joint R_thumb_proximal_pitch_joint
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.g1_control.ph import parse_urdf, build_tree, PHSystem, HAND12_NAMES


# -----------------------------------------------------------------------------
# Real PH model (same construction as analyze_controller.py's build_ph_system)
# -----------------------------------------------------------------------------
def build_ph_system(urdf_path: str) -> PHSystem:
    urdf = parse_urdf(urdf_path)
    tree = build_tree(urdf)
    return PHSystem(tree, control_joint_names=HAND12_NAMES)


def effective_inertia(ph_sys: PHSystem, tree_idx: int, armature: float, q0: np.ndarray) -> float:
    """
    Diagonal joint-space inertia at tree_idx, plus PhysX armature.
    The raw URDF/CRBA inertia for these finger links is ~1e-7 (numerically
    negligible); `armature` is what actually regularizes/dominates the
    simulated dynamics (see robot/unitree.py). Off-diagonal coupling is
    ignored -- each selected joint is simulated as an isolated 1-DOF system,
    matching the single-joint linearization the cited thesis's tuning rules
    are derived from (Chapters 3-4, saddle-point form around an equilibrium).
    """

    M = ph_sys.M(q0)
    return float(M[tree_idx, tree_idx]) + float(armature)


def gravity_torque(ph_sys: PHSystem, tree_idx: int, q_full: np.ndarray, eps: float = 1e-6) -> float:
    """g_c(q) = dU/dq at tree_idx, via central finite difference on U --
    the same pattern PHContactForceController.gravity_torques_hand12 uses."""
    qp = q_full.copy(); qp[tree_idx] += eps
    qm = q_full.copy(); qm[tree_idx] -= eps
    return (ph_sys.U(qp) - ph_sys.U(qm)) / (2.0 * eps)


# -----------------------------------------------------------------------------
# Closed-loop simulation: force-mode IDA-PBC virtual spring-damper
#   u = g_c(q) - K*(q - target(t)) - Kd*qdot,   K = Kf*K_spr
# (target(t) plays the role of q0_spr + tau_d/K_spr -- the force-mode
#  equilibrium -- and is stepped directly here for a clean, readable plot;
#  this is algebraically the exact law PHContactForceController.compute()
#  applies in force mode, see ph_impedance.py L221-227.)
# -----------------------------------------------------------------------------
def step_train(step_times, step_targets, t):
    """Piecewise-constant target(t), holding step_targets[i] for t in
    [step_times[i], step_times[i+1])."""
    idx = np.searchsorted(step_times, t, side="right") - 1
    idx = np.clip(idx, 0, len(step_targets) - 1)
    return step_targets[idx]


def simulate(ph_sys, tree_idx, q_full0, M_eff, K, Kd, tau_clamp,
             step_times, step_targets, t_end, dt):
    n = int(round(t_end / dt)) + 1
    t = np.linspace(0.0, t_end, n)
    target = step_train(np.asarray(step_times), np.asarray(step_targets), t)

    q = np.zeros(n)
    qd = np.zeros(n)
    u_hist = np.zeros(n)
    q_full = q_full0.copy()
    q[0] = q_full[tree_idx]

    def torque(qi, qdi, ti):
        q_full[tree_idx] = qi
        tgt = step_train(np.asarray(step_times), np.asarray(step_targets), np.array([ti]))[0]
        g_c = gravity_torque(ph_sys, tree_idx, q_full)
        u = g_c - K * (qi - tgt) - Kd * qdi
        if tau_clamp is not None and tau_clamp > 0.0:
            u = np.clip(u, -tau_clamp, tau_clamp)
        return u, g_c

    def accel(qi, qdi, ti):
        u, g_c = torque(qi, qdi, ti)
        # qddot = (u - g_c) / M_eff  -- g_c is a generalized force already
        # cancelled at equilibrium by the control law's own g_c(q) term, so
        # net actuation is (u - g_c); see PHContactForceController.compute().
        return (u - g_c) / M_eff, u

    for k in range(n - 1):
        ti = t[k]
        a1, u_k = accel(q[k], qd[k], ti)
        u_hist[k] = u_k
        a2, _ = accel(q[k] + 0.5 * dt * qd[k], qd[k] + 0.5 * dt * a1, ti + 0.5 * dt)
        a3, _ = accel(q[k] + 0.5 * dt * (qd[k] + 0.5 * dt * a1), qd[k] + 0.5 * dt * a2, ti + 0.5 * dt)
        a4, _ = accel(q[k] + dt * (qd[k] + 0.5 * dt * a2), qd[k] + dt * a3, ti + dt)
        qd[k + 1] = qd[k] + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        q[k + 1] = q[k] + dt * (qd[k] + qd[k + 1]) / 2.0
    u_hist[-1], _ = torque(q[-1], qd[-1], t[-1])

    return t, q, qd, u_hist, target


# ----------------------------------------------------------------------
# Metrics: empirical (from the simulated trace) + analytic (closed-form)
# ----------------------------------------------------------------------
def step_metrics_empirical(t, q, t_step, q_before, q_target, t_step_end=None, tol=0.02):
    """Overshoot %, 10-90% rise time, 2% settling time for the step transition
    starting at t_step, restricted to before the NEXT step (t_step_end) so a
    later target change doesn't get misread as "never settled"."""
    delta = q_target - q_before
    if abs(delta) < 1e-9:
        return dict(overshoot_pct=float("nan"), rise_time=float("nan"), settle_time=float("nan"))
    mask = (t >= t_step) if t_step_end is None else ((t >= t_step) & (t < t_step_end))
    tt = t[mask] - t_step
    s = (q[mask] - q_before) / delta  # normalized: 0 -> 1

    overshoot_pct = max(0.0, (float(np.max(s)) - 1.0) * 100.0)

    i10_arr = np.where(s >= 0.1)[0]
    i90_arr = np.where(s >= 0.9)[0]
    if i10_arr.size and i90_arr.size and i90_arr[0] >= i10_arr[0]:
        rise_time = float(tt[i90_arr[0]] - tt[i10_arr[0]])
    else:
        rise_time = float("nan")

    outside = np.where(np.abs(s - 1.0) > tol)[0]
    settle_time = float(tt[outside[-1]]) if outside.size else 0.0

    return dict(overshoot_pct=overshoot_pct, rise_time=rise_time, settle_time=settle_time)


def analytic_metrics(K, Kd, M_eff):
    """Closed-form zeta / wn / overshoot / settling-time from the thesis's
    Theorem 3.2-3.4 / 4.4-4.5 (single-DOF specialization: R=Kd, P=K, W=M_eff).
    """
    wn = np.sqrt(K / M_eff)
    zeta = Kd / (2.0 * np.sqrt(K * M_eff))
    no_overshoot = bool(Kd ** 2 >= 4.0 * K * M_eff)  # Theorem 3.2 / 4.4 condition
    out = dict(wn=float(wn), zeta=float(zeta), no_overshoot=no_overshoot)
    if zeta < 1.0:
        out["overshoot_pct"] = float(np.exp(-zeta * np.pi / np.sqrt(1.0 - zeta ** 2)) * 100.0)
        out["settle_time_2pct"] = float(4.0 / (zeta * wn))  # Theorem 3.4/4.5 rise-time-bound form
        out["peak_time"] = float(np.pi / (wn * np.sqrt(1.0 - zeta ** 2)))
    else:
        # overdamped: slower of the two real poles bounds the settling time
        lam_slow = zeta * wn - wn * np.sqrt(zeta ** 2 - 1.0)
        out["overshoot_pct"] = 0.0
        out["settle_time_2pct"] = float(4.0 / lam_slow)
        out["peak_time"] = float("nan")
    return out


def kd_for_target_zeta(K, M_eff, zeta):
    return 2.0 * zeta * np.sqrt(K * M_eff)


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def make_figure(out_path, results, step_times, step_targets):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.6 * n), squeeze=False)
    for row, res in enumerate(results):
        ax = axes[row][0]
        ax.plot(res["t"], res["target"], "k--", lw=1.4, label="target (step train)")
        ax.plot(res["t"], res["q_current"], color="#d9534f", lw=1.8,
                 label=f"current gains (Kd={res['current']['Kd']:.2f}, "
                       f"ζ={res['current_metrics_analytic']['zeta']:.2f})")
        ax.plot(res["t"], res["q_retuned"], color="#337ab7", lw=1.8,
                 label=f"retuned gains (Kd={res['retuned']['Kd']:.2f}, "
                       f"ζ={res['retuned_metrics_analytic']['zeta']:.2f})")
        ax.set_ylabel("q [rad]")
        ax.set_title(res["joint_name"], fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        if row == n - 1:
            ax.set_xlabel("time [s]")

        cm = res["current_metrics_empirical"]
        rm = res["retuned_metrics_empirical"]
        ca = res["current_metrics_analytic"]
        ra = res["retuned_metrics_analytic"]
        txt = (
            f"current:  overshoot={cm['overshoot_pct']:.1f}% (analytic {ca['overshoot_pct']:.1f}%)  "
            f"rise={cm['rise_time']*1e3:.0f}ms  settle2%={cm['settle_time']*1e3:.0f}ms  "
            f"no-overshoot? {ca['no_overshoot']}\n"
            f"retuned:  overshoot={rm['overshoot_pct']:.1f}% (analytic {ra['overshoot_pct']:.1f}%)  "
            f"rise={rm['rise_time']*1e3:.0f}ms  settle2%={rm['settle_time']*1e3:.0f}ms  "
            f"no-overshoot? {ra['no_overshoot']}"
        )
        ax.text(0.01, 0.02, txt, transform=ax.transAxes, fontsize=7.5,
                 va="bottom", ha="left", family="monospace",
                 bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(report_path, results, args):
    lines = []
    lines.append("analyze_step_response.py -- diagnostic report")
    lines.append(f"URDF: {args.ph_urdf_path}")
    lines.append(f"armature={args.armature}  Kf={args.Kf}  K_spr={args.K_spr}  "
                 f"(K=Kf*K_spr={args.Kf*args.K_spr:.3f})")
    lines.append(f"current Kd (--ph_kd)={args.ph_kd}  finger_tau_clamp={args.finger_tau_clamp}")
    lines.append(f"target_zeta (retuned)={args.target_zeta}")
    lines.append("")
    for res in results:
        lines.append(f"== {res['joint_name']} ==")
        lines.append(f"  M_eff = {res['M_eff']:.5f} kg*m^2 (armature-dominated)")
        for label, key_m, key_a in (("current", "current_metrics_empirical", "current_metrics_analytic"),
                                     ("retuned", "retuned_metrics_empirical", "retuned_metrics_analytic")):
            gset = res[label]
            m = res[key_m]
            a = res[key_a]
            lines.append(
                f"  [{label}] K={gset['K']:.3f} Kd={gset['Kd']:.3f} "
                f"tau_clamp={gset['tau_clamp']}"
            )
            lines.append(
                f"    analytic:  zeta={a['zeta']:.3f}  wn={a['wn']:.2f} rad/s  "
                f"overshoot={a['overshoot_pct']:.1f}%  settle2%={a['settle_time_2pct']*1e3:.0f}ms  "
                f"no-overshoot condition (Kd^2>=4*K*M)? {a['no_overshoot']}"
            )
            lines.append(
                f"    simulated: overshoot={m['overshoot_pct']:.1f}%  "
                f"rise(10-90%)={m['rise_time']*1e3:.0f}ms  settle2%={m['settle_time']*1e3:.0f}ms"
            )
        lines.append("")
    text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(text + "\n")
    print(text)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ph_urdf_path", type=str,
                     default="unitree_g1_inspire_assets/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")
    ap.add_argument("--joints", type=str, nargs="+",
                     default=["R_index_proximal_joint", "R_thumb_proximal_pitch_joint"],
                     help="HAND12 joint names to simulate (each treated as an isolated 1-DOF system).")
    ap.add_argument("--armature", type=float, default=0.01,
                     help="PhysX armature added to the joint-space inertia (matches robot/unitree.py).")
    ap.add_argument("--Kf", type=float, default=1.0, help="Force-loop gain (matches --Kf default).")
    ap.add_argument("--K_spr", type=float, default=2.5, help="Virtual spring stiffness (matches --K_spr default).")
    ap.add_argument("--ph_kd", type=float, default=1.4, help="Current run's damping gain (--ph_kd).")
    ap.add_argument("--finger_tau_clamp", type=float, default=1.0,
                     help="Torque clamp applied to the CURRENT-gains trace (0 = no clamp).")
    ap.add_argument("--target_zeta", type=float, default=0.6,
                     help="Damping ratio to retune Kd for, at fixed K=Kf*K_spr (Theorem 3.3/4.4).")
    ap.add_argument("--clamp_retuned", action="store_true",
                     help="Also apply --finger_tau_clamp to the retuned trace (off by default, to show "
                          "the ideal unsaturated classic response).")
    ap.add_argument("--step_amplitude", type=float, default=0.7,
                     help="Step size [rad] for the open<->closed target transitions.")
    ap.add_argument("--step_times", type=float, nargs="+", default=[0.0, 0.3, 3.2],
                     help="Times at which the target steps to the corresponding --step_targets value. "
                          "The closed-hold duration (times[2]-times[1]) must comfortably exceed the "
                          "current gains' settling time or the empirical settle-time metric gets "
                          "right-censored by the window.")
    ap.add_argument("--t_end", type=float, default=4.2)
    ap.add_argument("--dt", type=float, default=1.0 / 120.0, help="Integration step (matches typical sim dt).")
    ap.add_argument("--out", type=str, default=None,
                     help="Output figure path. Default: next to this script, diag_step_response.png")
    args = ap.parse_args()

    urdf_path = args.ph_urdf_path
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.join(REPO_ROOT, urdf_path)
    ph_sys = build_ph_system(urdf_path)
    name_to_tree = {n: i for i, n in enumerate(ph_sys.tree.joint_names)}

    step_targets = [0.0, args.step_amplitude, 0.0]
    if len(args.step_times) != len(step_targets):
        raise ValueError(
            f"--step_times must have {len(step_targets)} entries to match the built-in "
            f"open->closed->open target sequence {step_targets}; got {args.step_times}"
        )

    K = args.Kf * args.K_spr
    Kd_current = args.ph_kd

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out or os.path.join(out_dir, "diag_step_response.png")
    report_path = os.path.splitext(out_path)[0] + "_report.txt"

    results = []
    for jname in args.joints:
        if jname not in name_to_tree:
            raise ValueError(f"Joint '{jname}' not found in URDF tree (checked ph_sys.tree.joint_names)")
        tidx = name_to_tree[jname]
        q_full0 = np.zeros(ph_sys.tree.nb)
        M_eff = effective_inertia(ph_sys, tidx, args.armature, q_full0)
        Kd_retuned = kd_for_target_zeta(K, M_eff, args.target_zeta)

        t, q_cur, qd_cur, u_cur, target = simulate(
            ph_sys, tidx, q_full0.copy(), M_eff, K, Kd_current, args.finger_tau_clamp,
            args.step_times, step_targets, args.t_end, args.dt,
        )
        _, q_re, qd_re, u_re, _ = simulate(
            ph_sys, tidx, q_full0.copy(), M_eff, K, Kd_retuned,
            args.finger_tau_clamp if args.clamp_retuned else 0.0,
            args.step_times, step_targets, args.t_end, args.dt,
        )

        t_step, t_step_end = args.step_times[1], args.step_times[2]
        q_before, q_target = step_targets[0], step_targets[1]
        cur_emp = step_metrics_empirical(t, q_cur, t_step, q_before, q_target, t_step_end)
        re_emp = step_metrics_empirical(t, q_re, t_step, q_before, q_target, t_step_end)
        cur_ana = analytic_metrics(K, Kd_current, M_eff)
        re_ana = analytic_metrics(K, Kd_retuned, M_eff)

        results.append(dict(
            joint_name=jname, t=t, q_current=q_cur, q_retuned=q_re, target=target,
            M_eff=M_eff,
            current=dict(K=K, Kd=Kd_current, tau_clamp=args.finger_tau_clamp),
            retuned=dict(K=K, Kd=Kd_retuned,
                         tau_clamp=(args.finger_tau_clamp if args.clamp_retuned else 0.0)),
            current_metrics_empirical=cur_emp, retuned_metrics_empirical=re_emp,
            current_metrics_analytic=cur_ana, retuned_metrics_analytic=re_ana,
        ))
        print(f"[{jname}] M_eff={M_eff:.5f}  K={K:.3f}  "
              f"Kd_current={Kd_current:.3f} (zeta={cur_ana['zeta']:.2f})  "
              f"Kd_retuned={Kd_retuned:.3f} (zeta={re_ana['zeta']:.2f})")

    make_figure(out_path, results, args.step_times, step_targets)
    write_report(report_path, results, args)
    print(f"[PLOT] saved {out_path}")
    print(f"[REPORT] saved {report_path}")


if __name__ == "__main__":
    main()
