# mujoco_ph_harness.py
"""MuJoCo step-response harness for the HAND12 PH (IDA-PBC) finger controller.

Step A of the IsaacLab -> MuJoCo port. `lift_fingerPH.py`'s live
`--step_response` test (same file, ~L957 `run_step_response()`) is the
reference this mirrors line-for-line: step a subset of HAND12 joints
rest -> stepped -> rest in a real closed-loop simulation (not the offline
1-DOF ODE approximation in `analyze_step_response.py`), then report metrics
using that script's own formulas so the numbers are directly comparable.
Bypasses the camera/IK/grasp task entirely -- waist+arms hold their starting
pose via a plain PID, HAND12 joints not named in --joints hold their starting
angle too.

The `ph/` package (PHSystem, PHContactForceController, ...) is pure NumPy and
sim-agnostic -- reused completely unchanged here. Only the sim-side glue
(scene/model loading, gravity compensation, torque application, stepping)
differs from the IsaacLab reference.

Run:
  python scripts/g1_control/mujoco_ph_harness.py
  python scripts/g1_control/mujoco_ph_harness.py --mode force
  python scripts/g1_control/mujoco_ph_harness.py --joints L_index_proximal_joint --headless

Physical-fidelity notes (see scripts/g1_control/robot/unitree.py, the Isaac
reference ArticulationCfg, for the source of every number below):
  - armature=0.01 on every joint. Load-bearing: raw URDF link inertia for the
    finger segments is ~1e-7 kg.m^2, and the PH controller's gains were tuned
    assuming armature dominates the effective inertia (~0.01) -- skipping
    this makes the same gains wildly unstable in MuJoCo.
  - Legs (hip x3, knee) and feet (ankle x2) hold their spawn pose via native
    MuJoCo passive joint stiffness/damping (150/5 and 40/4 respectively,
    springref = spawn pose) -- Isaac's real implicit PD for these groups.
    They're dynamically decoupled from the waist/arm/hand subtree (both hang
    off the same fixed pelvis), so this has no effect on finger fidelity.
  - The 12 unactuated finger intermediate/distal ("mimic") joints get
    damping=0.05, zero stiffness -- a pure passive damper, matching Isaac's
    "hand_mimic_passive" actuator group.
  - Waist/arm/HAND12 joints get zero native stiffness/damping -- pure
    external torque every step, exactly like Isaac's set_joint_effort_target
    path. Waist/arm hold via a plain PID here (JointPDIController); HAND12 is
    driven by PHContactForceController.
  - The URDF's root link (pelvis) has no parent joint, so MuJoCo's URDF
    compiler fuses it into the world body automatically -- equivalent to
    Isaac's `fix_root_link=True`. Do not add a freejoint.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.g1_control.ph import (
    parse_urdf, build_tree, PHSystem,
    FingertipJacobian, PHContactForceController, JointPDIController,
    IK17_JOINT_NAMES, HAND12_NAMES,
)
from scripts.g1_control.analyze_step_response import (
    step_metrics_empirical, analytic_metrics, effective_inertia,
)
from scripts.g1_control.mujoco_common import (
    DEFAULT_URDF, TIP_LINK_NAMES, THUMB_YAW_IDX, THUMB_PITCH_IDX, ARMATURE,
    DEFAULT_JOINT_POS, load_model, apply_fidelity, joint_addrs,
)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MuJoCo closed-loop step-response test for the HAND12 PH finger controller."
    )
    p.add_argument("--urdf_path", type=str, default=DEFAULT_URDF)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--dt", type=float, default=0.002)

    p.add_argument("--joints", type=str, nargs="+",
                    default=["R_index_proximal_joint", "R_thumb_proximal_pitch_joint"],
                    help="HAND12 joint names to step (must be in HAND12_NAMES). Every other "
                         "HAND12 joint is held at its resting value when the test starts.")
    p.add_argument("--mode", type=str, default="position", choices=["position", "force"],
                    help="'position' steps q_des_hand12 by --amplitude rad (uses --ph_kp/--ph_kd); "
                         "'force' steps tau_d by --amplitude Nm via set_force_mode "
                         "(uses --Kf/--K_spr) -- the actual S3-S5 grasp law.")
    p.add_argument("--amplitude", type=float, default=0.7,
                    help="Step size above the resting level: rad (position) or Nm (force).")
    p.add_argument("--pre_hold", type=float, default=0.3)
    p.add_argument("--hold", type=float, default=2.9)
    p.add_argument("--post_hold", type=float, default=1.0)

    p.add_argument("--ph_kp", type=float, default=4.0)
    p.add_argument("--ph_kd", type=float, default=1.0)
    p.add_argument("--ph_ki", type=float, default=0.1)
    p.add_argument("--ph_i_clamp", type=float, default=0.35)
    p.add_argument("--ph_kp_thumb_yaw", type=float, default=1.0)
    p.add_argument("--Kf", type=float, default=1.0)
    p.add_argument("--K_spr", type=float, default=2.5)
    p.add_argument("--K_spr_thumb_pitch", type=float, default=14.0)
    p.add_argument("--finger_tau_clamp", type=float, default=1.0)

    p.add_argument("--kp", type=float, default=4.0, help="Waist+arm hold PID Kp")
    p.add_argument("--kd", type=float, default=1.0, help="Waist+arm hold PID Kd")
    p.add_argument("--ki", type=float, default=0.0, help="Waist+arm hold PID Ki")
    p.add_argument("--i_clamp", type=float, default=0.35)

    p.add_argument("--out", type=str, default=None,
                    help="Diagnostic PNG path (requires matplotlib). Default: no figure.")
    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = build_argparser().parse_args()

    for n in args.joints:
        if n not in HAND12_NAMES:
            raise ValueError(f"--joints: '{n}' not in HAND12_NAMES {HAND12_NAMES}")
    local_idx = {n: HAND12_NAMES.index(n) for n in args.joints}

    model = load_model(Path(args.urdf_path).resolve())
    apply_fidelity(model)
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)

    sim_joint_names, qposadr, dofadr = joint_addrs(model)

    q0 = np.array([DEFAULT_JOINT_POS.get(n, 0.0) for n in sim_joint_names])
    data.qpos[qposadr] = q0
    mujoco.mj_forward(model, data)

    # ----- PH HAND12 controller (unchanged from lift_fingerPH.py) -----
    urdf_model = parse_urdf(args.urdf_path)
    tree = build_tree(urdf_model)
    ph_sys = PHSystem(tree, control_joint_names=HAND12_NAMES)
    ftip_jac = FingertipJacobian(
        ph_sys=ph_sys, tip_link_names=TIP_LINK_NAMES,
        hand12_tree_idx=np.array(ph_sys.control_indices),
    )

    ph_kp_arr = np.full(12, args.ph_kp)
    ph_kp_arr[list(THUMB_YAW_IDX)] = args.ph_kp_thumb_yaw
    K_spr_arr = np.full(12, args.K_spr)
    K_spr_arr[list(THUMB_PITCH_IDX)] = args.K_spr_thumb_pitch

    ph_ctrl = PHContactForceController(
        ph_sys=ph_sys, ftip_jac=ftip_jac, sim_joint_names=sim_joint_names,
        Kp=ph_kp_arr, Kd=np.full(12, args.ph_kd), Ki=np.full(12, args.ph_ki),
        i_clamp=args.ph_i_clamp, Kf=args.Kf, K_spr=K_spr_arr,
    )
    print(f"[PH] IDA-PBC controller ready: NB={tree.nb}, HAND12 sim indices "
          f"{ph_ctrl.hand12_sim_indices}, {len(ph_ctrl.missing_in_sim)} tree DOFs not in sim.")

    # ----- waist+arm hold PID (no IK yet -- static target, Step B adds IK) -----
    waist_arm_idx = [sim_joint_names.index(n) for n in IK17_JOINT_NAMES]
    hold_ctrl = JointPDIController(
        num_joints=model.njnt, joint_indices=waist_arm_idx,
        Kp=args.kp, Kd=args.kd, Ki=args.ki, i_clamp=args.i_clamp,
    )
    grav_mask = np.zeros(model.njnt)
    grav_mask[waist_arm_idx] = 1.0

    effort_limits = np.array([urdf_model.joints_by_name[n].effort for n in sim_joint_names])

    viewer = None
    if not args.headless:
        import mujoco.viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    # ----- step-response schedule (mirrors lift_fingerPH.py's run_step_response) -----
    mode = args.mode
    amp = args.amplitude
    t_lo = args.pre_hold
    t_hi = t_lo + args.hold
    t_end = t_hi + args.post_hold
    n_steps = int(round(t_end / args.dt))

    q0_hand12 = data.qpos[qposadr[ph_ctrl.hand12_sim_indices]].copy()
    ph_ctrl._last_q_c = q0_hand12.copy()
    if mode == "position":
        ph_ctrl.set_position_mode()
        ph_ctrl.q_des_hand12 = q0_hand12.copy()
    else:
        ph_ctrl.set_force_mode(np.zeros(12))  # captures q0_spr=current pose, tau_d=0
    ph_ctrl.reset_integral()

    hist = {n: dict(t=[], q=[], ref=[], tau_delivered=[], tau_ideal=[]) for n in args.joints}

    print(f"[STEP] mode={mode} joints={args.joints} amplitude={amp} "
          f"pre_hold={t_lo}s hold={args.hold}s post_hold={args.post_hold}s ({n_steps} steps)")

    for k in range(n_steps + 1):
        tk = k * args.dt
        level = 1.0 if (t_lo <= tk < t_hi) else 0.0

        if mode == "position":
            q_target = q0_hand12.copy()
            for n in args.joints:
                q_target[local_idx[n]] = q0_hand12[local_idx[n]] + level * amp
            ph_ctrl.q_des_hand12 = q_target
        else:
            tau_d = np.zeros(12)
            for n in args.joints:
                tau_d[local_idx[n]] = level * amp
            ph_ctrl._tau_d = tau_d  # step tau_d only; keep the q0_spr anchor fixed

        q_sim = data.qpos[qposadr].copy()
        qd_sim = data.qvel[dofadr].copy()

        tau_ideal_12 = ph_ctrl.compute(q_sim, qd_sim, dt=args.dt)
        tau_delivered_12 = np.clip(tau_ideal_12, -args.finger_tau_clamp, args.finger_tau_clamp)

        grav = np.zeros(model.njnt)
        grav[waist_arm_idx] = data.qfrc_bias[dofadr][waist_arm_idx]
        tau_hold = hold_ctrl.compute(q_sim, qd_sim, q0, args.dt, gravity_comp=grav * grav_mask)
        tau_full = tau_hold + ph_ctrl.scatter_to_full(tau_delivered_12, model.njnt)
        tau_full = np.clip(tau_full, -effort_limits, effort_limits)

        data.qfrc_applied[dofadr] = tau_full
        mujoco.mj_step(model, data)
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()

        for n in args.joints:
            li = local_idx[n]
            if mode == "position":
                ref = q0_hand12[li] + level * amp
            else:
                ref = ph_ctrl._q0_spr[li] + (level * amp) / ph_ctrl.K_spr[li]
            hist[n]["t"].append(tk)
            hist[n]["q"].append(float(q_sim[ph_ctrl.hand12_sim_indices[li]]))
            hist[n]["ref"].append(float(ref))
            hist[n]["tau_delivered"].append(float(tau_delivered_12[li]))
            hist[n]["tau_ideal"].append(float(tau_ideal_12[li]))

        if k % 60 == 0:
            print(f"[STEP] t={tk:5.2f}s level={level:.0f}  " + "  ".join(
                f"{n}: q={hist[n]['q'][-1]:+.3f} ref={hist[n]['ref'][-1]:+.3f} "
                f"tau={hist[n]['tau_delivered'][-1]:+.2f}(ideal {hist[n]['tau_ideal'][-1]:+.2f})"
                for n in args.joints))

    if viewer is not None:
        viewer.close()

    # ----- metrics report (reusing analyze_step_response.py's formulas) -----
    report_lines = [
        "mujoco_ph_harness.py -- MuJoCo closed-loop step-response diagnostic",
        f"control_mode={mode}  amplitude={amp}  pre_hold={t_lo}s  hold={args.hold}s  "
        f"post_hold={args.post_hold}s  finger_tau_clamp={args.finger_tau_clamp}",
        "",
    ]
    figures = []
    for n in args.joints:
        li = local_idx[n]
        t_arr = np.asarray(hist[n]["t"])
        q_arr = np.asarray(hist[n]["q"])
        taui_arr = np.asarray(hist[n]["tau_ideal"])
        taud_arr = np.asarray(hist[n]["tau_delivered"])

        tree_idx = ph_ctrl.hand12_tree_indices[li]
        M_eff = effective_inertia(ph_sys, tree_idx, ARMATURE, np.zeros(ph_sys.tree.nb))
        if mode == "position":
            K = float(ph_ctrl.Kp[li])
            ref_before, ref_after = float(q0_hand12[li]), float(q0_hand12[li] + amp)
        else:
            K = float(ph_ctrl.Kf[li] * ph_ctrl.K_spr[li])
            ref_before = float(ph_ctrl._q0_spr[li])
            ref_after = float(ph_ctrl._q0_spr[li] + amp / ph_ctrl.K_spr[li])
        Kd = float(ph_ctrl.Kd[li])
        ana = analytic_metrics(K, Kd, M_eff)
        emp = step_metrics_empirical(t_arr, q_arr, t_lo, ref_before, ref_after, t_step_end=t_hi)

        report_lines += [
            f"== {n} (local idx {li}) ==",
            f"  M_eff={M_eff:.5f} kg*m^2  K={K:.3f}  Kd={Kd:.3f}",
            f"  analytic:  zeta={ana['zeta']:.3f}  wn={ana['wn']:.2f} rad/s  "
            f"overshoot={ana['overshoot_pct']:.1f}%  settle2%={ana['settle_time_2pct']*1e3:.0f}ms  "
            f"no-overshoot condition (Kd^2>=4*K*M)? {ana['no_overshoot']}",
            f"  measured:  overshoot={emp['overshoot_pct']:.1f}%  "
            f"rise(10-90%)={emp['rise_time']*1e3:.0f}ms  settle2%={emp['settle_time']*1e3:.0f}ms  "
            f"peak|tau_ideal|={np.max(np.abs(taui_arr)):.2f} Nm  "
            f"peak|tau_delivered|={np.max(np.abs(taud_arr)):.2f} Nm",
            "",
        ]
        figures.append((n, li, t_arr, q_arr, np.asarray(hist[n]["ref"]), taui_arr, taud_arr))

    report_text = "\n".join(report_lines)
    print(report_text)

    if args.out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not installed -- skipping --out figure.")
        else:
            fig, axes = plt.subplots(2 * len(figures), 1, figsize=(9, 3.0 * 2 * len(figures)),
                                      squeeze=False)
            for row, (n, li, t_arr, q_arr, ref_arr, taui_arr, taud_arr) in enumerate(figures):
                ax_q = axes[2 * row][0]
                ax_q.plot(t_arr, ref_arr, "k--", lw=1.4, label="reference")
                ax_q.plot(t_arr, q_arr, color="#d9534f", lw=1.8, label="measured q (MuJoCo)")
                ax_q.set_ylabel("q [rad]")
                ax_q.set_title(f"{n}  (mode={mode})", fontsize=10)
                ax_q.legend(fontsize=8, loc="lower right")
                ax_q.grid(alpha=0.3)

                ax_tau = axes[2 * row + 1][0]
                ax_tau.plot(t_arr, taui_arr, color="tab:orange", lw=1.3, ls="--",
                            label="PH law (pre-clip, ideal)")
                ax_tau.plot(t_arr, taud_arr, color="tab:blue", lw=1.4, label="delivered (post-clip)")
                ax_tau.set_ylabel("tau [Nm]")
                ax_tau.legend(fontsize=8, loc="lower right")
                ax_tau.grid(alpha=0.3)
                if row == len(figures) - 1:
                    ax_tau.set_xlabel("time [s]")
            fig.tight_layout()
            fig.savefig(args.out, dpi=150)
            plt.close(fig)
            print(f"[STEP] saved {args.out}")


if __name__ == "__main__":
    main()
