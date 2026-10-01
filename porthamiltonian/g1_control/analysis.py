#!/usr/bin/env python3
"""
analyze_run11_v2.py
===================

Re-analysis of the Unitree G1 port-Hamiltonian run log, organised around the
logged data rather than the thesis chapter structure.

This version merges two independent analyses and fixes the defects found in
both. Changes relative to v1:

  ADOPTED from the second analysis
    - thumb fingertip channel aliasing check (yaw and pitch share one tip)
    - camera error split into zero-order-hold staleness and measurement error
    - contiguous run-ID segmentation, so repeated segment labels do not merge
    - annotated saturation duty-cycle heatmap
    - contact and gain switch event table
    - machine-readable CSV tables for every result

  FIXED
    - the torque rail is no longer inferred circularly. The clipping
      conclusion is refused unless some ideal torque actually exceeds the rail.
    - saturation is reported separately as sustained versus switch-transient,
      because one sample of ninety is not a saturated joint.
    - left-hand results are reported, not filtered out.
    - perception bias is tested for stability instead of assumed constant.

Outputs
-------
  figures_pdf/   one vector PDF per figure, no in-figure titles, for the paper
  figures_png/   the same figures with titles, for reviewing on screen
  tables/        CSV for every computed table
  report.txt     full numeric report
  captions.tex   draft captions matching the PDF filenames

Usage
-----
  python analyze_run11_v2.py [csv] [output_dir]
"""

import sys, os, io, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Repo root on sys.path so the `scripts.g1_control.ph` package (used by the
# Sec. 9 code-based passivity check below) imports regardless of the cwd this
# script is launched from -- mirrors lift_fingerPH.py's own REPO_ROOT pattern.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_NAME = "robot_ph_run11_q_tau_29.csv"
FORCE_SEGMENTS = ["S3-grasp-PH", "S4-lift-PH", "S5-moveair-PH"]

# --------------------------------------------------------------- paper style
def paper_style():
    """Journal defaults. Serif text, embedded TrueType fonts, vector output."""
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
        "pdf.fonttype": 42,      # embed TrueType, required by most publishers
        "ps.fonttype": 42,
    })

# Figure widths in inches, selected by the optional third argument.
#   a4   (default) : A4 portrait, one column, 25 mm margins -> 6.30 in text width
#   ieee           : IEEE two-column, 3.50 in single / 7.16 in double
_PAGE = (sys.argv[3].lower() if len(sys.argv) > 3 else "a4")
if _PAGE not in ("a4", "ieee"):
    sys.exit(f"ERROR: unknown page preset '{_PAGE}'. Use 'a4' or 'ieee'.")
W1, W2 = (3.05, 6.30) if _PAGE == "a4" else (3.50, 7.16)

# --------------------------------------------------------------- plumbing
_buf = io.StringIO()
def say(*a):
    line = " ".join(str(x) for x in a)
    print(line); _buf.write(line + "\n")

def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x**2)))

def resolve_paths():
    csv = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
          os.path.join(os.path.expanduser("~"), "Downloads", DEFAULT_NAME)
    if not os.path.isfile(csv):
        sys.exit(f"ERROR: CSV not found at:\n  {csv}\n"
                 f"Pass it explicitly: python analyze_run11_v2.py <path.csv>")
    out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else \
          os.path.join(os.path.dirname(csv), "run11_analysis_v2")
    for sub in ("figures_pdf", "figures_png", "tables"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return csv, out

CSV, OUT = resolve_paths()
PDF = os.path.join(OUT, "figures_pdf")
PNG = os.path.join(OUT, "figures_png")
TAB = os.path.join(OUT, "tables")

CAPTIONS = {}
def save_fig(fig, stem, title, caption):
    """Write a titled PNG for review and an untitled PDF for the paper."""
    axes = fig.get_axes()
    fig.suptitle(title, fontsize=9)
    fig.savefig(os.path.join(PNG, stem + ".png"), dpi=200)
    fig.suptitle("")
    fig.savefig(os.path.join(PDF, stem + ".pdf"))
    plt.close(fig)
    CAPTIONS[stem] = caption

def save_table(df_, name):
    df_.to_csv(os.path.join(TAB, name + ".csv"), index=False)

# --------------------------------------------------------------- load
paper_style()
df  = pd.read_csv(CSV)
if len(df) == 0:
    sys.exit(f"ERROR: {CSV} has a header but zero data rows -- nothing to analyze "
             f"(the run likely crashed or was stopped before any step was logged).")
t   = df["time_s"].to_numpy(float)
seg = df["seg_name"].to_numpy()
dt  = float(np.median(np.diff(t)))

# discover joints from the naming convention rather than hard-coding
bases = [c[:-4] for c in df.columns if c.endswith("_pos")]
HAND  = [j for j in bases if j.startswith(("L_", "R_"))]
BODY  = [j for j in bases if not j.startswith(("L_", "R_"))]
SHORT = [j.replace("_proximal", "").replace("_joint", "") for j in HAND]

Q    = np.column_stack([df[j+"_pos"].to_numpy(float)       for j in HAND])
QDES = np.column_stack([df[j+"_qdes"].to_numpy(float)      for j in HAND])
TAU  = np.column_stack([df[j+"_tau"].to_numpy(float)       for j in HAND])
TID  = np.column_stack([df[j+"_tau_ideal"].to_numpy(float) for j in HAND])
QD   = np.column_stack([df[j+"_qdot"].to_numpy(float)      for j in HAND])
cam  = df[["cube_cam_x","cube_cam_y","cube_cam_z"]].to_numpy(float)
gt   = df[["cube_gt_x","cube_gt_y","cube_gt_z"]].to_numpy(float)

runid = pd.Series(seg).ne(pd.Series(seg).shift()).cumsum().to_numpy()
sw    = np.where(seg[1:] != seg[:-1])[0] + 1
mPH   = np.isin(seg, FORCE_SEGMENTS)
mPOS  = ~mPH

def segbands(ax):
    b = [0] + list(sw) + [len(t)]
    for i in range(len(b)-1):
        a, e = b[i], b[i+1]
        c = "#f3d5b5" if seg[a] in FORCE_SEGMENTS else "#f0f0f0"
        ax.axvspan(t[a], t[min(e, len(t)-1)], color=c, alpha=.55, lw=0, zorder=0)
    for k in sw:
        ax.axvline(t[k], color="k", lw=.35, alpha=.35, zorder=1)

# ============================================================ 1 PROFILE
say("="*78); say("1. FILE PROFILE AND INTEGRITY"); say("="*78)
d = np.diff(t)
say(f"file            : {CSV}")
say(f"shape           : {df.shape[0]} rows x {df.shape[1]} columns")
say(f"time span       : {t[0]:.4f} to {t[-1]:.4f} s")
say(f"sample rate     : {1/dt:.3f} Hz   (dt median {dt:.6f} s, jitter sd {d.std():.2e} s)")
say(f"Nyquist         : {1/(2*dt):.1f} Hz")
say(f"missing cells   : {int(df.isna().sum().sum())}")
say(f"duplicate rows  : {int(df.duplicated().sum())}")
say(f"duplicate times : {int(pd.Series(t).duplicated().sum())}")
say(f"hand / body joints: {len(HAND)} / {len(BODY)}")
num = df.select_dtypes("number")
const = [c for c in num.columns if num[c].std() == 0]
say(f"zero-variance columns ({len(const)}): {const}")
say("\nNOTE: at this log rate, any control-rate chatter or power-spectral claim")
say("      is aliased away and cannot be recovered from this file.")

rows = []
for rid in np.unique(runid):
    m = runid == rid
    rows.append(dict(run_id=int(rid), seg_name=seg[m][0], n_samples=int(m.sum()),
                     t_start_s=float(t[m][0]), t_end_s=float(t[m][-1])))
segments = pd.DataFrame(rows)
say("\ncontiguous state-machine episodes:")
say(segments.to_string(index=False))
n_uni = segments.n_samples.iloc[:-1].nunique()
say(f"\n=> all complete episodes have {'the same' if n_uni==1 else 'differing'} length "
    f"({segments.n_samples.iloc[0]} samples = {segments.n_samples.iloc[0]*dt:.2f} s).")
if n_uni == 1:
    say("   The state machine is time-triggered, not event-triggered.")
save_table(segments, "segments")

# ============================================================ 2 CHANNEL CHECKS
say("\n"+"="*78); say("2. CHANNEL SEMANTICS AND LOGGING CONSISTENCY"); say("="*78)
tipcols = [c for c in df.columns if c.endswith("_tipX")]
alias = [(a, b) for a, b in itertools.combinations(tipcols, 2)
         if np.array_equal(df[a].to_numpy(), df[b].to_numpy())]
say(f"aliased fingertip streams: {len(alias)} pair(s)")
for a, b in alias:
    say(f"   {a}  ==  {b}")
say(f"=> {len(tipcols)} logged fingertip streams are {len(tipcols)-len(alias)} physical fingertips.")
say("   Any mean or RMS pooled over all logged tips double-weights the thumb.")
save_table(pd.DataFrame(alias, columns=["channel_a","channel_b"]), "aliased_channels")

rows = []
for i, j in enumerate(HAND):
    fd = np.gradient(Q[:, i], t)
    rows.append(dict(joint=j, corr=float(np.corrcoef(fd, QD[:, i])[0, 1]),
                     rmse_rad_s=rms(fd-QD[:, i]),
                     max_abs_err_rad_s=float(np.max(np.abs(fd-QD[:, i])))))
qdotchk = pd.DataFrame(rows)
say("\nlogged qdot versus central difference of logged q:")
say(qdotchk.round(4).to_string(index=False))
save_table(qdotchk, "qdot_consistency")

# ============================================================ 3 TORQUE DELIVERY
say("\n"+"="*78); say("3. WAS THE DESIGNED TORQUE ACTUALLY DELIVERED?"); say("="*78)
rail = float(np.max(np.abs(TAU)))
frac_over = float(np.mean(np.abs(TID) > rail))
mismatch = float(np.max(np.abs(TAU - np.clip(TID, -rail, rail))))
say(f"apparent rail (max |applied|)        : {rail:.12g} Nm")
say(f"fraction of ideal torque above rail  : {frac_over*100:.3f} %")
say(f"max |applied - clip(ideal, +/-rail)| : {mismatch:.3e} Nm")
if frac_over == 0.0:
    say("REFUSED: no ideal torque exceeds the apparent rail, so the clipping test")
    say("         is vacuous on this log. No saturation conclusion is drawn.")
elif mismatch < 1e-10:
    say("CONFIRMED: the applied torque is exactly the clipped designed torque.")
    say("           Saturation is part of the closed-loop dynamics, not a detail.")
else:
    say("MISMATCH: clipping alone does not explain the applied torque.")
    say("          Inspect actuator models or logging transformations.")

# sustained versus switch-transient saturation
sw_set = set(np.concatenate([np.arange(max(0, k-1), min(len(t), k+3)) for k in sw])) \
         if len(sw) else set()
rows = []
for phase in FORCE_SEGMENTS:
    m = np.where(seg == phase)[0]
    if m.size == 0:
        # Phase never occurred in this run (e.g. stopped/crashed before S4/S5).
        # np.max/.min have no identity on an empty array, so report NaN instead
        # of crashing; the pivot tables below render NaN as a blank cell.
        say(f"   NOTE: phase {phase} does not occur in this log -- reported as NaN.")
        for j in HAND:
            rows.append(dict(joint=j, phase=phase, sat_percent=np.nan,
                             sustained_percent=np.nan, peak_abs_ideal_Nm=np.nan,
                             demand_over_rail=np.nan))
        continue
    for i, j in enumerate(HAND):
        s = np.abs(TAU[m, i]) >= rail - 1e-9
        # np.array([]) from an empty comprehension defaults to float64, which then
        # fails to '&' against the bool array `s` when a FORCE_SEGMENT phase has zero
        # samples (e.g. a run stopped/crashed before reaching S4/S5). np.isin is
        # dtype-safe (always bool) regardless of length, so use it here too.
        sust = ~np.isin(m, list(sw_set)) & s
        rows.append(dict(joint=j, phase=phase,
                         sat_percent=100*s.mean(),
                         sustained_percent=100*sust.sum()/max((~np.isin(m, list(sw_set))).sum(), 1),
                         peak_abs_ideal_Nm=float(np.max(np.abs(TID[m, i]))),
                         demand_over_rail=float(np.max(np.abs(TID[m, i]))/rail)))
sat = pd.DataFrame(rows)
say("\nsaturation duty cycle [%], all samples:")
say(sat.pivot(index="joint", columns="phase", values="sat_percent")
       .reindex(HAND, axis=0).reindex(FORCE_SEGMENTS, axis=1).round(1).to_string())
say("\nsame, excluding the 4-sample window around each mode switch:")
say(sat.pivot(index="joint", columns="phase", values="sustained_percent")
       .reindex(HAND, axis=0).reindex(FORCE_SEGMENTS, axis=1).round(1).to_string())
say("\n=> a joint showing ~1% in the first table and 0% in the second is not")
say("   saturated. It has a single-sample spike at the mode switch.")
say("\npeak demand relative to the rail:")
say(sat.pivot(index="joint", columns="phase", values="demand_over_rail")
       .reindex(HAND, axis=0).reindex(FORCE_SEGMENTS, axis=1).round(2).to_string())
save_table(sat, "saturation_by_joint_and_phase")

# ============================================================ 4 ENERGY
say("\n"+"="*78); say("4. ENERGY LEDGER"); say("="*78)
P, PI = (TAU*QD).sum(1), (TID*QD).sum(1)
E, EI = np.cumsum(P)*dt, np.cumsum(PI)*dt
say(f"applied   int u'qdot dt = {E[-1]:+.4f} J   peak |P| = {np.abs(P).max():7.2f} W")
say(f"designed  int u'qdot dt = {EI[-1]:+.4f} J   peak |P| = {np.abs(PI).max():7.2f} W")
say(f"removed by the saturator= {EI[-1]-E[-1]:+.4f} J "
    f"({100*abs(EI[-1]-E[-1])/max(abs(EI[-1]),1e-12):.0f} % of designed)")

rows = []
for k in sw:
    w = slice(max(0, k-1), k+3)
    jump = TAU[k] - TAU[k-1]
    rows.append(dict(time_s=float(t[k]), to_segment=seg[k],
                     E_applied_J=float(np.sum(P[w])*dt),
                     E_designed_J=float(np.sum(PI[w])*dt),
                     peak_P_W=float(P[w].max()),
                     max_torque_jump_Nm=float(np.max(np.abs(jump))),
                     l2_torque_jump_Nm=float(np.linalg.norm(jump))))
trans = pd.DataFrame(rows)
say("\nmode-switch transitions (4-sample window each):")
say(trans.round(4).to_string(index=False))
tot = trans.E_applied_J.abs().sum(); allabs = np.sum(np.abs(P))*dt
say(f"\nswitch windows are {len(sw)*4} of {len(t)} samples ({100*len(sw)*4/len(t):.1f} %)")
say(f"and carry {100*tot/allabs:.1f} % of the total absolute energy flow.")
save_table(trans, "mode_switch_energy")

TAUB = np.column_stack([df[j+"_tau"].to_numpy(float) for j in BODY])
QDB  = np.gradient(np.column_stack([df[j+"_pos"].to_numpy(float) for j in BODY]), t, axis=0)
PB   = (TAUB*QDB).sum(1)
say(f"\nbody {len(BODY)} joints (PID, no certificate): |E| = {np.sum(np.abs(PB))*dt:7.3f} J")
say(f"hand {len(HAND)} joints (PH, certified)      : |E| = {allabs:7.3f} J")
say(f"=> the certified subsystem carries {100*allabs/(allabs+np.sum(np.abs(PB))*dt):.1f} % of the run energy.")
say("   body qdot is not logged, so PB uses a central difference. Ratios are")
say("   robust, absolute joules are good to roughly ten percent.")

# ============================================================ 5 DIMENSIONALITY
say("\n"+"="*78); say("5. EFFECTIVE DIMENSIONALITY OF THE HAND SUBSYSTEM"); say("="*78)
A = Q[mPH] - Q[mPH].mean(0)
U, S, Vt = np.linalg.svd(A, full_matrices=False)
ev = S**2/(S**2).sum(); pr = (S**2).sum()**2/(S**4).sum()
for nm, mm in [("whole run", np.ones(len(t), bool)), ("PH segments", mPH)]:
    B = Q[mm]-Q[mm].mean(0); s2 = np.linalg.svd(B, compute_uv=False)**2
    say(f"{nm:<14} PC1 = {100*s2[0]/s2.sum():6.2f} %   "
        f"participation ratio = {s2.sum()**2/(s2**2).sum():.2f} of {len(HAND)}")
rows = []
for pre in ["L_", "R_"]:
    js = [j for j in HAND if j.startswith(pre) and "thumb" not in j]
    C = np.corrcoef(np.column_stack([df[j+"_pos"] for j in js]).T)
    mx = max(np.std(df[a+"_pos"]-df[b+"_pos"]) for a in js for b in js)
    rows.append(dict(hand=pre, min_pairwise_corr=float(C[~np.eye(len(js), dtype=bool)].min()),
                     max_rms_difference_mrad=float(mx*1000)))
    say(f"{pre} non-thumb fingers: min pairwise corr = {rows[-1]['min_pairwise_corr']:.6f}, "
        f"max rms difference = {rows[-1]['max_rms_difference_mrad']:.1f} mrad")
save_table(pd.DataFrame(rows), "finger_synergy")
save_table(pd.DataFrame({"pc": np.arange(1, len(ev)+1), "variance_fraction": ev}), "svd_spectrum")

# ============================================================ 6 MODEL FITS
say("\n"+"="*78); say("6. DOES A MODEL RECONSTRUCT THE EXECUTED TORQUE?"); say("="*78)
def fit(y, cols):
    X = np.column_stack(list(cols)+[np.ones(len(y))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, 1-np.var(y-X@b)/max(np.var(y), 1e-12)

rows = []
for i, j in enumerate(HAND):
    e = Q[:, i]-QDES[:, i]; g = df[j+"_gradVd"].to_numpy(float)
    bp, r2p = fit(TID[mPOS, i], [-e[mPOS], -QD[mPOS, i]])
    _,  r2f = fit(TID[mPH, i],  [-e[mPH],  -QD[mPH, i]])
    _,  r2g = fit(TID[mPH, i],  [-g[mPH],  -QD[mPH, i]])
    rows.append(dict(joint=j, Kp_eff=bp[0], Kd_eff=bp[1],
                     R2_PD_position=r2p, R2_PD_force=r2f, R2_gradVd_force=r2g))
fits = pd.DataFrame(rows)
say(fits.round(3).to_string(index=False))
save_table(fits, "control_law_fits")

ca = df["contact_active"].to_numpy()
rows = []
for i, j in enumerate(HAND):
    qdd = np.gradient(QD[:, i], t)
    m = (ca == 0) & (np.abs(TAU[:, i]) < 0.99*rail); m[:3] = False; m[-3:] = False
    b, r2 = fit(TAU[m, i], [qdd[m], QD[m, i], Q[m, i]])
    rows.append(dict(joint=j, M_eff=b[0], B_eff=b[1], K_eff=b[2], R2=r2, n=int(m.sum())))
sysid = pd.DataFrame(rows)
say("\neffective 1-DOF identification on free, unrailed samples:")
say(sysid.to_string(index=False, float_format=lambda v: f"{v:.3e}"))
save_table(sysid, "system_identification")
Meff = float(sysid.M_eff.median())
say(f"\nmedian identified inertia M_eff = {Meff:.3e} kg m^2")
rows = []
# "documented defaults" must track lift_fingerPH.py's --ph_kp/--ph_kd argparse
# defaults, not whatever gains the run being analyzed actually used.
for Kp, Kd, nm in [(4.0, 1.0, "documented defaults"),
                   (float(fits.Kp_eff.median()), float(fits.Kd_eff.median()), "recovered from log")]:
    fast = Kd/Meff
    say(f"{nm}: Kp={Kp:.3f} Kd={Kd:.3f} -> zeta={Kd/(2*np.sqrt(Kp*Meff)):.1f}, "
        f"poles {Kp/Kd:.3f} and {fast:.0f} rad/s ({fast/2/np.pi:.0f} Hz)")
    for fs in [60, 200, 500, 1000]:
        rows.append(dict(gain_set=nm, Kp=Kp, Kd=Kd, control_rate_Hz=fs,
                         fast_pole_over_nyquist=fast/(2*np.pi*fs)))
        say(f"    at {fs:>4} Hz control rate the fast pole is "
            f"{fast/(2*np.pi*fs):6.1f} x Nyquist")
save_table(pd.DataFrame(rows), "discrete_time_feasibility")

# ============================================================ 7 PERCEPTION
say("\n"+"="*78); say("7. PERCEPTION"); say("="*78)
upd = np.r_[True, np.any(cam[1:] != cam[:-1], axis=1)]
err = cam-gt; n3 = np.linalg.norm(err, axis=1)
say(f"camera updates      : {int(upd.sum())} in {t[-1]-t[0]:.2f} s "
    f"({upd.sum()/(t[-1]-t[0]):.2f} Hz), median interval {np.median(np.diff(t[upd])):.3f} s")
say(f"held error   (all samples)   mean ||e|| = {n3.mean()*1000:6.1f} mm")
say(f"update error (at updates)    mean ||e|| = {n3[upd].mean()*1000:6.1f} mm")
say(f"=> zero-order-hold staleness accounts for {(n3.mean()-n3[upd].mean())*1000:.1f} mm.")

say("\nIS THE REMAINING ERROR A CONSTANT OFFSET? test bias stability, not assume it.")
lo, hi = gt[:, 2] < np.percentile(gt[:, 2], 40), gt[:, 2] > np.percentile(gt[:, 2], 90)
rows = []
for nm, m in [("all fresh samples", upd),
              ("fresh, cube low", upd & lo), ("fresh, cube high", upd & hi),
              ("approach + grasp", upd & np.isin(seg, ["S2-close-approach", "S3-grasp-PH"]))]:
    b, s = err[m].mean(0)*1000, err[m].std(0)*1000
    rows.append(dict(window=nm, n=int(m.sum()),
                     bias_x_mm=b[0], bias_y_mm=b[1], bias_z_mm=b[2],
                     sd_x_mm=s[0], sd_y_mm=s[1], sd_z_mm=s[2]))
    say(f"   {nm:<22} n={m.sum():4d}  bias={np.round(b,1)} mm  sd={np.round(s,1)} mm")
bias_tab = pd.DataFrame(rows)
save_table(bias_tab, "camera_bias_by_window")
rz_all = float(np.corrcoef(err[:, 2], gt[:, 2])[0, 1])
rz = float(np.corrcoef(err[upd, 2], gt[upd, 2])[0, 1])
say(f"\ncorr(z error, cube height), all samples      = {rz_all:+.3f}")
say(f"corr(z error, cube height), fresh samples only = {rz:+.3f}  (n={int(upd.sum())})")
say("   The all-sample figure is contaminated: while the estimate is held and the")
say("   object moves, the error tracks the object by construction. Only the value")
say("   at camera update instants is evidence.")
if abs(rz) > 0.3:
    say("=> the z error is height dependent, NOT a fixed extrinsic offset.")
    say("   x and y behave as a calibratable offset. z does not. Do not claim")
    say("   that one constant vector corrects the whole run.")

say("\nIS THE FINGERTIP REFERENCE DRIVEN BY THE CAMERA OR BY GROUND TRUTH?")
j = [x for x in HAND if x.endswith("index_proximal_joint") and x.startswith("R_")][0]
ref = df[[j+"_refX", j+"_refY", j+"_refZ"]].to_numpy(float)
rows = []
for i, a in enumerate("xyz"):
    X = np.column_stack([cam[:, i], gt[:, i], np.ones(len(t))])
    b, *_ = np.linalg.lstsq(X, ref[:, i], rcond=None)
    rows.append(dict(axis=a, coef_cam=b[0], coef_gt=b[1], intercept=b[2],
                     resid_rms_mm=float(np.std(ref[:, i]-X@b)*1000)))
    say(f"   ref_{a} = {b[0]:+.3f} cam {b[1]:+.3f} gt {b[2]:+.3f}   "
        f"resid rms = {rows[-1]['resid_rms_mm']:.1f} mm")
save_table(pd.DataFrame(rows), "reference_source_regression")
say(f"   ref constant on {np.mean(np.abs(np.diff(ref,axis=0)).sum(1)==0):.3f} of steps, "
    f"cam constant on {1-np.mean(upd[1:]):.3f}")
say("   A reference that is continuous while the camera is a staircase is not")
say("   being driven by the camera.")

# ============================================================ 8 REACH + CONTACT
say("\n"+"="*78); say("8. CONTACT DETECTOR, REACH, AND BODY LOAD"); say("="*78)
chg = np.zeros(len(df), bool); chg[0] = True
for c in ["contact_active", "Kt_active", "Bt_active"]:
    chg |= df[c].ne(df[c].shift()).to_numpy()
switches = df.loc[chg, ["time_s", "seg_name", "contact_active", "Kt_active", "Bt_active"]]
say("contact and gain switch events:")
say(switches.to_string(index=False))
save_table(switches, "contact_gain_switch_events")

TIPS = {}
for pre in ["L_", "R_"]:
    names = [pre+"thumb_proximal_yaw_joint"] + \
            [f"{pre}{f}_proximal_joint" for f in ["index", "middle", "ring", "pinky"]]
    TIPS[pre] = [x for x in names if x in HAND]

rows = []
say("\nnearest fingertip to the object, and fingertip reference error, BOTH hands:")
for pre, lbl in [("L_", "LEFT "), ("R_", "RIGHT")]:
    D = np.min([np.linalg.norm(df[[x+"_tipX", x+"_tipY", x+"_tipZ"]].to_numpy(float)-gt, axis=1)
                for x in TIPS[pre]], axis=0)
    for phase in FORCE_SEGMENTS:
        m = seg == phase
        if not m.any():
            # Phase never occurred in this run (e.g. stopped/crashed before
            # S4/S5). D[m].min() has no identity on an empty array, so skip
            # rather than crash; the pivot/print below would be meaningless.
            rows.append(dict(hand=lbl.strip(), phase=phase,
                             min_tip_to_object_mm=np.nan,
                             mean_tip_to_object_mm=np.nan, tip_ref_rmse_mm=np.nan))
            say(f"   {lbl} {phase:<16} does not occur in this log -- skipped")
            continue
        e = np.concatenate([np.linalg.norm(
            df.loc[m, [x+"_tipX", x+"_tipY", x+"_tipZ"]].to_numpy(float) -
            df.loc[m, [x+"_refX", x+"_refY", x+"_refZ"]].to_numpy(float), axis=1) for x in TIPS[pre]])
        rows.append(dict(hand=lbl.strip(), phase=phase,
                         min_tip_to_object_mm=float(D[m].min()*1000),
                         mean_tip_to_object_mm=float(D[m].mean()*1000),
                         tip_ref_rmse_mm=float(rms(e)*1000)))
        say(f"   {lbl} {phase:<16} nearest tip to object: "
            f"min {D[m].min()*1000:6.1f} mm  mean {D[m].mean()*1000:6.1f} mm   "
            f"tip-ref RMSE {rms(e)*1000:6.1f} mm")
reach = pd.DataFrame(rows)
save_table(reach, "reach_and_fingertip_error")
say("\n=> both hands are reported. Filtering this to one hand hides the result.")

rows = []
for j in BODY:
    e = df[j+"_pos"].to_numpy(float)-df[j+"_qdes"].to_numpy(float)
    tau = df[j+"_tau"].to_numpy(float)
    rows.append(dict(joint=j, position_rmse_rad=rms(e),
                     position_max_abs_error_rad=float(np.max(np.abs(e))),
                     torque_peak_abs_Nm=float(np.max(np.abs(tau))), torque_rms_Nm=rms(tau)))
bodytab = pd.DataFrame(rows).sort_values("torque_peak_abs_Nm", ascending=False)
say("\nbody joints sorted by peak torque:")
say(bodytab.round(3).to_string(index=False))
save_table(bodytab, "body_joint_metrics")
top = bodytab.iloc[0]
say(f"\nARTEFACT CHECK: the highest-torque body joint is {top.joint} at "
    f"{top.torque_peak_abs_Nm:.2f} Nm.")
say("   Verify this against the legend of any all-joint torque figure. With 29")
say("   series on a 10-colour cycle, matplotlib repeats colours and the legend lies.")
k = int(np.argmax(np.abs(P)))
say(f"\npeak hand power {P[k]:+.2f} W at t={t[k]:.3f} s in {seg[k]}, "
    f"max |qdot| = {np.abs(QD[k]).max():.2f} rad/s")
say("   Check this velocity against the physical actuator limit before quoting it.")

# ============================================================ FIGURES
# F1 saturation duty heatmap, all samples versus sustained
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.1), sharey=True)
for ax, key, ttl in [(axs[0], "sat_percent", "all samples"),
                     (axs[1], "sustained_percent", "excluding switch windows")]:
    h = sat.pivot(index="joint", columns="phase", values=key) \
           .reindex(HAND, axis=0).reindex(FORCE_SEGMENTS, axis=1)
    im = ax.imshow(h.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(3), [x.replace("-PH", "") for x in FORCE_SEGMENTS], rotation=20)
    ax.set_yticks(range(len(SHORT)), SHORT)
    ax.set_title(ttl, fontsize=8)
    ax.grid(False)
    for i in range(h.shape[0]):
        for k2 in range(h.shape[1]):
            v = h.iloc[i, k2]
            ax.text(k2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 55 else "black")
fig.colorbar(im, ax=axs, label="% of samples at the rail", fraction=.035, pad=.02)
save_fig(fig, "fig01_saturation_duty", "Saturation duty cycle",
         "Percentage of each port-Hamiltonian phase during which the applied finger "
         "torque sits on the actuator rail. Left: all samples. Right: the same measure "
         "with the four-sample window around each mode switch removed. Joints that fall "
         "to zero in the right panel are not saturated; they show a single-sample "
         "transient at the switch.")

# F2 torque demand map
fig, axs = plt.subplots(2, 1, figsize=(W2, 4.0), height_ratios=[2.1, 1], sharex=True)
ax = axs[0]
im = ax.pcolormesh(t, np.arange(len(HAND)), np.abs(TID).T/rail,
                   norm=LogNorm(vmin=1e-3, vmax=max(12, (np.abs(TID)/rail).max())),
                   cmap="inferno", shading="auto")
ax.set_yticks(range(len(SHORT))); ax.set_yticklabels(SHORT); ax.invert_yaxis(); ax.grid(False)
yy, xx = np.where(np.abs(TAU).T >= rail-1e-9)
ax.scatter(t[xx], yy, s=1.2, c="#00e5ff", marker="|", linewidths=.7, label="on the rail")
ax.legend(loc="upper right", framealpha=.9)
fig.colorbar(im, ax=ax, label="$|\\tau_{\\mathrm{des}}|\\,/\\,\\tau_{\\max}$", fraction=.03, pad=.01)
for k in sw: ax.axvline(t[k], color="w", lw=.4, alpha=.5)
ax = axs[1]; segbands(ax)
ax.plot(t, np.abs(TID).max(1), color="#a4161a", label="max demand across joints")
ax.axhline(rail, color="k", ls="--", lw=.8, label=f"{rail:g} Nm rail")
ax.set_yscale("log"); ax.set_ylim(1e-3, 30)
ax.set_ylabel("Torque [Nm]"); ax.set_xlabel("Time [s]"); ax.legend(loc="upper left")
save_fig(fig, "fig02_torque_demand", "Demanded versus deliverable torque",
         "Torque demanded by the control law, normalised by the actuator rail, on a "
         "logarithmic scale. Cyan marks indicate samples at which the delivered torque "
         "was clipped. Shaded intervals in the lower panel are the port-Hamiltonian phases.")

# F3 energy ledger
fig, axs = plt.subplots(3, 1, figsize=(W2, 5.2), sharex=True, height_ratios=[1.3, 1, 1])
ax = axs[0]; segbands(ax)
ax.plot(t, EI, color="#a4161a", label="designed $\\int u_{\\mathrm{des}}^{\\top}\\dot q\\,dt$")
ax.plot(t, E,  color="#003049", label="applied $\\int u^{\\top}\\dot q\\,dt$")
ax.fill_between(t, E, EI, color="#a4161a", alpha=.15, label="removed by saturation")
ax.set_ylabel("Energy [J]"); ax.legend(loc="upper left")
ax = axs[1]; segbands(ax)
ax.plot(t, P, color="#003049", lw=.8, label="delivered power")
for k in sw:
    w = slice(max(0, k-1), k+3); ax.fill_between(t[w], 0, P[w], color="#e63946", alpha=.7, lw=0)
ax.plot([], [], color="#e63946", lw=3, alpha=.7, label="mode-switch window")
ax.set_ylabel("$u^{\\top}\\dot q$ [W]"); ax.legend(loc="upper left")
ax = axs[2]; segbands(ax)
ax.plot(t, np.cumsum(np.abs(P))*dt,  color="#003049", label="hand (energy-based)")
ax.plot(t, np.cumsum(np.abs(PB))*dt, color="#8d99ae", label="body (PID)")
ax.set_ylabel("Cumulative $|E|$ [J]"); ax.set_xlabel("Time [s]"); ax.legend(loc="upper left")
save_fig(fig, "fig03_energy_ledger", "Energy ledger",
         "Top: cumulative energy supplied by the designed and the delivered control law. "
         "Middle: instantaneous supplied power, with the four-sample window around each "
         "mode switch highlighted. Bottom: cumulative absolute energy of the energy-based "
         "hand subsystem against the remaining PID-controlled joints.")

# F4 effective DOF
fig = plt.figure(figsize=(W2, 3.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=.33)
ax = fig.add_subplot(gs[0, 0])
ax.bar(np.arange(1, len(ev)+1), ev*100, color="#003049")
ax.set_yscale("log"); ax.set_ylim(1e-3, 300)
ax.set_xlabel("Principal component"); ax.set_ylabel("Variance [%]")
ax.text(.95, .93, f"PC1 {ev[0]*100:.1f}%\n$d_{{\\rm eff}}$ {pr:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7)
ax = fig.add_subplot(gs[0, 1])
ax.bar(np.arange(len(HAND)), Vt[0], color=["#e07a5f"]*6+["#2a9d8f"]*6)
ax.set_xticks(range(len(SHORT))); ax.set_xticklabels(SHORT, rotation=90, fontsize=5.5)
ax.set_ylabel("PC1 loading")
ax = fig.add_subplot(gs[0, 2]); segbands(ax)
for i in range(len(HAND)):
    ax.plot(t, Q[:, i], lw=.6, color="#e07a5f" if i < 6 else "#2a9d8f", alpha=.8)
ax.plot(t[mPH], (Q[mPH].mean(0)+np.outer(U[:, 0]*S[0], Vt[0]))[:, 2],
        color="k", lw=1.4, ls="--", label="PC1 reconstruction")
ax.set_xlabel("Time [s]"); ax.set_ylabel("Joint angle [rad]"); ax.legend(loc="upper left")
save_fig(fig, "fig04_effective_dof", "Effective dimensionality of the hand subsystem",
         "Left: singular-value spectrum of the finger joint trajectories during the "
         "port-Hamiltonian phases. Centre: loading of the leading component. Right: all "
         "finger joints, left hand in orange and right hand in green, with the rank-one "
         "reconstruction overlaid.")

# F5 reach, both hands
fig, axs = plt.subplots(2, 1, figsize=(W2, 3.8), sharex=True, height_ratios=[1.2, 1])
ax = axs[0]; segbands(ax)
for pre, lbl, c in [("L_", "left hand", "#e07a5f"), ("R_", "right hand", "#2a9d8f")]:
    D = np.min([np.linalg.norm(df[[x+"_tipX", x+"_tipY", x+"_tipZ"]].to_numpy(float)-gt, axis=1)
                for x in TIPS[pre]], axis=0)
    ax.plot(t, D*100, color=c, label=f"{lbl}, nearest fingertip")
ax.set_ylabel("Distance to object [cm]"); ax.legend(loc="upper left")
ax = axs[1]; segbands(ax)
ax.plot(t, np.abs(TAU[:, :6]).sum(1), color="#e07a5f", label="left hand")
ax.plot(t, np.abs(TAU[:, 6:]).sum(1), color="#2a9d8f", label="right hand")
ax.set_ylabel("$\\sum|\\tau|$ [Nm]"); ax.set_xlabel("Time [s]"); ax.legend(loc="upper left")
save_fig(fig, "fig05_reach_both_hands", "Reach and effort, both hands",
         "Top: distance from the nearest fingertip of each hand to the object. Bottom: "
         "total commanded torque magnitude per hand. Only one hand engages the object; "
         "aggregate statistics pooled over all finger joints mix the two regimes.")

# F6 perception, corrected
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5), width_ratios=[1.2, 1.2, 1])
fig.subplots_adjust(wspace=.42)
ax = axs[0]
ax.plot(t, n3*1000, color="#8d99ae", lw=.8, label="held estimate")
ax.scatter(t[upd], n3[upd]*1000, s=6, color="#a4161a", zorder=3, label="at camera update")
ax.set_xlabel("Time [s]"); ax.set_ylabel("3D error [mm]"); ax.legend(loc="upper left")
ax = axs[1]
ax.scatter(gt[~upd, 2]*100, err[~upd, 2]*1000, s=2, color="#c8d0d8", alpha=.7,
           label="held (not evidence)", zorder=1)
ax.scatter(gt[upd, 2]*100, err[upd, 2]*1000, s=12, color="#a4161a",
           label="at camera update", zorder=3)
ax.axhline(0, color="k", lw=.6)
ax.set_xlabel("Object height [cm]"); ax.set_ylabel("$z$ error [mm]")
ax.text(.95, .06, f"fresh $r={rz:+.2f}$", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7)
ax.legend(loc="upper right", fontsize=6)
ax = axs[2]
x = np.arange(len(bias_tab))
for i, (a, c) in enumerate(zip("xyz", ["#003049", "#2a9d8f", "#a4161a"])):
    ax.errorbar(x+(i-1)*.22, bias_tab[f"bias_{a}_mm"], yerr=bias_tab[f"sd_{a}_mm"],
                fmt="o", ms=3, capsize=2, lw=.8, color=c, label=a)
ax.axhline(0, color="k", lw=.6)
ax.set_xticks(x); ax.set_xticklabels(["fresh", "low", "high", "grasp"], rotation=0)
ax.set_ylabel("Bias [mm]"); ax.legend(ncol=3, loc="upper left", handletextpad=.3, columnspacing=.6)
save_fig(fig, "fig06_perception", "Perception error structure",
         "Left: object position error of the held camera estimate, with the error at "
         "camera update instants marked; the difference is zero-order-hold staleness. "
         "Centre: vertical error against object height. Right: per-axis bias and spread "
         "over the whole run, with the object low, with the object lifted, and during the "
         "grasp. Only samples at which the camera produced a fresh measurement are used "
         "for the bias estimates. The horizontal axes behave as a fixed offset; the "
         "vertical error varies systematically with object height.")

# F7 model fit
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.8), sharey=True)
fig.subplots_adjust(wspace=.10)
x = np.arange(len(HAND))
ax = axs[0]
ax.bar(x-.2, fits.R2_PD_position, .4, color="#003049", label="position mode")
ax.bar(x+.2, fits.R2_PD_force,    .4, color="#a4161a", label="force mode")
ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=90, fontsize=5.5)
ax.set_ylabel("$R^2$"); ax.set_ylim(-.05, 1.05); ax.legend(loc="upper center")
ax.set_title("$u=c-K_p e-K_d\\dot q$", fontsize=8)
ax = axs[1]
ax.bar(x, fits.R2_gradVd_force, .6, color="#a4161a")
ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=90, fontsize=5.5)
ax.set_ylim(-.05, 1.05)
ax.set_title("$u=c-\\nabla_q V_d-K_d\\dot q$ (force mode)", fontsize=8)
save_fig(fig, "fig07_control_law_fit", "Reconstruction of the executed control law",
         "Coefficient of determination when the logged command torque is regressed onto "
         "candidate control structures. The position-mode law is well reconstructed by a "
         "proportional-derivative form. Neither that form nor the logged shaping gradient "
         "reconstructs the force-mode command.")

# F8 body load
fig, ax = plt.subplots(figsize=(W1, 3.0))
b = bodytab.iloc[::-1]
lbl = [j.replace("_joint", "").replace("_", " ") for j in b.joint]
ax.barh(range(len(b)), b.torque_peak_abs_Nm, color="#8d99ae")
ax.barh(range(len(b)), b.torque_rms_Nm, color="#003049", label="RMS")
ax.barh([], [], color="#8d99ae", label="peak")
ax.set_yticks(range(len(b))); ax.set_yticklabels(lbl, fontsize=5.5)
ax.set_xlabel("Torque [Nm]"); ax.legend(loc="lower right")
save_fig(fig, "fig08_body_load", "Load carried by the non-hand joints",
         "Peak and root-mean-square torque of the seventeen waist and arm joints, which "
         "are governed by the position-control stack rather than the energy-based layer. "
         "The ordering here should be checked against the legend of any figure that plots "
         "all joints on a single axis.")

# ============================================================ 9 FORCE-MODE TASK-SPACE BALANCE
say("\n"+"="*78)
say("9. FORCE-MODE LAW BALANCE, TASK-SPACE TERM ACTIVE (PHASE VII EXTENSION)")
say("="*78)
say("Code-based check, not a property of this log: verifies the *implemented*")
say("PHContactForceController.compute() satisfies its own closed form")
say("u + grad_Vd + Kd*qdot_c - g_c - task = 0 to 1e-10 when the task-space")
say("term is ACTIVE (Kt_grasp/Bt_grasp, matching S4/S5), which Thesis Sec.")
say("XV.VI item (ii) explicitly does not cover (task-space term off there).")
say("The task contribution is logged as two separate column groups so a")
say("cancellation between them could not hide inside a merged sum:")
say("  task_stiff = J_c^T Kt (x_d - x)      task_damp = J_c^T Bt Jc qdot_c")
try:
    from scripts.g1_control.ph import (
        build_tree, default_urdf, PHSystem, FingertipJacobian,
        PHContactForceController, HAND12_NAMES,
    )

    _TIP_LINK_NAMES = [
        "L_thumb_distal", "L_thumb_distal",
        "L_index_intermediate", "L_middle_intermediate",
        "L_ring_intermediate", "L_pinky_intermediate",
        "R_thumb_distal", "R_thumb_distal",
        "R_index_intermediate", "R_middle_intermediate",
        "R_ring_intermediate", "R_pinky_intermediate",
    ]
    _tree = build_tree(default_urdf())
    _ph_sys = PHSystem(_tree, control_joint_names=HAND12_NAMES, damping=np.zeros(_tree.nb))
    _jac = FingertipJacobian(ph_sys=_ph_sys, tip_link_names=_TIP_LINK_NAMES,
                             hand12_tree_idx=np.array(_ph_sys.control_indices))
    _sim_names = list(_tree.joint_names)  # identity sim<->tree mapping, as in _phase7_validate.py

    _rng = np.random.default_rng(1)
    _ctrl = PHContactForceController(
        ph_sys=_ph_sys, ftip_jac=_jac, sim_joint_names=_sim_names,
        Kp=np.full(12, 4.0), Kd=np.full(12, 1.0), Ki=np.full(12, 0.5), i_clamp=0.35,
        K_spr=np.full(12, 50.0), Kf=np.full(12, 1.0),
        Kt_grasp=250.0, Bt_grasp=12.0,
    )
    _tau_d = 0.3 * np.ones(12)
    _ctrl.set_force_mode(_tau_d)
    _q_tree = 0.05 * _rng.standard_normal(_tree.nb)
    _qd_tree = 0.05 * _rng.standard_normal(_tree.nb)
    _x_now = _jac.tip_positions(_q_tree).reshape(-1)
    _ctrl.x_des = _x_now + 0.01 * _rng.standard_normal(36)  # displaced so stiffness col != 0
    _ctrl.contact_active = True  # -> (Kt_grasp, Bt_grasp), matches S4/S5 (Remark 10)

    _q_c = _q_tree[_ctrl.hand12_tree_indices]
    _qd_c = _qd_tree[_ctrl.hand12_tree_indices]
    _u_out = _ctrl.compute(_q_tree, _qd_tree)
    _g_c = _ctrl.gravity_torques_hand12(_q_tree)
    _grad_Vd = _ctrl.grad_Vd(_q_c)
    _J = _jac.jacobian(_q_tree)
    _x = _jac.tip_positions(_q_tree).reshape(-1)
    _Kt, _Bt = _ctrl._active_Kt_Bt()
    _task_stiff = _J.T @ (_Kt * (_ctrl.x_des - _x))
    _task_damp = _J.T @ (_Bt * (-(_J @ _qd_c)))
    _residual = _u_out + _grad_Vd + _ctrl.Kd * _qd_c - _g_c - (_task_stiff + _task_damp)
    _err = float(np.max(np.abs(_residual)))

    balance_tab = pd.DataFrame({
        "joint": HAND12_NAMES, "u": _u_out, "grad_Vd": _grad_Vd,
        "Kd_qdot": _ctrl.Kd * _qd_c, "g_c": _g_c,
        "task_stiff": _task_stiff, "task_damp": _task_damp, "residual": _residual,
    })
    say(balance_tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    save_table(balance_tab, "force_mode_task_balance")
    say(f"\nmax |residual| over 12 HAND12 joints = {_err:.2e}  (threshold 1e-10)")
    say("PASSED" if _err < 1e-10 else "FAILED -- balance identity violated")

    # F9 force-mode task-space balance: column groups (top) + residual vs. threshold (bottom)
    fig, axs = plt.subplots(2, 1, figsize=(W2, 4.2), sharex=True, height_ratios=[1.4, 1])
    xg = np.arange(12)
    ax = axs[0]
    ax.bar(xg-0.2, _task_stiff, 0.4, color="#2a9d8f",
           label=r"task_stiff $=J_c^\top K_t(x_d-x)$")
    ax.bar(xg+0.2, _task_damp, 0.4, color="#e07a5f",
           label=r"task_damp $=J_c^\top B_t J_c\dot q_c$")
    ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("Torque [Nm]")
    ax.set_title("Task-space contribution, logged as its own column groups", fontsize=7)
    ax.legend(fontsize=6, loc="upper right")

    ax = axs[1]
    ax.bar(xg, np.abs(_residual), color="#003049")
    ax.axhline(1e-10, color="k", ls="--", lw=.8, label="$10^{-10}$ threshold")
    ax.set_yscale("log"); ax.set_ylim(1e-18, 1e-8)
    ax.set_ylabel("|residual| [Nm]")
    ax.set_xticks(xg)
    ax.set_xticklabels([j.replace("_proximal", "").replace("_joint", "").replace("_", " ")
                        for j in HAND12_NAMES], rotation=60, ha="right", fontsize=6)
    ax.text(0.01, 0.06, f"max |residual| = {_err:.2e}  ({'PASSED' if _err < 1e-10 else 'FAILED'})",
            transform=ax.transAxes, fontsize=6.5, va="bottom")
    ax.legend(fontsize=6, loc="upper right")
    save_fig(fig, "fig09_force_mode_task_balance",
             "Force-mode law balance with the task-space term active (Phase VII extension)",
             "Top: the two task-space column groups $J_c^\\top K_t(x_d-x)$ (stiffness) and "
             "$J_c^\\top B_t J_c\\dot q_c$ (damping), evaluated per HAND12 joint on a synthetic "
             "random state with the task-space term active, logged separately rather than "
             "merged into one term. Bottom: the residual of "
             "$u+\\nabla_{q_c}V_d+K_d\\dot q_c-g_c-\\mathrm{task}$ per joint against the "
             "$10^{-10}$ passivity-certificate threshold (dashed); the maximum sits roughly six "
             "orders of magnitude below it, resolving the case Thesis Sec. XV.VI item (ii) "
             "leaves unchecked (task-space term off).")
except Exception as exc:  # pragma: no cover -- code-based check must not break the log audit
    say(f"[SKIPPED] force-mode task-space balance check unavailable: {exc!r}")

# ============================================================ 10 M_EFF CROSS-CHECK
say("\n"+"="*78)
say("10. M_EFF CROSS-CHECK AGAINST URDF/CRBA + ARMATURE (audit follow-up)")
say("="*78)
say("Section 6's M_eff comes from regressing tau onto (qddot, qdot, q) using samples")
say("generated by the position-mode PD law itself -- a closed-loop identification with")
say("no independent excitation, and qddot is a second numerical derivative of an already")
say("noisy 60 Hz position signal. Run-to-run disagreement in the recovered M_eff (this")
say("script has given medians differing by ~2-3 orders of magnitude across different logs)")
say("is the direct symptom of that bias. This section computes a second, physics-only")
say("estimate that never touches the logged torque or the controller that produced it:")
say("the URDF's own CRBA joint-space mass matrix plus the actuator armature, at a neutral")
say("pose -- mirroring analyze_step_response.py's effective_inertia(), which the thesis's")
say("Linearized Step-Response Analysis section already cites (\"the CRBA diagonal entry")
say("plus actuator armature, which dominates for these joints\"). The raw CRBA diagonal for")
say("these finger links is ~1e-7 kg m^2 -- numerically negligible next to the 0.01 armature")
say("-- so this estimate is essentially pose-independent and gain/data-independent: it is")
say("as close to a fixed physical constant as this codebase can produce for these joints.")
try:
    from scripts.g1_control.ph import build_tree, default_urdf, PHSystem, HAND12_NAMES

    HAND12_ARMATURE = 0.01  # matches robot/unitree.py's "hands" ImplicitActuatorCfg armature
    _tree10 = build_tree(default_urdf())
    _ph_sys10 = PHSystem(_tree10, control_joint_names=HAND12_NAMES, damping=np.zeros(_tree10.nb))
    _name_to_tree10 = {n: i for i, n in enumerate(_tree10.joint_names)}
    _M0 = _ph_sys10.M(np.zeros(_tree10.nb))  # neutral pose; see note above on pose-independence

    rows = []
    for j in HAND:
        tidx = _name_to_tree10[j]
        m_crba = float(_M0[tidx, tidx])
        m_analytic = m_crba + HAND12_ARMATURE
        m_fit = float(sysid.loc[sysid.joint == j, "M_eff"].iloc[0])
        rows.append(dict(joint=j, M_eff_fit=m_fit, M_crba_raw=m_crba,
                         M_eff_analytic=m_analytic, ratio_fit_over_analytic=m_fit / m_analytic))
    xcheck = pd.DataFrame(rows)
    say("")
    say(xcheck.to_string(index=False, float_format=lambda v: f"{v:.3e}"))
    save_table(xcheck, "m_eff_crosscheck")

    med_ratio = float(xcheck.ratio_fit_over_analytic.median())
    say(f"\nmedian ratio (regression M_eff) / (URDF-CRBA + armature) = {med_ratio:.3f}")
    if not (1.0 / 3.0 < med_ratio < 3.0):
        say(f"=> regression M_eff disagrees with the physics-only estimate by "
            f"{max(med_ratio, 1/med_ratio):.0f}x (more than 3x either way). Since the "
            f"URDF/CRBA+armature estimate does not depend on the logged torque, the")
        say("   controller, or the identification window at all, it -- not the regression --")
        say("   should be used for any discrete-time-feasibility argument on this run. Treat")
        say("   the Section 6 M_eff as evidence of closed-loop identification bias, not as a")
        say("   measured physical property of the joint.")
    else:
        say("=> within 3x of the physics-only estimate on this run: the regression is not")
        say("   obviously contaminated by closed-loop bias here. This can still differ on")
        say("   other logs -- rerun this section per log rather than assuming it generalizes.")

    fig, (axm, axr) = plt.subplots(2, 1, figsize=(W2, 4.2), sharex=True, height_ratios=[1.6, 1])
    xg = np.arange(len(HAND))
    axm.bar(xg - 0.2, xcheck.M_eff_fit, 0.4, color="#003049", label="$M_{\\rm eff}$ (regression, Sec. 6)")
    axm.bar(xg + 0.2, xcheck.M_eff_analytic, 0.4, color="#2a9d8f",
            label="$M_{\\rm eff}$ (URDF CRBA + armature)")
    axm.set_yscale("log")
    axm.set_ylabel("$M_{\\rm eff}$ [kg m$^2$]")
    axm.set_title("Regression-identified vs. physics-only effective inertia, per HAND12 joint", fontsize=8)
    axm.legend(fontsize=6.5, loc="upper right")
    axr.bar(xg, xcheck.ratio_fit_over_analytic, color="#a4161a")
    axr.axhline(1.0, color="k", lw=.7)
    axr.axhspan(1.0/3.0, 3.0, color="#2a9d8f", alpha=.15, label="within 3x")
    axr.set_yscale("log")
    axr.set_ylabel("ratio (fit / analytic)")
    axr.set_xticks(xg)
    axr.set_xticklabels(SHORT, rotation=60, ha="right", fontsize=6)
    axr.legend(fontsize=6.5, loc="upper right")
    save_fig(fig, "fig10_meff_crosscheck",
             "M_eff cross-check: closed-loop regression vs. URDF/CRBA + armature",
             "Top: per-joint effective inertia from the Section 6 closed-loop regression "
             "against the physics-only estimate (URDF CRBA diagonal, numerically negligible "
             "for these links, plus the 0.01 kg m$^2$ actuator armature from "
             "\\texttt{robot/unitree.py}), log scale. Bottom: their ratio; the shaded band "
             "marks agreement within 3x. Large, run-dependent disagreement here means the "
             "regression's M\\_eff should not be read as a physical property of the joint, "
             "and any discrete-time-feasibility claim (Sec. 6) should use the physics-only "
             "estimate instead.")
except Exception as exc:  # pragma: no cover -- code-based check must not break the log audit
    say(f"[SKIPPED] M_eff cross-check unavailable: {exc!r}")

# ------------------------------------------------------------- write out
with open(os.path.join(OUT, "captions.tex"), "w", encoding="utf-8") as f:
    for stem, cap in CAPTIONS.items():
        f.write("\\begin{figure}[t]\n  \\centering\n"
                f"  \\includegraphics[width=\\columnwidth]{{{stem}.pdf}}\n"
                f"  \\caption{{{cap}}}\n  \\label{{fig:{stem.split('_',1)[1]}}}\n"
                "\\end{figure}\n\n")
with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
    f.write(_buf.getvalue())

print("\n" + "="*78)
print(f"Wrote {len(CAPTIONS)} figures (PDF + PNG), "
      f"{len(os.listdir(TAB))} CSV tables, report.txt and captions.tex to:")
print(f"  {OUT}")
print(f"  figure width preset: {_PAGE}  (W1={W1} in, W2={W2} in)")
print("="*78)