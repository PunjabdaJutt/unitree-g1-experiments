# Real_logs/ -- architecture of the control loops behind each CSV

Context for the analysis pass on `Real_logs/`. Answers one question per file: *which
script produced it, under which control law, and what does each column mean.* Verified
against the actual producing scripts (not just filenames) and against the files' own
headers/segment names on 2026-09-19 -- see "How each file was identified" at the bottom.

**Files were renamed on 2026-09-19** to match the naming convention each producing script
itself uses (`arm_ph_<mode>_<run_ts>.csv` for `g1_hw_arm_ph_control.py`,
`arm_positions_<run_ts>.csv`/`hand_state_<run_ts>.csv` for `grasp_ph.py` -- the same
convention already used, untouched, by `Successfull_grasps/`). Original names, for
anyone cross-referencing older notes: `Porthamiltnian_control.csv` ->
`arm_ph_ph_20260918_183908.csv`, `Port_Hamiltonian_control2.csv` ->
`arm_ph_position_20260918_184048.csv`, `positional_control_port_hamiltonian.csv` ->
`arm_positions_20260918_185109.csv`, `port_hamiltoinain_finger.csv` ->
`hand_state_20260918_185109.csv`.

**Correction to the original assumption**: only 2 of the 4 top-level CSVs here came from
`scripts/g1_control/hardware/g1_hw_arm_ph_control.py`. The other 2 came from
`grasp_ph.py` (which drives the arm with plain position-PD, and the hand with
`g1_hw_ph_finger_control.py`'s PH-admittance law) -- these are two different control
architectures, on two different subsystems (arm vs. hand), and none of the numbers in one
should be read as if it came from the other. `Successfull_grasps/arm_positions_*.csv` +
`hand_state_*.csv` pairs are further `grasp_ph.py` runs, same schema as the two
`grasp_ph.py` files below.

| File | Producing script | Subsystem | Control law | Segments (in order) |
|---|---|---|---|---|
| `arm_ph_ph_20260918_183908.csv` | `g1_hw_arm_ph_control.py --mode ph` | right arm (7 DOF) | PH gravity-comp + impedance TORQUE | hold_in, move_to_pose, hold_torque, return, release |
| `arm_ph_position_20260918_184048.csv` | `g1_hw_arm_ph_control.py --mode position` | right arm (7 DOF) | plain position-PD (baseline) | hold_in, move_to_pose, hold_torque, return, release |
| `arm_positions_20260918_185109.csv` | `grasp_ph.py` (arm log) | full 29-DOF body (arm commanded, legs/waist just observed) | plain position-PD (arm_sdk), IK-solved Cartesian targets | hold, side_up, approach_out, approach, descend, grasp, grasp_hold, lift, lift_hold, lower, release, retreat, retreat_out, side_down |
| `hand_state_20260918_185109.csv` | `grasp_ph.py` (hand log, via `g1_hw_ph_finger_control.HandAdmittancePH`) | 12 finger DOF (both hands) | PH torque law wrapped in a 2nd-order virtual admittance (position output) | same 14 segments as above |

`arm_positions_20260918_185109.csv` and `hand_state_20260918_185109.csv` are companion
logs from the **same** `grasp_ph.py` run: identical `run_ts` (both file-open timestamps
land in the same second, `18:51:09`), same 14 segments, same `STATE_LOG_HZ=10`, 1196 vs.
1195 rows, ~36.7 s. `arm_ph_ph_20260918_183908.csv` and
`arm_ph_position_20260918_184048.csv` are the direct baseline-vs-PH comparison pair
`g1_hw_arm_ph_control.py`'s own docstring describes: same choreography (17.08 s: 2+4+5+4+2 s
segments), one run per `--mode`, run back-to-back about 1.5 minutes apart.
`arm_ph_ph_20260918_183908.csv` was cross-checked byte-for-byte identical against
`scripts/g1_control/hardware/grasp_logs/arm_ph_ph_20260918_183908.csv`, the script's own
original log file for that run -- confirming both the run_ts and that this copy is
unmodified.

---

## 1. Right-arm bench test -- `g1_hw_arm_ph_control.py`

**File:** `scripts/g1_control/hardware/g1_hw_arm_ph_control.py`
**Produces:** `arm_ph_ph_20260918_183908.csv` (`--mode ph`),
`arm_ph_position_20260918_184048.csv` (`--mode position`).

### Purpose

Answers a narrow, specific question the rest of this repo's hardware scripts never
tested: *does the G1 arm's own `rt/arm_sdk` firmware accept a real nonzero torque
(`tau`), and if so, does a PH-style gravity-comp + impedance law track as well as the
plain position-PD baseline every other script here uses for the arm?* The finger
controller (`g1_hw_ph_finger_control.py`, section 2 below) never answers this because the
real Inspire hand firmware has no torque input at all -- its "PH control" is necessarily
an admittance wrapper. The arm's DDS wire format (`MotorCmd_`: `mode/q/dq/tau/kp/kd` per
joint) supports torque in principle; this script is the first to actually set `tau != 0`.

### Choreography (both modes; only `hold_torque`'s law switches)

```
hold_in (2s, position, weight 0->1)
  -> move_to_pose (4s, position, HOME -> DEFAULT_TEST_POSE)
  -> hold_torque (5s, --mode-controlled: position OR ph)
  -> return (4s, position, DEFAULT_TEST_POSE -> HOME)
  -> release (2s, position, weight 1->0)
```

HOME is whatever `Start()` reads as the arm's actual current position (not hardcoded).
`DEFAULT_TEST_POSE` is `grasp_absolute.py`'s `TRAJ_POSE_2` right-arm entries -- a pose
already exercised on this hardware under position-PD, so a `--mode ph` run's
tracking/settling has a known-reachable target to be judged against, not an arbitrary
offset from HOME.

### Control law, `hold_torque` segment only

**`--mode position` (baseline):** `rt/arm_sdk` position-PD, `tau=0`,
`kp=arm_kp` (default 60), `kd=arm_kd` (default 1.5) -- identical to the arm law used by
`g1_hw_ph_finger_control.py` and `grasp_ph.py` (section 2).

**`--mode ph` (the actual test):**

```
tau7 = g(q) - Kp_ph * (q - q_des) - Kd_ph * qdot
tau7 = clip(tau7, -max_tau, +max_tau)   # max_tau = 40% of URDF <limit effort=...> per joint
```

sent as `mc.tau` on `rt/arm_sdk`, **plus** a deliberately low-authority PD floor
(`safety_kp=3.0`, `safety_kd=0.3` -- far below the baseline's 60/1.5) sent on the *same*
`MotorCmd_`, so the firmware's own position servo -- not just software -- catches a wrong
torque law.

- `g(q)` is Pinocchio's generalized gravity vector for the 7 right-arm joints, computed
  with the waist at its **live measured** angle (gravity genuinely depends on it), via
  `pin_kinematics_client.gravity_right_arm_remote()` talking to a separate
  `pin_kinematics_server.py` process (conda `pinocchio` env -- the arm-control process
  itself runs in `.venv` and never imports Pinocchio directly).
- `Kp_ph`/`Kd_ph` (`DEFAULT_PH_KP`/`DEFAULT_PH_KD`, per-joint, RIGHT_ARM_JOINT_NAMES
  order: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw) are heavier on
  shoulder/elbow (bigger links, more gravity torque to correct), lighter on the wrist.
  This is the arm-scale analogue of `PHContactForceController`'s position-mode law
  (`u = g_c(q) - Kp*e - Kd*qdot`) minus the fingertip-Jacobian/HAND12 machinery, which
  doesn't apply to a 7-DOF rigid chain.
- Non-right-arm joints (waist, left arm) always stay plain position-PD, untouched by the
  torque law, in both modes.
- **Runaway watchdog:** if `|qdot|` on any right-arm joint exceeds 3.0 rad/s, or tracking
  error exceeds 0.6 rad, the script aborts into a graceful winddown immediately instead of
  continuing the segment.

### CSV schema (81 columns)

`timestamp, t_s, segment, mode` + all 29 `G1_JOINT_NAMES` body-pose columns (measured,
legs included though never commanded -- logged only to confirm they stayed put) +
`{name}_qdes_rad` for each of the 17 `ARM_JOINT_NAMES` (the actual ratio-interpolated
target this tick commanded, not the segment's final goal) + `{name}_qerr_rad` (measured
minus qdes) for the same 17, + per right-arm joint (7): `{name}_tau_cmd_Nm` (the clipped
torque actually sent; identically 0.0 for every row in a `--mode position` run) and
`{name}_g_Nm` (the gravity-comp term `g(q)` alone, logged even in `--mode position` rows
via `last_g7`'s stale/zero value -- check `mode` column before trusting it).

---

## 2. Hand PH-admittance control -- `g1_hw_ph_finger_control.py`

**File:** `scripts/g1_control/hardware/g1_hw_ph_finger_control.py`
**Used by:** run standalone (own `hand_state_phdemo_*.csv`, 4 cols/joint -- not one of the
files analyzed here), or imported as `phctrl` by `grasp_ph.py`, which is what actually
produced `hand_state_20260918_185109.csv`.

### Why this is an admittance wrapper, not direct torque control

Same `ph/` PH code as simulation (`PHContactForceController`, unmodified), run every real
control tick using the Inspire hand's own position feedback as `q`/`qdot` -- but the real
Inspire hand firmware has **no torque input** (confirmed against
`inspire_hand_modbus_bridge.py`; hand I/O here is direct Modbus TCP, not DDS, because
`rt/inspire/state` does not actually publish on this robot's bus). So the PH torque
(`tau_ideal`) can only be realized by converting it into a *position* setpoint via a
second-order virtual admittance, which the firmware's own internal position servo then
tracks:

```
tau_ideal   = PHContactForceController.compute(q, qdot, dt)     # PH law, unmodified from sim
tau_contact = -contact_gain * max(|F_ACT| - force_guard_frac * force_limit, 0)   # software guard
M_a * ddq_cmd + B_a * dq_cmd + K_track * (q_cmd - q_act) = tau_ideal + tau_contact
q_cmd      += dq_cmd * dt        # rate- and range-clamped, this is what's sent to the hand
```

`K_track` pulls the *reference* `q_cmd` back toward the hand's real measured `q_act`, so
it can't drift away from reality if the internal servo lags. `tau_contact` is a software
guard, independent of the firmware's own `FORCE_SET` stall ceiling, using real
`FORCE_ACT` (grams-force, Modbus register, converted to Newtons at this boundary) -- it
starts resisting *before* that ceiling and hard-freezes `q_cmd := q_act` once
`|FORCE_ACT|` reaches the limit, preventing wind-up. Uses `abs(FORCE_ACT)` because the
sign convention for "closing" direction was never confirmed on hardware.

None of `admit_mass`/`admit_damping`/`admit_ktrack`/`force_guard_frac` (nor `GRASP_TAU_D`)
are bench-tuned -- treat every gain here as a placeholder, not a validated constant.

### PH law itself (`PHContactForceController.compute`, `ph/ph_impedance.py`)

Two modes, switched per-segment:

- **Position mode:** `u_ph = g_c(q) - Kp*(q - q_des) - Kd*qdot - Ki*z` (`z` = clamped
  integral of position error, reset to 0 every mode switch).
- **Force mode:** `u_ph = g_c(q) - Kf*(K_spr*(q_c - q0_spr) - tau_d) - Kd*qdot` -- a
  virtual joint spring pulling toward a target grip torque `tau_d`
  (`GRASP_TAU_D` = 0 except right hand's 6 grip DOF), not a fixed closed position, so
  final closure genuinely depends on what's actually being grasped.

`g_c(q)` is the gravity torque on the 12 HAND12 finger DOF, from a Featherstone tree built
off the same URDF (central finite-difference on potential energy `U(q)`, not an
analytical Jacobian). Both terms are per-joint, HAND12_NAMES order (thumb_yaw, thumb_pitch,
index, middle, ring, pinky -- L block then R block).

### Segments used by `grasp_ph.py` (the ones in `hand_state_20260918_185109.csv`)

`hold (2s, hold) -> side_up -> approach_out -> approach -> descend (all position-mode,
hand held open) -> grasp (arm moves in) -> grasp_hold (1.5s, FORCE mode, hand_target =
GRASP_TAU_D -- the actual PH grip) -> lift -> lift_hold (hold) -> lower -> release
(position-mode, open) -> retreat -> retreat_out -> side_down`. Only `grasp_hold` runs the
force-mode spring law; every other segment is either `hold` (admittance coasts, no active
target) or `position` (impedance-regulated tracking of a fixed open/closed pose).

### CSV schema (`hand_state_20260918_185109.csv`, `grasp_ph.py`'s hand-log writer)

`timestamp, t_s, segment` + 7 columns per HAND12 joint (12 joints x 7 = 84 data columns),
in this order per joint: `pos_rad` (measured, from Modbus `ANGLE_ACT`), `force_g`
(measured `FORCE_ACT`, signed grams-force -- **not yet Newtons**, unlike the internal
admittance math above), `open_rad`/`close_rad` (the joint's fixed URDF limits, constant
every row -- not measurements), `qdes_rad` (`hand_ctrl.q_cmd_hand12`, the admittance
*reference* actually sent to the hand this tick -- not the segment's raw target, since the
admittance ramps toward it with its own 2nd-order dynamics), `qerr_rad` (`pos_rad -
qdes_rad`), `tau_des_Nm` (`PHContactForceController._tau_d`, the force-mode spring
target -- only meaningful during `grasp_hold`; identically 0 in every other segment, same
convention `GRASP_TAU_D` itself uses).

**Reading note:** `qdes_rad`/`qerr_rad` describe tracking of the *admittance's own*
reference, not the segment's nominal target -- a joint can show near-zero `qerr_rad` while
still being far from where the segment "asked" for, because the admittance hasn't caught
up yet. `tau_des_Nm` is only nonzero and only meaningful in `grasp_hold` rows.

---

## 3. Companion arm log for the same `grasp_ph.py` run -- `arm_positions_20260918_185109.csv`

Same script/run as section 2, but this is `grasp_ph.py`'s own **arm** log (plain
position-PD, IK-solved Cartesian targets for each segment's `arm_targets`/`arm_pR_target`,
right-arm-only reach -- no PH/torque law is ever applied to the arm in `grasp_ph.py`,
unlike section 1). 66 columns: `timestamp, t_s, segment` + all 29 `G1_JOINT_NAMES` +
`{name}_qdes_rad` and `{name}_qerr_rad` for the 17 `ARM_JOINT_NAMES` (no `mode` column,
since there is only ever one arm law here -- contrast with section 1's files, which do
have a `mode` column because that script's whole point is comparing two arm laws).

---

## 4. Formulas and exact call chain per log (added 2026-09-30)

Paths are relative to `scripts/g1_control/`. Line numbers are as of 2026-09-30 and will
drift if the scripts are edited -- search for the function name if a line is off.

### Which logs are inside the Port-Hamiltonian (PH) system

| Log | PH law active? | Where |
|---|---|---|
| `arm_ph_ph_*.csv` | **Yes** (arm torque) | `hold_torque` segment only (5 s); the other 4 segments are position-PD |
| `arm_ph_position_*.csv` | **No** -- baseline | `tau = 0` on every row; same choreography as above |
| `arm_positions_*.csv` | **No** | position-PD from IK targets in every segment; no PH law ever touches the arm in `grasp_ph.py` |
| `hand_state_*.csv` | **Yes** (finger torque, via admittance) | position-mode PH in the open-hand segments, force-mode PH in `grasp_hold`; `hold` segments coast |

### 4.1 `arm_ph_ph_*.csv` and `arm_ph_position_*.csv`

Producer: `hardware/g1_hw_arm_ph_control.py` (`--mode ph` / `--mode position`).
Call chain: `main()` (L451) -> `Custom` (L246) -> per-tick torque block (L411-426) ->
`pin_kinematics_client.gravity_right_arm_remote()` (L411) for `g(q)`. Segments come from
`build_segments()` (L230). CSV is written by `csv.writer` in `main()` (L533; the per-joint
`tau_cmd`/`g` columns at L567).

PH mode (`hold_torque` only), L411-413:

```
tau7 = g(q) - Kp_ph * (q - q_des) - Kd_ph * qdot
tau7 = clip(tau7, -max_tau, +max_tau)        # max_tau = tau_fraction * URDF effort limit (L264)
```

sent as `mc.tau` (L423) with a weak floor `mc.kp = safety_kp = 3.0`, `mc.kd = safety_kd = 0.3`
(L426). Position mode: `mc.tau = 0`, `mc.kp = arm_kp = 60`, `mc.kd = arm_kd = 1.5`.
Waist and left arm stay position-PD in both modes.

Not a full PH system: there is no storage function or energy shaping here, just gravity
compensation plus a PD term (the arm-scale analogue of the finger controller's position mode).

### 4.2 `arm_positions_*.csv`

Producer: `hardware/grasp_ph.py`, arm log (`csv.writer` at L1043; header at L1048-1051).
Arm targets come from `build_segments()` (L724) using IK waypoints from
`solve_waypoint_ik()` (L245). The arm is commanded by `g1_hw_ph_finger_control.Custom._tick`
(L759; arm write at L864-868):

```
mc.tau = 0,  mc.q = q_arm_cmd,  mc.kp = arm_kp (60),  mc.kd = arm_kd (1.5)
```

i.e. the firmware runs `tau_fw = kp*(q_cmd - q) + kd*(0 - qdot)`. No PH law.

### 4.3 `hand_state_*.csv`

Producer: `hardware/grasp_ph.py`, hand log (`csv.writer` at L1053; per-joint columns
L1055-1069, `qdes_rad` = `custom.hand_ctrl.q_cmd_hand12` at L1113). Call chain:

```
grasp_ph.main()  -> phctrl.Custom(...)  (L969)              # phctrl = g1_hw_ph_finger_control (L89)
Custom._tick()   -> self.hand_ctrl.step(...)  (L884)         # g1_hw_ph_finger_control.py
HandAdmittancePH.step() (L428) -> self.ph_ctrl.compute(...) (L460)
PHContactForceController.compute()   ph/ph_impedance.py L203
```

**(a) PH torque law** -- `PHContactForceController.compute`, `ph/ph_impedance.py` L221-239,
12 finger DOF:

```
position mode:  u_ph = g_c(q) - Kp * (q_c - q_des) - Kd * qdot_c - Ki * z          (L225)
                z   = clip(z + e*dt, -i_clamp, +i_clamp)                            (L224)
force mode:     u_ph = g_c(q) - grad_Vd(q_c) - Kd * qdot_c                          (L227)
                grad_Vd(q_c) = Kf * (K_spr * (q_c - q0_spr) - tau_d)                (L169-171)
                task = J^T [ Kt (x_des - x) - Bt (J qdot_c) ]   only if x_des set   (L237)
return u_ph + task
```

`g_c(q) = dU/dq_c` by central finite difference on the potential energy `U(q)` (L185-194).
`tau_d` is logged as `tau_des_Nm`; it is nonzero only in `grasp_hold`.

**(b) Admittance wrapper** -- `HandAdmittancePH.step`, `g1_hw_ph_finger_control.py` L460-477.
The Inspire hand firmware has no torque input, so `tau_ideal` becomes a position reference:

```
tau_contact = -contact_gain * max(|F_ACT| - force_guard_n, 0)                       (L462-463)
ddq_cmd     = (tau_ideal + tau_contact + K_track*(q_act - q_cmd) - B_a*dq_cmd) / M_a (L467-468)
dq_cmd     += ddq_cmd * dt                                                          (L469)
q_cmd       = clip(q_cmd + clip(dq_cmd*dt, +-qdot_max*dt), q_lo, q_hi)              (L471-473)
if |F_ACT| >= force_limit_n:  q_cmd := q_act,  dq_cmd := 0                            (L475-477)
```

Equivalent to `M_a*ddq_cmd + B_a*dq_cmd + K_track*(q_cmd - q_act) = tau_ideal + tau_contact`.
`q_cmd` is sent to the hand over Modbus and logged as `qdes_rad`. `F_ACT` is converted from
grams-force to newtons before `step()` (see the docstring at L430-438).

### 4.4 Caveats when quoting these in the thesis

- The admittance and PH gains are placeholders, not bench-tuned (see section 2).
- In `hand_state`, `qerr_rad` is measured against the admittance reference `q_cmd`, not the
  segment's nominal target.
- The arm PH law (4.1) and the hand PH law (4.3) are separate controllers on separate
  hardware and are never active in the same log; `arm_positions` is the arm's plain-PD
  companion to the hand PH log.

---

## How each file was identified (and renamed)

The files' **original** names in `Real_logs/` didn't line up 1:1 with script names
(`port_hamiltoinain_finger` did not literally say "grasp_ph"), so identification was done
by matching each file's **header schema and segment-name sequence** against the exact
`csv.writer` header/segment code in each candidate script, not by filename. Files were
then renamed to the convention each producing script itself uses for its own log output
(see the note at the top of this doc for the old -> new mapping):

- `g1_hw_arm_ph_control.py` header (`main()`, ~L538-543): `mode` column present, 81 total
  columns, segment names `hold_in/move_to_pose/hold_torque/return/release` -- matches
  `arm_ph_ph_20260918_183908.csv` (`mode` always `ph`, old name
  `Porthamiltnian_control.csv`) and `arm_ph_position_20260918_184048.csv` (`mode` always
  `position`, old name `Port_Hamiltonian_control2.csv`) exactly, including row count (172)
  and duration (17.08 s = the script's own 2+4+5+4+2 s segment lengths). The first file
  was further confirmed byte-identical to the script's own untouched output,
  `scripts/g1_control/hardware/grasp_logs/arm_ph_ph_20260918_183908.csv`.
- `grasp_ph.py` arm-log header (~L1048-1051): no `mode` column, 66 columns, segment names
  from its own `build_segments()` (`hold, side_up, approach_out, ...`) -- matches
  `arm_positions_20260918_185109.csv` exactly (old name
  `positional_control_port_hamiltonian.csv`; 1196 rows, 36.72 s).
- `grasp_ph.py` hand-log header (~L1054-1069): 7 columns/joint including `qdes_rad`,
  `qerr_rad`, `tau_des_Nm` -- matches `hand_state_20260918_185109.csv` exactly (old name
  `port_hamiltoinain_finger.csv`; 1195 rows, same 14 segment names, same run_ts as the arm
  log above). `g1_hw_ph_finger_control.py`'s *own* standalone `--plot` logger
  (~L1015-1018) only writes 4 columns/joint (no qdes/qerr/tau_des) and uses
  `default_segments()`'s 3-segment set (`hold, grip_cylinder, release`) -- ruled out on
  both column count and segment names.
