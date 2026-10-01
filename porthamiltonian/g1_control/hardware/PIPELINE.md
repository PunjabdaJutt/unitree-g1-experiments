# Sim -> real-hardware porting pipeline

Status of porting the PH (Port-Hamiltonian IDA-PBC) finger controller and the
two-arm reach/grasp task off Isaac Lab / MuJoCo and onto the physical G1.
Each stage only depends on the previous one being bench-verified -- don't
skip ahead. All hardware scripts here keep the legs under the robot's own
locomotion FSM (no `rt/lowcmd` publish, ever); only waist/arms (`rt/arm_sdk`)
and the Inspire hands (Modbus TCP) are actuated.

## Stage 0 -- Sim validation (done)
- `lift_fingerPH.py` (Isaac Lab) and `mujoco_lift_finger.py` (MuJoCo): full
  pick-cylinder-place-in-box task, two-hand DLS IK for arms/waist, PH IDA-PBC
  (`ph/ph_impedance.PHContactForceController`) as the sole finger actuator.
  `analyze_controller.py` / `analyze_step_response.py` verify the PH law's
  passivity (dissipation inequality) and step response against theory.
- Confirms: the control law itself, gain ranges (`--ph_kp`, `--K_spr`,
  `--grasp_torque`, etc.), and the HAND12 joint/limit/mimic conventions in
  `ph/robot_indices.py`.
- Does NOT confirm: anything about the real actuators. Sim applies the PH
  torque directly; the real Inspire hand has no torque input at all (see
  Stage 2).

## Stage 1 -- Hardware connectivity smoke tests (do this first, on hardware)
- `tools/inspire_hand_modbus_read.py`, `tools/inspire_hand_modbus_control.py`:
  outside this nested repo, already confirm both hands are reachable at
  `192.168.123.210` / `.211`, healthy (ERROR=0), and manually drivable.
- `g1_hw_finger_close_only.py --dry_run`: connects to both hands over
  Modbus, reads current angles + error flags once, prints them, sends
  nothing. Run this before any script here that actually commands motion --
  it validates the network path and register map without moving anything.
- Also worth a one-off check per the reference article
  (dev.classmethod.jp/articles/unitree-g1-inspire-hand-demo-deploy): confirm
  `rt/lowstate` is actually publishing on this robot (the
  `ChannelSubscriber` snippet in that article's "DDS Connection
  Verification" section). This repo's own finding is that `rt/inspire/state`
  does NOT publish here (hence Modbus for hand I/O everywhere in
  `hardware/`), but `rt/lowstate` (body joint state, unrelated topic) has
  been assumed to work -- verify rather than assume before Stage 3+.

## Stage 2 -- Hardware hands-only PH closed loop (current step)
**`g1_hw_finger_close_only.py`** -- close the fingers using the PH
controller, hold, reopen. No arm/waist motion (never publishes
`rt/arm_sdk`/`rt/lowcmd`); reads `rt/lowstate` only passively, for the
gravity-comp term.

This is an **admittance approximation**, not real torque control: the
Inspire firmware only accepts a position setpoint + a stall-force ceiling,
so `PHContactForceController.compute()`'s output torque is integrated into a
position increment (`q_cmd += admittance_gain * tau_ideal * dt`) rather than
applied directly. `--admittance_gain` has no theoretically-correct value --
the hand's internal position-servo bandwidth is undocumented. Bench-tuning
procedure:
1. `--dry_run` (Stage 1) to confirm connectivity.
2. Run with `--admittance_gain 0.02 --close_fraction 0.2` (very
   conservative), e-stop in reach, workspace clear of pinch points.
3. If the fingers barely move / feel sluggish: raise `--admittance_gain` in
   small steps. If they oscillate or overshoot the commanded angle
   noticeably: lower it, or raise `--hand_qdot_max`'s divisor effect by
   lowering `--hand_qdot_max` itself (hard velocity cap, independent knob).
4. Once stable at `--close_fraction 0.2`, raise toward the default `0.4`,
   then whatever closure the real grasp task needs -- cross-check against
   `tools/inspire_hand_modbus_control.py`'s FORCE_ACT readback so the grip
   force stays sane before ever closing on an object.
5. Record what `--admittance_gain` / `--hand_qdot_max` converged on; those
   become the defaults for Stage 3.

## Stage 3 -- Hardware arm + hand demo (exists, not yet bench-verified)
**`g1_hw_ph_finger_control.py`** -- same PH admittance hand loop as Stage 2,
plus `rt/arm_sdk` position-PD for the 17 waist/arm joints (verbatim pattern
from `g1_hw_arm_hand_test.py`). Default `default_segments()` holds the arms
at their start pose the whole time (safest default) -- edit `arm_targets` in
that function once Stage 2's gains are confirmed and you want to test arm
reaching too. Its `open -> approach_close -> grasp_hold (force mode) -> open
-> release` finger sequence is a superset of Stage 2's close-only test, so
don't run it until Stage 2 has produced known-good `--admittance_gain`.

`g1_hw_arm_hand_test.py` is the older, arm+hand-over-DDS variant (talks to
the hands via `rt/inspire/cmd`/`rt/inspire/state`, not Modbus) -- per Stage 1
above, that DDS path is believed non-functional on this robot for hand
*state* (unconfirmed whether hand *commands* over DDS would work write-only;
untested). Treat it as a reference for the `rt/arm_sdk` pattern only, not as
a runnable hand-control path, unless Stage 1's DDS check says otherwise.

## Stage 4 -- Full pick-and-place port (not started)
Port `lift_fingerPH.py` / `mujoco_lift_finger.py`'s `S1-pregrasp .. S10-idle`
choreography and camera-driven cube detection onto real hardware:
- Two-hand DLS IK (`ph/kinematics.ik_two_hands_dls`) is pure numpy -- no
  porting needed, just call it with the real arm state as `q0` each segment
  transition (as Stage 3 already does for its arm targets, just without IK).
- Object detection: replace the Isaac/MuJoCo semantic-segmentation +
  depth-unproject pipeline with the real D435 feed. `tools/camera_grabber.py`
  and `ros2_ws/src/g1_bridge/g1_bridge/camera_view.py` already pull frames
  from the real camera in this repo -- reuse that path rather than
  reimplementing capture; only the detection (`cube_centroid_in_camera_frame`
  in `lift_fingerPH.py`) needs a real-image equivalent (e.g. color/depth
  thresholding or a small detector), since there's no ground-truth semantic
  mask on hardware.
- Grasp-target IK (`ph/ik_fingers_analytical.py`, `ph/fk_fingers.py`): pure
  numpy, reusable as-is once real fingertip/object positions are available.
- Finger force mode (S3-S5's `set_force_mode` / `tau_des_grasp`): same
  admittance-wrapper caveat as Stage 2/3 -- force mode becomes "drive toward
  the virtual-spring equilibrium angle" on hardware, not literal torque
  regulation. Validate equilibrium angles per `lift_fingerPH.py`'s own
  `--K_spr` docstring (`q* = tau_d/K_spr`) land inside the real finger's
  travel before trusting a grasp.
- Everything above only composes once Stage 3's arm reaching (not just
  arm-hold) has been bench-verified with real IK targets, not just the
  fixed start-pose hold Stage 3 defaults to.

## Stage 5 -- Hardening (not started, do alongside/after Stage 4)
- Logging: `lift_fingerPH.py`'s CSV logging + `analyze_controller.py` /
  `analyze_step_response.py` already give a template for turning a run into
  a diagnostic figure -- point the same analysis at hardware-run CSVs once a
  hardware script emits them (none currently do).
- A supervisory watchdog beyond the current per-tick error-flag abort
  (Modbus read failures currently just skip a tick and hold last command --
  decide a max-consecutive-failures bound before that becomes "stop").
- `--dry_run`-style connect-and-read-only modes (Stage 2 has one; Stage 3
  doesn't yet) before any script that also drives `rt/arm_sdk`.
