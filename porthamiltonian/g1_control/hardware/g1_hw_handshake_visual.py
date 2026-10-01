
# g1_hw_handshake_visual.py
"""Real-hardware G1 handshake with camera-driven visual approach: right arm
only, closed-loop, force-limited.

Unlike g1_hw_handshake.py (camera only TRIGGERS a fixed HANDSHAKE_POSE),
this script continuously steers the right shoulder/elbow toward the
detected hand using its image-plane position and apparent size (see
hand_detector.HandPresenceDetector.hand_center_and_size()) -- closing in as
the hand appears bigger in frame -- then grips once close, holds, releases,
and retracts. There is still no metric depth/calibration (see
hand_detector.py's module docstring); this is image-based visual servoing,
not real Cartesian/IK reaching, so it only needs consistent sign/scale, not
a calibrated camera.

Sequence: SEARCH (arm at rest, watch for a hand) -> APPROACH (visual servo
toward the hand, aborting back to SEARCH if it's lost for --lost_timeout_s)
-> CLOSE (freeze pose, grip with FORCE_SET capped at HAND_FORCE_G grams,
hold, release, retract) -> back to SEARCH.

The whole session runs under ONE continuously-held rt/arm_sdk weight,
ramped in once at startup and only released in Shutdown() when the script
actually exits -- deliberately NOT re-instantiating a fresh Custom (as
g1_hw_handshake.py does per handshake cycle) for the CLOSE phase, since
that would drop the weight toward 0 and briefly release the arm to the
whole-body FSM between phases, which is exactly the "stale/released
command" hazard hw_safety.py exists to prevent (see safety_config.json).

Left arm, waist, and left hand are never targeted -- same convention as
g1_hw_handshake.py.

On exit (normal completion, Ctrl+C, or an exception) HandshakeVisualController
.Shutdown() winds the arm_sdk weight down to 0 over --shutdown_ramp_s (2s by
default) while holding the arm at its current pose, rather than releasing it
in one step -- an instant release hands control straight to the whole-body
FSM's own arm controller (at a similarly high stiffness, arm_kp=60 by
default) from wherever the arm happened to be, which looks like a hard snap
if it happens mid-approach. See HandshakeVisualController.start_winddown().

Gated by hw_safety.py exactly like every other script here: flush_stale_arm
_sdk() then require_locked_standing() before any rt/arm_sdk publishing.

Run:
    python scripts/g1_control/hardware/g1_hw_handshake_visual.py eth0

Use --dry_run FIRST to validate the vision + steering logic against a live
camera with zero hardware risk: no rt/arm_sdk publishing, no Modbus hand
connections, no hw_safety gate, but the full SEARCH/APPROACH/CLOSE state
machine and target computation run and print, against the real camera.

    python scripts/g1_control/hardware/g1_hw_handshake_visual.py eth0 --dry_run

WARNING (non-dry-run): this drives real arm + hand hardware autonomously
as it tracks and closes in on a detected hand. Clear the workspace and
keep the e-stop within reach before running.
"""
import argparse
import itertools
import os
import sys
import time

# Must be the first project import -- see hand_detector.py's module
# docstring (its module-level reexec_with_clean_env() fixes LD_LIBRARY_PATH
# for the whole process before anything below transitively imports
# unitree_sdk2py/cv2).
import hand_detector

import numpy as np

from g1_hw_arm_hand_test import (
    G1JointIndex, ARM_JOINT_NAMES, ARM_JOINTS,
    FINGER_ORDER, FINGER_LIMITS, Segment,
    build_finger_target, to_modbus_angle,
    InspireHand, DEFAULT_SPEED,
)
from g1_hw_handshake import HANDSHAKE_POSE, HAND_FORCE_G
import hw_safety
from hand_detector import HandPresenceDetector

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.go2.video.video_client import VideoClient

# ph package (kinematics for the reach-bounds check) lives at
# <PH repo root>/scripts/g1_control/ph -- same sys.path pattern as
# g1_hw_handshake.py's validate_reach().
_PH_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PH_REPO_ROOT not in sys.path:
    sys.path.insert(0, _PH_REPO_ROOT)
from scripts.g1_control.ph import default_urdf, fk_chain  # noqa: E402


# =============================================================================
# Visual-servo gains and bounds. UNTUNED, same as HANDSHAKE_POSE in
# g1_hw_handshake.py -- bench-validate on hardware before trusting these.
# Start a first bench trial with a smaller --max_step_rad_s than the
# default below.
#
# IMPORTANT -- role assignment was corrected after a hardware test showed the
# arm dropping down/back toward the robot instead of extending toward the
# person. FK on this URDF (fk_chain from the ph package) shows:
#   - INCREASING right_shoulder_pitch moves the wrist BACKWARD (toward/
#     behind the pelvis) and doesn't reliably raise it -- the opposite of
#     what "shoulder pitch raised" in HANDSHAKE_POSE's own comment suggests.
#     Around shoulder_pitch in [-1.15, -0.4] (NEGATIVE), by contrast, the
#     wrist's forward (+x) coordinate stays consistently positive and grows
#     as pitch decreases -- i.e. pitch is what actually drives reach toward
#     a person on this robot, and it needs to go negative to do it.
#   - right_elbow in that same pitch range barely changes forward reach (x
#     varies ~0.31-0.36m across its whole safe range) but changes HEIGHT (z)
#     monotonically and by a lot (dropping ~0.15m end to end) -- i.e. elbow
#     is a much better vertical-aim control here than a "closeness" control.
# So the mapping below is: shoulder_roll -> horizontal aim (unchanged),
# shoulder_pitch -> depth/closeness (was elbow's job), elbow -> vertical aim
# (was shoulder_pitch's job). See the FK sweep this was derived from -- run
# `python3 -c` snippets against scripts.g1_control.ph.{default_urdf,fk_chain}
# with right_shoulder_pitch_joint/right_wrist_yaw_link -- before changing
# these numbers again.
# =============================================================================
KP_ROLL = 0.3     # rad of shoulder_roll offset per unit of normalized horizontal centering error
KP_ELBOW = 0.3    # rad of elbow offset per unit of normalized vertical centering error
ELBOW_BASE = 0.3   # center of the elbow (vertical-aim) range -- lowered from an earlier 0.45 (a
                    # lower elbow angle means a HIGHER wrist, per the FK data above) since logged
                    # hardware runs mostly saw the hand in the upper half of frame (cy well under
                    # 0.5), so the useful elbow range needed more room below the old base than above.
SIZE_FAR = 0.12    # hand_center_and_size() 'size' below this -> retracted (pitch = PITCH_FAR)
SIZE_NEAR = 0.30   # 'size' at/above this -> full forward reach (pitch = PITCH_NEAR)
PITCH_FAR = -0.4    # retracted-but-still-forward-facing (x~+0.27, z~+0.10)
PITCH_NEAR = -1.15  # full reach toward the person (x~+0.33, z~+0.33)
SIZE_CLOSE_THRESHOLD = 0.32  # default for --close_size_threshold: 'size' at/above this, held for
                              # --close_debounce_s, triggers CLOSE (grip). Logged hardware runs
                              # showed this triggering almost immediately on detection (observed
                              # sizes were routinely 0.33-0.53) -- raise --close_size_threshold at
                              # the CLI to require the hand to look closer before it grips.

# Servo target envelope -- every commanded (shoulder_pitch, shoulder_roll,
# elbow) is clamped inside this box, which is FK-checked against the arm's
# real max reach at startup (see validate_bounds()) before any 1hardware run.
# Worst-case corner clears the required 0.02m margin by ~0.027m (arm max
# reach ~0.375m on this URDF; worst bounds corner ~0.348m). Within this box,
# the wrist's forward (+x) coordinate is positive at every corner checked
# (see the FK sweep in the comment above) -- unlike the old (0.0,0.6)/
# (-0.4,0.4)/(0.15,0.65) bounds, which put x negative or near-zero
# everywhere and caused the earlier "drops toward the robot" behavior.
# (The elbow range was previously widened to (0.2,0.7), which actually
# FAILED this check by 0.004m -- narrowed back here, with room shifted
# downward to match the new lower ELBOW_BASE.)
SAFE_JOINT_BOUNDS = {
    "right_shoulder_pitch": (-1.15, -0.4),
    "right_shoulder_roll": (-0.25, 0.25),
    "right_elbow": (-0.2, 0.55),
}


def validate_bounds(bounds: dict, wrist_fixed: dict, margin_m: float = 0.02) -> None:
    """FK-check every corner of the servo target envelope (bounds x the
    fixed wrist joints) against the arm's max reach -- same spirit as
    g1_hw_handshake.validate_reach(), but for the whole box of targets the
    visual-servo phase can command, not just one fixed pose. Raises if any
    corner exceeds max reach; startup sanity check, not a runtime clamp
    (SAFE_JOINT_BOUNDS itself is the runtime clamp)."""
    urdf = default_urdf()
    shoulder = fk_chain(urdf, "pelvis", "right_shoulder_pitch_link")[:3, 3]

    elbow = urdf.joints_by_name["right_elbow_joint"]
    best = 0.0
    for e in np.linspace(elbow.lower, elbow.upper, 60):
        w = fk_chain(urdf, "pelvis", "right_wrist_yaw_link", {"right_elbow_joint": float(e)})[:3, 3]
        best = max(best, float(np.linalg.norm(w - shoulder)))

    keys = list(bounds.keys())
    worst = 0.0
    for combo in itertools.product(*[bounds[k] for k in keys]):
        pose = {**wrist_fixed, **dict(zip(keys, combo))}
        pose_j = {f"{k}_joint": v for k, v in pose.items()}
        wrist = fk_chain(urdf, "pelvis", "right_wrist_yaw_link", pose_j)[:3, 3]
        worst = max(worst, float(np.linalg.norm(wrist - shoulder)))

    print(f"  -> bounds check: worst-case servo target reach = {worst:.3f}m, "
          f"arm max (elbow swept) = {best:.3f}m")
    if worst > best - margin_m:
        raise ValueError(
            f"SAFE_JOINT_BOUNDS worst-case reach ({worst:.3f}m) is not comfortably under "
            f"the arm's max reach ({best:.3f}m, {margin_m:.3f}m margin) -- tighten "
            "SAFE_JOINT_BOUNDS before running on hardware."
        )


def close_segments(hold_close_s: float = 1.5) -> list:
    """Segment list for the CLOSE phase: grip -> hold -> release ->
    retract. No 'reach' segment -- the visual-servo phase already got the
    arm to the hand, so this starts from wherever it left off (Custom-style
    'omitted arm_targets = hold')."""
    open_R = {f: 0.0 for f in FINGER_ORDER}
    close_R = {"pinky": 1.6, "ring": 1.6, "middle": 1.6, "index": 1.6,
               "thumb_bend": 0.6, "thumb_rot": 0.0}
    rest = {**{n: 0.0 for n in HandshakeVisualController.SERVO_NAMES},
            **{n: 0.0 for n in HandshakeVisualController.FIXED_NAMES}}
    return [
        Segment("grip", 2.0, hand_R=close_R),  # hand firmware stops early at FORCE_SET
        Segment("hold_shake", hold_close_s),
        Segment("release", 1.5, hand_R=open_R),
        Segment("retract", 3.0, arm_targets=rest),
    ]


# =============================================================================
# Controller
# =============================================================================
class HandshakeVisualController:
    """Right-arm rt/arm_sdk + Inspire-hand controller for the full session:
    idle-at-rest -> visual-servo approach -> grip/hold/release/retract ->
    back to idle -- see module docstring for why this stays one
    continuously-running controller instead of chaining separate Custom
    instances per phase.

    Two target-following modes share one slew-rate-limited per-tick
    follower: 'servo' (a live `target` dict, updated every camera frame by
    set_target_from_frame()) and 'segment' (a Segment list played back with
    linear ramps, like Custom, for the fixed grip/hold/release/retract
    close-out that doesn't need visual feedback). Only one is active at a
    time; only 'segment' mode ever changes the hand targets.
    """

    SERVO_NAMES = ["right_shoulder_pitch", "right_shoulder_roll", "right_elbow"]
    FIXED_NAMES = ["right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]

    def __init__(self, control_dt: float = 0.02, arm_kp: float = 20.0, arm_kd: float = 0.5,
                 max_step_rad_s: float = 0.5, hand_speed: int = DEFAULT_SPEED,
                 hand_force: int = HAND_FORCE_G,
                 left_hand_ip: str = "192.168.123.210", right_hand_ip: str = "192.168.123.211",
                 dry_run: bool = False):
        self.control_dt = control_dt
        self.arm_kp = arm_kp
        self.arm_kd = arm_kd
        self.max_step_per_tick = max_step_rad_s * control_dt
        self.dry_run = dry_run

        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self._have_low_state = False

        self.hands = None if dry_run else {"L": InspireHand(left_hand_ip), "R": InspireHand(right_hand_ip)}
        self.hand_speed = hand_speed
        self.hand_force = hand_force
        self.hand_q_target = {
            "L": {f: FINGER_LIMITS[f][0] for f in FINGER_ORDER},
            "R": {f: FINGER_LIMITS[f][0] for f in FINGER_ORDER},
        }

        # q_cmd/target cover ALL 17 rt/arm_sdk joints (ARM_JOINT_NAMES), not just the 6 this
        # controller actively steers -- arm_sdk's kNotUsedJoint weight is a single blend value
        # for the whole 17-joint group, not per-joint. Any joint left out of the per-tick
        # motor_cmd writes below would sit at LowCmd_'s zero-initialized kp=kd=0 while arm_sdk
        # still holds authority over it, i.e. commanded to zero torque instead of left under the
        # whole-body FSM -- that's what caused the waist/left-arm/right_shoulder_yaw to droop.
        # Start() fills these with the measured pose for every joint; only SERVO_NAMES/
        # FIXED_NAMES ever get a *different* target after that (set_mode_rest(),
        # set_target_from_frame(), close_segments()) -- everything else holds its Start()-time
        # value for the whole session, same as Custom's "omitted = hold" in g1_hw_arm_hand_test.py.
        self._rest = {n: 0.0 for n in ARM_JOINT_NAMES}
        self.q_cmd = dict(self._rest)
        self.target = dict(self._rest)

        self._mode = "servo"  # "servo" or "segment"
        self._segments: list = []
        self._seg_idx = 0
        self._seg_t0 = 0.0
        self._seg_q_start = dict(self._rest)
        self.segment_done = False

        self.t = 0.0
        self._weight = 0.0
        self._enable_ramp_T = 2.0

        self._winddown = False
        self._winddown_T = 2.0
        self._winddown_w0 = 0.0
        self._winddown_t0 = 0.0
        self._winddown_hold: dict = {}

    def Init(self):
        if self.dry_run:
            return
        self.arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_pub.Init()
        self.low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_sub.Init(self._on_low_state, 10)
        for side, hand in self.hands.items():
            if not hand.connect():
                raise RuntimeError(
                    f"failed to connect to Inspire {side} hand at {hand.ip} over Modbus TCP "
                    "-- check it's on the robot's 192.168.123.0/24 network before retrying"
                )
            hand.configure(self.hand_speed, self.hand_force)

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg
        self._have_low_state = True

    def Start(self):
        if not self.dry_run:
            while not self._have_low_state:
                time.sleep(0.1)
            for name in ARM_JOINT_NAMES:
                j = ARM_JOINTS[ARM_JOINT_NAMES.index(name)]
                self.q_cmd[name] = self.low_state.motor_state[j].q
        self.target = dict(self.q_cmd)
        self.thread = RecurrentThread(interval=self.control_dt, target=self._tick, name="handshake_visual")
        self.thread.Start()

    # ---- servo mode -----------------------------------------------------
    # Idle/rest target for the servoed joints: the retracted end of the safe
    # reaching envelope (SAFE_JOINT_BOUNDS), NOT (0,0,0). (0,0,0) is a
    # perfectly safe pose on its own (used by close_segments()'s "retract",
    # which isn't bounds-clamped -- see _tick()), but shoulder_pitch=0 and
    # elbow=0 both sit outside SAFE_JOINT_BOUNDS now that pitch has to go
    # negative to reach forward (see the comment above SAFE_JOINT_BOUNDS):
    # requesting them here would just get silently clamped to the bound
    # edge every tick anyway, so make that the explicit, intentional target
    # instead of relying on the clamp to paper over it.
    # right_elbow uses the UPPER bound here, not the lower one -- higher
    # elbow angle means a LOWER wrist on this URDF (see the FK comment
    # above SAFE_JOINT_BOUNDS), so this is the lowest-hanging rest pose
    # available inside the already-validated safe envelope. Using the
    # lower bound (0.1) instead gave the highest, not lowest, rest height
    # (z~+0.17m vs z~+0.08m at pitch=-0.4) -- that's what made it "high."
    _REST_SERVO = {
        "right_shoulder_pitch": PITCH_FAR,
        "right_shoulder_roll": 0.0,
        "right_elbow": SAFE_JOINT_BOUNDS["right_elbow"][1],
    }

    def set_mode_rest(self):
        """Retarget the servoed right-arm joints to _REST_SERVO (see
        above) and the fixed wrist joints to 0. Every other arm_sdk joint
        (waist, left arm, right_shoulder_yaw) is left alone -- self.target
        for those never changes after Start(), so they stay held at
        wherever they were when the script started."""
        self._mode = "servo"
        for name, val in self._REST_SERVO.items():
            self.target[name] = val
        for name in self.FIXED_NAMES:
            self.target[name] = 0.0

    def set_target_from_frame(self, cx: float, cy: float, size: float):
        """Absolute proportional mapping from the hand's image-plane
        position/size to a right-arm target -- NOT integrated/accumulated,
        so a noisy or momentarily-lost detection can't build up drift.
        Every value is clamped into SAFE_JOINT_BOUNDS before being used.

        Role assignment (see the comment above SAFE_JOINT_BOUNDS for the FK
        data behind this): shoulder_roll steers horizontal aim (cx),
        shoulder_pitch steers depth/closeness (as the hand's apparent
        `size` grows, pitch sweeps PITCH_FAR -> PITCH_NEAR, extending the
        arm toward the person), elbow steers vertical aim (cy)."""
        self._mode = "servo"
        roll = 0.0 + KP_ROLL * (0.5 - cx)
        size_ratio = float(np.clip((size - SIZE_FAR) / (SIZE_NEAR - SIZE_FAR), 0.0, 1.0))
        pitch = PITCH_FAR + size_ratio * (PITCH_NEAR - PITCH_FAR)
        elbow = ELBOW_BASE + KP_ELBOW * (cy - 0.5)
        raw = {"right_shoulder_roll": roll, "right_shoulder_pitch": pitch, "right_elbow": elbow}
        for name in self.SERVO_NAMES:
            lo, hi = SAFE_JOINT_BOUNDS[name]
            self.target[name] = float(np.clip(raw[name], lo, hi))
        for name in self.FIXED_NAMES:
            self.target[name] = HANDSHAKE_POSE[name]

    def current_pose(self) -> dict:
        return dict(self.q_cmd)

    # ---- segment mode (grip / hold / release / retract) ------------------
    def run_close_sequence(self, segments: list, poll_dt: float = 0.1):
        """Blocking: play `segments` to completion on the SAME thread and
        arm_sdk weight session that was already running -- no weight dip
        between the servo phase that preceded this and this sequence."""
        self._segments = segments
        self._seg_idx = -1
        self.segment_done = False
        self._load_segment(0)
        self._mode = "segment"
        while not self.segment_done:
            time.sleep(poll_dt)
            print("\r" + self.status_line() + " " * 8, end="", flush=True)
        print()

    def _load_segment(self, i: int):
        seg = self._segments[i]
        self._seg_idx = i
        self._seg_t0 = self.t
        self._seg_q_start = dict(self.q_cmd)
        if seg.arm_targets:
            for name, val in seg.arm_targets.items():
                self.target[name] = float(val)
        self.hand_q_target = build_finger_target(self.hand_q_target, seg.hand_L, seg.hand_R)
        if not self.dry_run:
            if seg.hand_L:
                self.hands["L"].set_angles([to_modbus_angle(f, self.hand_q_target["L"][f]) for f in FINGER_ORDER])
            if seg.hand_R:
                self.hands["R"].set_angles([to_modbus_angle(f, self.hand_q_target["R"][f]) for f in FINGER_ORDER])
        print(f"\n[SEG] -> {seg.name} (T={seg.T:.1f}s)")

    # ---- winddown (graceful release on Shutdown) ----------------------
    def start_winddown(self, T: float = 2.0):
        """Freeze the arm at its current pose and ramp the arm_sdk weight
        down to 0 over T seconds, instead of releasing it in one abrupt
        step. An instant weight drop hands control straight to the
        whole-body FSM's own arm controller from whatever pose the arm
        happened to be in -- with the same effectively-high stiffness
        (arm_kp=60) the FSM's own controller uses -- which is what looks
        like a hard snap on Ctrl+C. Call this before stopping the control
        thread; _tick() ramps self._weight -> 0 while holding q_cmd fixed
        until winddown_done()."""
        self._winddown_hold = dict(self.q_cmd)
        self._winddown_w0 = self._weight
        self._winddown_t0 = self.t
        self._winddown_T = max(T, 1e-6)
        self._winddown = True

    def winddown_done(self) -> bool:
        return self._winddown and self._weight <= 0.0

    # ---- shared tick -------------------------------------------------
    def _tick(self):
        self.t += self.control_dt

        if self._winddown:
            ratio = float(np.clip((self.t - self._winddown_t0) / self._winddown_T, 0.0, 1.0))
            self._weight = self._winddown_w0 * (1.0 - ratio)
            self.q_cmd = dict(self._winddown_hold)
            if self.dry_run:
                return
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = self._weight
            for name, val in self.q_cmd.items():
                j = ARM_JOINTS[ARM_JOINT_NAMES.index(name)]
                mc = self.low_cmd.motor_cmd[j]
                mc.tau = 0.0
                mc.q = float(val)
                mc.dq = 0.0
                mc.kp = self.arm_kp
                mc.kd = self.arm_kd
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.arm_pub.Write(self.low_cmd)
            return

        self._weight = min(1.0, self._weight + self.control_dt / self._enable_ramp_T)

        if self._mode == "segment":
            seg = self._segments[self._seg_idx]
            seg_t = self.t - self._seg_t0
            ratio = float(np.clip(seg_t / max(seg.T, 1e-6), 0.0, 1.0))
            for name in ARM_JOINT_NAMES:
                start = self._seg_q_start[name]
                self.q_cmd[name] = (1.0 - ratio) * start + ratio * self.target[name]
            if seg_t >= seg.T:
                if self._seg_idx + 1 < len(self._segments):
                    self._load_segment(self._seg_idx + 1)
                else:
                    self.segment_done = True
        else:
            # All 17 arm_sdk joints, not just SERVO_NAMES/FIXED_NAMES -- see the comment on
            # self._rest in __init__. For every joint besides the 6 actively steered ones,
            # self.target never changes after Start(), so tgt == q_cmd and step == 0: this
            # just re-publishes their held pose every tick, it doesn't move them.
            for name in ARM_JOINT_NAMES:
                lo_hi = SAFE_JOINT_BOUNDS.get(name)
                tgt = self.target[name]
                if lo_hi:
                    tgt = float(np.clip(tgt, *lo_hi))
                step = float(np.clip(tgt - self.q_cmd[name], -self.max_step_per_tick, self.max_step_per_tick))
                self.q_cmd[name] += step

        if self.dry_run:
            return

        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = self._weight
        for name, val in self.q_cmd.items():
            j = ARM_JOINTS[ARM_JOINT_NAMES.index(name)]
            mc = self.low_cmd.motor_cmd[j]
            mc.tau = 0.0
            mc.q = float(val)
            mc.dq = 0.0
            mc.kp = self.arm_kp
            mc.kd = self.arm_kd
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_pub.Write(self.low_cmd)

    def status_line(self) -> str:
        q = np.array([self.q_cmd[n] for n in (*self.SERVO_NAMES, *self.FIXED_NAMES)])
        q_str = np.array2string(q, precision=2, suppress_small=True, max_line_width=200)
        return f"[{self._mode:7s}] q={q_str}"

    def Shutdown(self, ramp_down_T: float = 2.0):
        """Stop the control thread and release rt/arm_sdk before exiting --
        call on every exit path (normal, Ctrl+C, exception).

        First winds the arm_sdk weight down to 0 over `ramp_down_T`
        seconds via start_winddown() (holding the arm at its current pose,
        not moving it), so control hands back to the whole-body FSM
        gently instead of all at once -- then, same as
        g1_hw_arm_hand_test.Custom.Shutdown(), calls
        hw_safety.publish_neutral_arm_sdk() as the final guarantee that
        nothing is left retained at nonzero weight."""
        if not self.dry_run and hasattr(self, "thread") and hasattr(self, "arm_pub"):
            self.start_winddown(ramp_down_T)
            while not self.winddown_done():
                time.sleep(0.05)
        if hasattr(self, "thread"):
            self.thread.Wait(1.0)
        if not self.dry_run:
            if hasattr(self, "arm_pub"):
                hw_safety.publish_neutral_arm_sdk(self.arm_pub)
            if self.hands:
                for hand in self.hands.values():
                    hand.close()


# =============================================================================
# Main state machine: SEARCH -> APPROACH -> CLOSE -> SEARCH
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="G1 real-hardware handshake with camera-driven visual approach "
                     "(right arm only, force-capped grip)."
    )
    parser.add_argument("network_interface", nargs="?", default=None,
                         help="NIC on the robot's LAN (e.g. eth0). Omit to use the default.")
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--arm_kp", type=float, default=60.0)
    parser.add_argument("--arm_kd", type=float, default=1.5)
    parser.add_argument("--max_step_rad_s", type=float, default=0.5,
                         help="Max commanded-joint slew rate during visual servo, rad/s. "
                              "Start smaller for a first bench trial.")
    parser.add_argument("--shutdown_ramp_s", type=float, default=2.0,
                         help="Seconds to ramp the arm_sdk weight down to 0 on exit (normal, "
                              "Ctrl+C, or error), holding the arm at its current pose while doing "
                              "so, instead of releasing it to the whole-body FSM in one abrupt step.")
    parser.add_argument("--hand_speed", type=int, default=DEFAULT_SPEED,
                         help="Inspire hand ANGLE_SET ramp speed, 0-1000.")
    parser.add_argument("--left_hand_ip", default="192.168.123.210")
    parser.add_argument("--right_hand_ip", default="192.168.123.211")
    parser.add_argument("--cooldown_s", type=float, default=5.0,
                         help="Seconds to ignore new hand detections after a handshake.")
    parser.add_argument("--detect_window", type=int, default=20,
                         help="HandPresenceDetector debounce window: how many of the most recent "
                              "polled frames are considered for hand_present() (see --min_detections). "
                              "Raise this (with --min_detections) to require the hand to be seen "
                              "consistently over more frames before SEARCH -> APPROACH triggers.")
    parser.add_argument("--min_detections", type=int, default=12,
                         help="Of the last --detect_window polled frames, how many must see a hand "
                              "before hand_present() triggers SEARCH -> APPROACH.")
    parser.add_argument("--lost_timeout_s", type=float, default=2.0,
                         help="Abort back to SEARCH if the hand is lost for this long mid-approach.")
    parser.add_argument("--close_debounce_s", type=float, default=0.5,
                         help="How long 'size' must stay >= --close_size_threshold before triggering CLOSE.")
    parser.add_argument("--close_size_threshold", type=float, default=SIZE_CLOSE_THRESHOLD,
                         help="hand_center_and_size() 'size' at/above which the hand is considered "
                              "close enough to grip (raise this to require the hand to look closer "
                              "-- i.e. wait longer into the approach -- before CLOSE triggers).")
    parser.add_argument("--hold_shake_s", type=float, default=2.5,
                         help="Seconds to hold the grip closed during the handshake (CLOSE phase's "
                              "'hold_shake' segment) before releasing.")
    parser.add_argument("--show", action="store_true",
                         help="Open a live window showing the camera feed with detected hand "
                              "landmarks overlaid (needs a display -- X11 forwarding or similar if "
                              "running over SSH). Off by default.")
    parser.add_argument("--dry_run", action="store_true",
                         help="Print computed servo targets/state transitions against the real "
                              "camera; never publish to rt/arm_sdk or touch the Inspire hands, "
                              "and skip the hw_safety fsm gate. Use this before a hardware run.")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] no hardware will be touched -- printing computed targets/state only.")
    else:
        print("WARNING: this drives REAL arm + hand hardware AUTONOMOUSLY as it tracks and "
              "closes in on a detected hand. Clear the workspace and keep the e-stop within reach.")
        input("Press Enter to continue...")

    validate_bounds(SAFE_JOINT_BOUNDS, {n: HANDSHAKE_POSE[n] for n in HandshakeVisualController.FIXED_NAMES})

    if args.network_interface:
        ChannelFactoryInitialize(0, args.network_interface)
    else:
        ChannelFactoryInitialize(0)

    if not args.dry_run:
        # Safety gate (safety_config.json): clear any stale rt/arm_sdk command,
        # then confirm the robot is actually in an allowed fsm_id. Same contract
        # as every other script in this directory -- see hw_safety.py.
        hw_safety.flush_stale_arm_sdk()
        try:
            hw_safety.require_locked_standing()
        except hw_safety.RobotNotReadyError as e:
            print(f"[ABORT] {e}")
            sys.exit(1)

    video_client = VideoClient()
    video_client.SetTimeout(3.0)
    video_client.Init()
    detector = HandPresenceDetector(
        video_client, cooldown_s=args.cooldown_s, show=args.show,
        window=args.detect_window, min_detections=args.min_detections,
    )

    controller = HandshakeVisualController(
        control_dt=args.control_dt, arm_kp=args.arm_kp, arm_kd=args.arm_kd,
        max_step_rad_s=args.max_step_rad_s, hand_speed=args.hand_speed,
        left_hand_ip=args.left_hand_ip, right_hand_ip=args.right_hand_ip,
        dry_run=args.dry_run,
    )
    controller.Init()
    controller.Start()

    state = "SEARCH"
    close_since = None
    lost_since = None
    print("[INFO] watching for a hand -- will approach and shake automatically. Ctrl+C to stop.")
    try:
        while True:
            detector.poll()
            cs = detector.hand_center_and_size()

            if state == "SEARCH":
                controller.set_mode_rest()
                if detector.hand_present():
                    print("\n[STATE] SEARCH -> APPROACH")
                    detector.arm_cooldown()
                    state = "APPROACH"
                    close_since = None
                    lost_since = None

            elif state == "APPROACH":
                if cs is not None:
                    lost_since = None
                    cx, cy, size = cs
                    controller.set_target_from_frame(cx, cy, size)
                    tgt = {k: round(controller.target[k], 2) for k in controller.SERVO_NAMES}
                    print(f"\r[SERVO] cx={cx:.2f} cy={cy:.2f} size={size:.3f} target={tgt}" + " " * 8,
                          end="", flush=True)
                    if size >= args.close_size_threshold:
                        close_since = close_since or time.time()
                        if time.time() - close_since >= args.close_debounce_s:
                            print("\n[STATE] APPROACH -> CLOSE")
                            state = "CLOSE"
                    else:
                        close_since = None
                else:
                    lost_since = lost_since or time.time()
                    if time.time() - lost_since >= args.lost_timeout_s:
                        print("\n[STATE] APPROACH -> SEARCH (hand lost)")
                        state = "SEARCH"

            elif state == "CLOSE":
                controller.run_close_sequence(close_segments(hold_close_s=args.hold_shake_s))
                print(f"[INFO] handshake complete{' (dry run)' if args.dry_run else ''}")
                detector.arm_cooldown()
                state = "SEARCH"

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] stopped by user")
    finally:
        controller.Shutdown(ramp_down_T=args.shutdown_ramp_s)
        detector.close()


if __name__ == "__main__":
    main()
