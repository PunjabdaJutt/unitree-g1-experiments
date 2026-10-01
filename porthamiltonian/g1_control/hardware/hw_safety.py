# hw_safety.py
"""Shared safety gate for scripts in this directory that publish rt/arm_sdk.

See safety_config.json (same directory) for the incident this exists to
prevent: a script was run while the robot was in Damp (fsm_id 1); Damp does
not act on rt/arm_sdk, so the arms stayed still -- but the command was still
cached/latent, and switching to Locked Standing (fsm_id 4) afterwards made
the robot instantly snap to that stale, unrelated command.

Every script here that publishes rt/arm_sdk should, in this order:
  1. flush_stale_arm_sdk()   -- before anything else, overwrite whatever is
                                 currently latent with a neutral command.
  2. require_locked_standing() -- confirm the robot is in the one fsm_id
                                 this repo has verified is safe, abort
                                 otherwise. Never auto-switch fsm state here.
  3. ... run the real control loop ...
  4. publish_neutral_arm_sdk() again on every exit path (normal, Ctrl+C,
     exception) so nothing is left retained at nonzero weight.
"""
import json
import os
import time

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "safety_config.json")
with open(_CONFIG_PATH) as _f:
    SAFETY_CONFIG = json.load(_f)

ALLOWED_FSM_IDS = set(SAFETY_CONFIG["required_state"]["allowed_fsm_ids"])
FSM_NAMES = SAFETY_CONFIG["required_state"].get("fsm_names", {})
K_NOT_USED_JOINT = 29  # G1JointIndex.kNotUsedJoint: 1=arm_sdk enabled, 0=released to whole-body FSM


class RobotNotReadyError(RuntimeError):
    """Raised when the robot is not confirmed to be in an allowed fsm_id."""


def require_locked_standing(timeout: float = 5.0):
    """Abort unless the robot is confirmed in an allowed fsm_id (see
    safety_config.json's required_state.allowed_fsm_ids -- fsm 4 Locked
    Standing by default, plus any state deliberately added there).

    Call this after ChannelFactoryInitialize(), before starting any control
    loop that publishes rt/arm_sdk. Deliberately does NOT switch fsm state
    itself -- that responsibility stays with the operator (via
    g1_mode_switcher.py), so a hardware test script can never trigger the
    "mode switch reveals a stale command" failure mode on its own.
    """
    client = LocoClient()
    client.SetTimeout(timeout)
    client.Init()
    code, fsm_id = client.GetFsmId()
    if code != 0 or fsm_id not in ALLOWED_FSM_IDS:
        allowed_str = ", ".join(f"{i} ({FSM_NAMES.get(str(i), '?')})" for i in sorted(ALLOWED_FSM_IDS))
        raise RobotNotReadyError(
            f"robot is not in an allowed fsm_id (fsm_id={fsm_id}, rpc code={code}; "
            f"allowed: {allowed_str}). Run "
            "`python .unitree_sdk2_python_src/modes/g1_mode_switcher.py <iface> "
            "--mode 3` first and confirm it reports fsm 4, THEN re-run this "
            "script. Do not switch modes while this script (or any rt/arm_sdk "
            "publisher) is running -- see safety_config.json's incident notes."
        )
    print(f"  -> safety check ok: robot confirmed in fsm_id={fsm_id} ({FSM_NAMES.get(str(fsm_id), '?')})")


def _neutral_low_cmd() -> LowCmd_:
    cmd = unitree_hg_msg_dds__LowCmd_()
    for i in range(K_NOT_USED_JOINT + 1):
        mc = cmd.motor_cmd[i]
        mc.q = 0.0
        mc.dq = 0.0
        mc.tau = 0.0
        mc.kp = 0.0
        mc.kd = 0.0
    # Redundant with the loop above (kNotUsedJoint index already zeroed) --
    # spelled out because this is the one field that actually matters:
    # weight=0 releases arm_sdk back to the whole-body FSM regardless of
    # whatever stale q/kp/kd values a previous command left behind.
    cmd.motor_cmd[K_NOT_USED_JOINT].q = 0.0
    return cmd


def publish_neutral_arm_sdk(pub: ChannelPublisher, cycles: int = 10, dt: float = 0.02):
    """Publish `cycles` neutral (weight=0) rt/arm_sdk samples on `pub`.

    Safe to call regardless of current fsm state -- weight=0 means "arm_sdk
    is released," so it never commands a specific joint position. Use this
    both before starting a run (to clear a stale retained command) and on
    every exit path (to avoid leaving one behind).
    """
    crc = CRC()
    cmd = _neutral_low_cmd()
    for _ in range(cycles):
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        time.sleep(dt)


def flush_stale_arm_sdk(cycles: int = 10, dt: float = 0.02) -> ChannelPublisher:
    """Create a fresh rt/arm_sdk publisher and immediately neutralize it.

    Call this first thing after ChannelFactoryInitialize(), before doing
    anything else -- see module docstring. Returns the publisher in case the
    caller wants to reuse it instead of creating its own.
    """
    pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    pub.Init()
    publish_neutral_arm_sdk(pub, cycles=cycles, dt=dt)
    print(f"  -> flushed {cycles} neutral rt/arm_sdk samples (weight=0) to clear any stale retained command")
    return pub
