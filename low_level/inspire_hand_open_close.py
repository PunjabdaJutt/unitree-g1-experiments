"""
inspire_hand_open_close.py

Interactive open/close control for the Unitree G1's Inspire DFX dexterous
hands over DDS, using low, "compliant" gains so the fingers move gently.

DDS interface (per unitreerobotics/dfx_inspire_service):
  - Command topic : "rt/inspire/cmd"   (unitree_go.msg.dds_.MotorCmds_)
  - State topic   : "rt/inspire/state" (unitree_go.msg.dds_.MotorStates_)
  - 12 motors total, indices 0-5 = right hand, 6-11 = left hand.
  - Per-hand joint order: pinky, ring, middle, index, thumb_bend, thumb_rotation.
  - Only `q` is documented as functional on the real hand firmware; `kp`/`kd`
    are still sent (small values) in case the driver honours them, but the
    hand's actual compliance mostly comes from its own current-limited
    actuators, not these gains.
  - q is normalized: 0.0 = fully closed, 1.0 = fully open.

IMPORTANT: verify HAND_KP/HAND_KD and OPEN_Q/CLOSE_Q against your own hand
before running unattended -- these are conservative defaults, not
manufacturer-certified limits.

Run:
  python3 inspire_hand_open_close.py <network_interface>
"""

import sys
import time

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorCmd_, MotorStates_
from unitree_sdk2py.utils.thread import RecurrentThread

NUM_MOTORS = 12
RIGHT_HAND_IDX = list(range(0, 6))
LEFT_HAND_IDX = list(range(6, 12))
JOINT_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]

OPEN_Q = 1.0
CLOSE_Q = 0.0

# Minimal/compliant gains, as requested.
HAND_KP = 0.6
HAND_KD = 0.05

CONTROL_DT = 0.02      # 50 Hz command rate
RAMP_SPEED = 0.6       # q units per second -- lower = gentler motion


class InspireHandController:
    def __init__(self):
        self.q_state = [0.5] * NUM_MOTORS   # last known measured q
        self.q_target = [0.5] * NUM_MOTORS  # commanded target q
        self.q_cmd = [0.5] * NUM_MOTORS      # ramped command actually sent
        self.state_received = False

    def Init(self):
        self.cmd_publisher = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self.cmd_publisher.Init()

        self.state_subscriber = ChannelSubscriber("rt/inspire/state", MotorStates_)
        self.state_subscriber.Init(self._StateHandler, 10)

    def _StateHandler(self, msg: MotorStates_):
        for i, s in enumerate(msg.states[:NUM_MOTORS]):
            self.q_state[i] = s.q
        if not self.state_received:
            # Seed the ramp from the hand's actual current pose, not a guess.
            self.q_cmd = list(self.q_state)
            self.q_target = list(self.q_state)
            self.state_received = True

    def Start(self):
        self.write_thread = RecurrentThread(
            interval=CONTROL_DT, target=self._Write, name="inspire_hand_write"
        )
        print("[INFO] Waiting for first hand state message...")
        timeout = time.time() + 5.0
        while not self.state_received and time.time() < timeout:
            time.sleep(0.05)
        if not self.state_received:
            print("[WARN] No state received from rt/inspire/state -- "
                  "starting from a neutral guess. Check the hand driver is running.")
        self.write_thread.Start()

    def _Write(self):
        max_step = RAMP_SPEED * CONTROL_DT
        cmd = MotorCmds_(cmds=[])
        for i in range(NUM_MOTORS):
            err = self.q_target[i] - self.q_cmd[i]
            step = max(-max_step, min(max_step, err))
            self.q_cmd[i] += step
            cmd.cmds.append(MotorCmd_(
                mode=1, q=self.q_cmd[i], dq=0.0, tau=0.0,
                kp=HAND_KP, kd=HAND_KD, reserve=[0, 0, 0],
            ))
        self.cmd_publisher.Write(cmd)

    def set_hand(self, indices, q):
        for i in indices:
            self.q_target[i] = q

    def open_hand(self, side="both"):
        if side in ("right", "both"):
            self.set_hand(RIGHT_HAND_IDX, OPEN_Q)
        if side in ("left", "both"):
            self.set_hand(LEFT_HAND_IDX, OPEN_Q)

    def close_hand(self, side="both"):
        if side in ("right", "both"):
            self.set_hand(RIGHT_HAND_IDX, CLOSE_Q)
        if side in ("left", "both"):
            self.set_hand(LEFT_HAND_IDX, CLOSE_Q)


HELP_TEXT = """
Commands:
  open           - open both hands
  close          - close both hands
  open left      - open left hand only
  open right     - open right hand only
  close left     - close left hand only
  close right    - close right hand only
  status         - print current commanded/measured q
  help           - show this message
  quit / exit    - stop and exit
"""


def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(1)

    print("WARNING: Ensure fingers are clear of pinch points before running this script.")
    input("Press Enter to continue...")

    hand = InspireHandController()
    hand.Init()
    hand.Start()

    print(HELP_TEXT)
    while True:
        try:
            cmd = input("hand> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd in ("quit", "exit"):
            break
        elif cmd == "open":
            hand.open_hand("both")
        elif cmd == "close":
            hand.close_hand("both")
        elif cmd == "open left":
            hand.open_hand("left")
        elif cmd == "open right":
            hand.open_hand("right")
        elif cmd == "close left":
            hand.close_hand("left")
        elif cmd == "close right":
            hand.close_hand("right")
        elif cmd == "status":
            for i in range(NUM_MOTORS):
                side = "R" if i in RIGHT_HAND_IDX else "L"
                name = JOINT_NAMES[i % 6]
                print(f"  [{side}] {name:15s} q_meas={hand.q_state[i]:+.3f} "
                      f"q_cmd={hand.q_cmd[i]:+.3f} q_target={hand.q_target[i]:+.3f}")
        elif cmd in ("help", ""):
            print(HELP_TEXT)
        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for options.")

    print("[INFO] Exiting.")


if __name__ == "__main__":
    main()
