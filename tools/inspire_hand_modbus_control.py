"""Interactive open/close control for the G1's Inspire RH56DFTP hands over
Modbus TCP -- see tools/inspire_hand_modbus_read.py for the network/register
background (hands live on 192.168.123.210 / .211, not the factory-default
192.168.11.210, and confirmed healthy: ERROR=0, ~35-40C, all DOFs reachable).

Unlike the DDS-based scripts (low_level/inspire_hand_open_close.py,
ros2_ws .../inspire_hand_node.py), which ramp position themselves at 50Hz,
Modbus writes go straight to the hand's own firmware -- so we set a
conservative SPEED_SET and FORCE_SET once at startup, and the hand's own
controller ramps to each ANGLE_SET target and stops early if the force
threshold is hit. No software ramping needed.

Registers used (RH56DFTP manual section 2.6):
  1486  ANGLE_SET(m)   W, 6 shorts, -1 (no-op) or 0-1000 (0=closed, 1000=open)
  1498  FORCE_SET(m)   W, 6 shorts, 0-3000 g gripping-force threshold
  1522  SPEED_SET(m)   W, 6 shorts, 0-1000 (1000 = ~600ms full stroke)
  1546  ANGLE_ACT(m)   R, 6 shorts, current angle (same 0-1000 scale)
DOF order: (0) pinky, (1) ring, (2) middle, (3) index, (4) thumb_bend,
(5) thumb_rotation.

Caveat: hand .211's thumb_rotation force sensor read a large baseline
offset (-2009g) at rest during the health check, i.e. it likely needs
FORCE sensor calibration (register 1009, GESTURE_FORCE_CLB -- hand must be
open with fingers touching nothing during calibration). Until that's done,
force-threshold behavior on that specific joint may be unreliable; angle
control itself is unaffected.

Confirmed by observation against the physical robot: .210 is the anatomical
LEFT hand, .211 is RIGHT.

Run:
  .venv/bin/python3 tools/inspire_hand_modbus_control.py
  .venv/bin/python3 tools/inspire_hand_modbus_control.py --hands 192.168.123.210
"""
import argparse
import sys
import time

from pymodbus.client import ModbusTcpClient

PORT = 6000
DEVICE_ID = 1

JOINT_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]
NUM_DOF = 6

ANGLE_SET = 1486
FORCE_SET = 1498
SPEED_SET = 1522
ANGLE_ACT = 1546
FORCE_ACT = 1582
ERROR = 1606

# Manual specs FORCE_ACT as -4000..4000g. A reading outside that band is not
# a real grip force -- it's the sensor/ADC/wiring reporting garbage. Flag it
# loudly rather than just printing the number: FORCE_SET-based safety stops
# cannot be trusted on a joint reporting out-of-spec force.
FORCE_ACT_MAX_G = 4000

# ERROR is byte-packed (2 DOFs/register, see inspire_hand_modbus_read.py).
ERROR_FLAG_NAMES = ["locked-rotor", "overtemp", "overcurrent", "abnormal", "comms"]


def decode_error_bits(byte_val: int) -> list:
    return [name for i, name in enumerate(ERROR_FLAG_NAMES) if byte_val & (1 << i)]

OPEN_ANGLE = 1000
CLOSED_ANGLE = 0
NO_OP = -1  # ANGLE_SET sentinel: leave this joint's current target unchanged

# Conservative defaults -- speed=1000 is ~600ms full stroke, so 250 is a
# slow, easy-to-abort-by-watching move. Force threshold well under the
# 0-3000 range so the hand stops pushing long before anything like its
# rated 30N grip.
DEFAULT_SPEED = 250
DEFAULT_FORCE = 300

# Hand 1 (first --hands entry, i.e. .210 by default) pinky needs more than
# the 300g default to fully close -- found via inspire_hand_modbus_calibrate.py
# close-force tuning: stalls ~872 at 400g, fully closes at 500g. 550g leaves
# a little headroom above the measured minimum.
HAND1_PINKY_FORCE = 550


class InspireHand:
    def __init__(self, ip: str):
        self.ip = ip
        self.client = ModbusTcpClient(ip, port=PORT, timeout=3)
        self.force_config = [DEFAULT_FORCE] * NUM_DOF

    def connect(self) -> bool:
        return self.client.connect()

    def close(self) -> None:
        self.client.close()

    def configure(self, speed: int, force: int) -> None:
        self.client.write_registers(SPEED_SET, [speed] * NUM_DOF, device_id=DEVICE_ID)
        self.force_config = [force] * NUM_DOF
        self.client.write_registers(FORCE_SET, self.force_config, device_id=DEVICE_ID)

    def set_joint_force(self, joint_index: int, force: int) -> None:
        self.force_config[joint_index] = force
        self.client.write_registers(FORCE_SET, self.force_config, device_id=DEVICE_ID)

    def set_angles(self, angles: list) -> None:
        assert len(angles) == NUM_DOF
        registers = [a & 0xFFFF for a in angles]  # NO_OP (-1) -> 0xFFFF for the wire
        result = self.client.write_registers(ANGLE_SET, registers, device_id=DEVICE_ID)
        if result.isError():
            print(f"  [{self.ip}] write ERROR: {result}")

    def set_joint_angle(self, joint_index: int, angle: int) -> None:
        angles = [NO_OP] * NUM_DOF
        angles[joint_index] = angle
        self.set_angles(angles)

    def open_hand(self) -> None:
        self.set_angles([OPEN_ANGLE] * NUM_DOF)

    def close_hand(self) -> None:
        self.set_angles([CLOSED_ANGLE] * NUM_DOF)

    def read_angles(self):
        result = self.client.read_holding_registers(ANGLE_ACT, count=NUM_DOF, device_id=DEVICE_ID)
        if result.isError():
            return None
        return result.registers

    def read_forces(self):
        result = self.client.read_holding_registers(FORCE_ACT, count=NUM_DOF, device_id=DEVICE_ID)
        if result.isError():
            return None
        return [v - 0x10000 if v >= 0x8000 else v for v in result.registers]

    def read_errors(self):
        result = self.client.read_holding_registers(ERROR, count=3, device_id=DEVICE_ID)
        if result.isError():
            return None
        unpacked = []
        for v in result.registers:
            unpacked.append(v & 0xFF)
            unpacked.append((v >> 8) & 0xFF)
        return unpacked


HELP_TEXT = """
Commands:
  open                    - open all connected hands
  close                   - close all connected hands
  set <hand> <joint> <angle>
                          - set one finger's angle (0=closed .. 1000=open)
                            hand:  1, 2, ... (order of --hands), an IP or
                                   IP suffix (e.g. 210), or 'all'
                            joint: pinky, ring, middle, index, thumb_bend,
                                   thumb_rotation (or 0-5)
  diag <hand> <joint> <angle> [force]
                          - like 'set', but for ONE hand only, and streams
                            live angle/force/error for that joint until the
                            angle settles (or ctrl-c) -- use this to see why
                            a joint is stopping short of its target
                            optional [force] (0-3000g) overrides just that
                            joint's FORCE_SET before moving, and stays in
                            effect until changed again
  status                  - print current angle/force/error per hand/joint
  help                    - show this message
  quit / exit             - stop and exit
"""


def run_diag(hand: "InspireHand", joint_idx: int, target: int, force: int = None, timeout: float = 5.0) -> None:
    joint_name = JOINT_NAMES[joint_idx]
    if force is not None:
        hand.set_joint_force(joint_idx, force)
        print(f"  [{hand.ip}] {joint_name} FORCE_SET -> {force}g")
    print(f"  [{hand.ip}] driving {joint_name} -> {target} (ctrl-c to stop early)")
    hand.set_joint_angle(joint_idx, target)

    seen_flags = set()
    peak_force = 0
    sensor_fault_seen = False
    last_angle = None
    stable_count = 0
    start = time.time()
    try:
        while time.time() - start < timeout:
            angles = hand.read_angles()
            forces = hand.read_forces()
            errors = hand.read_errors()
            if angles is None or forces is None or errors is None:
                print("\n  read error, stopping")
                break

            angle = angles[joint_idx]
            force = forces[joint_idx]
            flags = decode_error_bits(errors[joint_idx])
            seen_flags.update(flags)
            peak_force = max(peak_force, abs(force))
            if abs(force) > FORCE_ACT_MAX_G:
                sensor_fault_seen = True

            flag_str = ",".join(flags) if flags else "-"
            fault_marker = "  !!! SENSOR FAULT !!!" if abs(force) > FORCE_ACT_MAX_G else ""
            print(f"\r  angle={angle:4d}  force={force:5d}g  error={flag_str:24s}{fault_marker}", end="", flush=True)

            if last_angle is not None and abs(angle - last_angle) <= 1:
                stable_count += 1
                if stable_count >= 4:
                    break
            else:
                stable_count = 0
            last_angle = angle
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    print()
    final_angles = hand.read_angles()
    final_angle = final_angles[joint_idx] if final_angles else last_angle
    flags_str = ", ".join(sorted(seen_flags)) if seen_flags else "none"
    print(
        f"  [{hand.ip}] {joint_name} settled at angle={final_angle} (target {target}), "
        f"peak |force|={peak_force}g, error flags seen: {flags_str}"
    )
    if sensor_fault_seen:
        print(
            f"  [{hand.ip}] {joint_name}: FORCE_ACT exceeded the sensor's documented "
            f"+/-{FORCE_ACT_MAX_G}g range during this move -- treat as a sensor/wiring "
            "fault, not a real force reading. Do not rely on FORCE_SET to protect this joint."
        )


def resolve_hands(token: str, hands: list) -> list:
    if token == "all":
        return list(hands)
    if token.isdigit():
        idx = int(token)
        if 1 <= idx <= len(hands):
            return [hands[idx - 1]]
        return []
    return [h for h in hands if h.ip == token or h.ip.endswith(token)]


def resolve_joint(token: str):
    if token.isdigit():
        idx = int(token)
        return idx if 0 <= idx < NUM_DOF else None
    if token in JOINT_NAMES:
        return JOINT_NAMES.index(token)
    matches = [i for i, name in enumerate(JOINT_NAMES) if name.startswith(token)]
    return matches[0] if len(matches) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hands", nargs="+", default=["192.168.123.210", "192.168.123.211"],
        help="Modbus TCP IPs of the hands to control (default: both known hands)",
    )
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="0-1000, higher=faster")
    parser.add_argument("--force", type=int, default=DEFAULT_FORCE, help="0-3000 grams")
    args = parser.parse_args()

    hands = [InspireHand(ip) for ip in args.hands]
    for i, hand in enumerate(hands):
        if not hand.connect():
            print(f"FAILED to connect to {hand.ip}")
            sys.exit(1)
        hand.configure(args.speed, args.force)
        print(f"[INFO] connected to {hand.ip}, speed={args.speed} force={args.force}g")
        if i == 0:
            hand.set_joint_force(JOINT_NAMES.index("pinky"), HAND1_PINKY_FORCE)
            print(f"[INFO] {hand.ip} pinky FORCE_SET -> {HAND1_PINKY_FORCE}g (default override)")

    print("WARNING: Ensure fingers are clear of pinch points before running this script.")
    input("Press Enter to continue...")

    print(HELP_TEXT)
    try:
        while True:
            try:
                cmd = input("hand> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if cmd in ("quit", "exit"):
                break
            elif cmd == "open":
                for hand in hands:
                    hand.open_hand()
            elif cmd == "close":
                for hand in hands:
                    hand.close_hand()
            elif cmd.startswith("set"):
                parts = cmd.split()
                if len(parts) != 4:
                    print("Usage: set <hand> <joint> <angle>")
                    continue
                _, hand_tok, joint_tok, angle_tok = parts

                matched_hands = resolve_hands(hand_tok, hands)
                if not matched_hands:
                    print(f"Unknown hand '{hand_tok}'. Use 1, 2, an IP (suffix), or 'all'.")
                    continue

                joint_idx = resolve_joint(joint_tok)
                if joint_idx is None:
                    print(f"Unknown joint '{joint_tok}'. Options: {', '.join(JOINT_NAMES)} (or 0-5).")
                    continue

                try:
                    angle = int(angle_tok)
                except ValueError:
                    print(f"Angle must be an integer, got '{angle_tok}'.")
                    continue
                if not (CLOSED_ANGLE <= angle <= OPEN_ANGLE):
                    print(f"Angle must be within {CLOSED_ANGLE}-{OPEN_ANGLE}.")
                    continue

                for hand in matched_hands:
                    hand.set_joint_angle(joint_idx, angle)
                    print(f"  [{hand.ip}] {JOINT_NAMES[joint_idx]} -> {angle}")
            elif cmd.startswith("diag"):
                parts = cmd.split()
                if len(parts) not in (4, 5):
                    print("Usage: diag <hand> <joint> <angle> [force]")
                    continue
                hand_tok, joint_tok, angle_tok = parts[1], parts[2], parts[3]
                force_tok = parts[4] if len(parts) == 5 else None

                matched_hands = resolve_hands(hand_tok, hands)
                if len(matched_hands) != 1:
                    print(f"'diag' needs exactly one hand, got {len(matched_hands)} matches for '{hand_tok}'.")
                    continue

                joint_idx = resolve_joint(joint_tok)
                if joint_idx is None:
                    print(f"Unknown joint '{joint_tok}'. Options: {', '.join(JOINT_NAMES)} (or 0-5).")
                    continue

                try:
                    angle = int(angle_tok)
                except ValueError:
                    print(f"Angle must be an integer, got '{angle_tok}'.")
                    continue
                if not (CLOSED_ANGLE <= angle <= OPEN_ANGLE):
                    print(f"Angle must be within {CLOSED_ANGLE}-{OPEN_ANGLE}.")
                    continue

                force = None
                if force_tok is not None:
                    try:
                        force = int(force_tok)
                    except ValueError:
                        print(f"Force must be an integer, got '{force_tok}'.")
                        continue
                    if not (0 <= force <= 3000):
                        print("Force must be within 0-3000.")
                        continue

                run_diag(matched_hands[0], joint_idx, angle, force)
            elif cmd == "status":
                for hand in hands:
                    angles = hand.read_angles()
                    forces = hand.read_forces()
                    errors = hand.read_errors()
                    if angles is None or forces is None or errors is None:
                        print(f"  [{hand.ip}] read error")
                        continue
                    print(f"  [{hand.ip}]")
                    for i, joint in enumerate(JOINT_NAMES):
                        flags = decode_error_bits(errors[i])
                        flag_str = f"  ERROR: {','.join(flags)}" if flags else ""
                        fault_str = ""
                        if abs(forces[i]) > FORCE_ACT_MAX_G:
                            fault_str = (
                                f"  !!! FORCE_ACT {forces[i]}g exceeds sensor spec "
                                f"(+/-{FORCE_ACT_MAX_G}g) -- SENSOR FAULT SUSPECTED, do not "
                                "trust force-based safety stop on this joint !!!"
                            )
                        print(f"      {joint:15s} angle={angles[i]:4d}  force={forces[i]:5d}g{flag_str}{fault_str}")
            elif cmd in ("help", ""):
                print(HELP_TEXT)
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for options.")
    finally:
        for hand in hands:
            hand.close()
        print("[INFO] Exiting.")


if __name__ == "__main__":
    main()
