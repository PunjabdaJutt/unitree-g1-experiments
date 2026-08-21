"""Force-sensor calibration for the G1's Inspire RH56DFTP hands over Modbus TCP.

Per the RH56DFTP manual (section 2.6.6): writing 1 to GESTURE_FORCE_CLB
(register 1009) zeroes ALL SIX force sensors' baseline AT ONCE, using
whatever each sensor currently reads. There is no per-DOF calibration
register -- "calibrating one finger" really means "get that finger into a
true, untouched rest position, then fire the one hand-wide trigger while
it's there." This is what fixed looks like when it's needed: hand .211's
thumb_rotation FORCE_ACT read a ~-2000g baseline at rest during the initial
health check (tools/inspire_hand_modbus_read.py), which is exactly the kind
of offset this clears.

Because every trigger re-zeros all 6 sensors simultaneously, we calibrate
incrementally instead of opening the whole hand at once:
  1. Close every joint (known, collision-free starting pose).
  2. Open the thumb (both thumb_bend and thumb_rotation) with the four
     fingers still closed and out of its way. Calibrate.
  3. One at a time -- pinky, ring, middle, index -- open that single finger
     (thumb and any already-done fingers stay put; not-yet-done fingers
     stay closed) and calibrate again.
Each trigger re-baselines every sensor from its *current* reading, so a
joint's calibration is only correct as of the most recent trigger fired
while it was genuinely open and touching nothing -- which is why already
-calibrated joints are left open (never re-closed) for the rest of the
sequence, and why fingers are brought out one at a time rather than
together: it keeps the set of "just moved" joints small, so a physical
snag shows up (as an ERROR flag or leftover FORCE_ACT offset) against a
single known joint instead of a moving target.

Once every sensor is zeroed, a separate problem shows up: FORCE_SET isn't
just a safety ceiling, it's what determines whether a joint can finish its
stroke at all. Investigation on hand .210 found pinky stalling around
angle~955 (should be 0) at the 300g default while ring closed fine at
~178g peak -- pinky genuinely needs more grip-force budget to complete its
travel, confirmed by raising FORCE_SET alone (400g -> stall ~872, 500g ->
fully closed). So after force-sensor calibration, we run a second phase:
for each joint, starting at 200g and stepping by 100g, command it closed
and poll ANGLE_ACT/ERROR at 10Hz (matching inspire_hand_modbus_control.py's
`diag`) until it actually reaches CLOSED_ANGLE. Whatever force that takes
is left as that joint's FORCE_SET; if it still won't close by the safety
cap, or trips an ERROR flag along the way, that's reported as a likely
mechanical issue rather than pushed through with more force.

Registers used (RH56DFTP manual section 2.6):
  1009  GESTURE_FORCE_CLB  W, 1 byte, write 1 to start calibration (whole hand)
  1486  ANGLE_SET(m)       W, 6 shorts, 0-1000 (0=closed, 1000=open)
  1498  FORCE_SET(m)       W, 6 shorts, 0-3000 g gripping-force threshold
  1522  SPEED_SET(m)       W, 6 shorts, 0-1000 (1000 = ~600ms full stroke)
  1546  ANGLE_ACT(m)       R, 6 shorts, current angle
  1582  FORCE_ACT(m)       R, 6 shorts, signed grams
  1606  ERROR(m)           R, 6 bytes (packed), bit0=locked-rotor bit1=overtemp
                            bit2=overcurrent bit3=abnormal bit4=comms

DOF order: (0) pinky, (1) ring, (2) middle, (3) index, (4) thumb_bend,
(5) thumb_rotation.

Run:
  .venv/bin/python3 tools/inspire_hand_modbus_calibrate.py 192.168.123.211
"""
import argparse
import time

from pymodbus.client import ModbusTcpClient

PORT = 6000
DEVICE_ID = 1
NUM_DOF = 6
JOINT_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]

GESTURE_FORCE_CLB = 1009
ANGLE_SET = 1486
FORCE_SET = 1498
SPEED_SET = 1522
ANGLE_ACT = 1546
FORCE_ACT = 1582
ERROR = 1606

OPEN_ANGLE = 1000
CLOSED_ANGLE = 0
ANGLE_TOLERANCE = 30  # ANGLE_ACT units; sensor noise/mechanical slop near full stroke
MOVE_TIMEOUT_SEC = 5.0
FORCE_OFFSET_FLAG_G = 50  # |FORCE_ACT| above this after calibration = still offset

# Conservative defaults, same rationale as inspire_hand_modbus_control.py:
# slow and easy to abort-by-watching, well under the hand's rated grip force.
DEFAULT_SPEED = 250
DEFAULT_FORCE = 300

# Close-force tuning (phase 2): per joint, ramp FORCE_SET from START in
# STEP increments, checking at 10Hz, until the joint actually reaches
# CLOSED_ANGLE or we hit MAX (safety ceiling well under the 3000g register
# max -- if a joint needs more than this to close, that's worth a physical
# look before just cranking the number higher).
FORCE_TUNE_START = 200
FORCE_TUNE_STEP = 100
FORCE_TUNE_MAX = 1500
FORCE_TUNE_POLL_SEC = 0.1  # 10Hz

# Calibration order: thumb (both DOF) first while the four fingers stay
# closed and clear, then one finger at a time so a snag is always
# attributable to the single joint that just moved.
THUMB_JOINTS = [4, 5]
FINGER_ORDER = [0, 1, 2, 3]

ERROR_FLAG_NAMES = ["locked-rotor", "overtemp", "overcurrent", "abnormal", "comms"]


def decode_error_bits(byte_val: int) -> list:
    return [name for i, name in enumerate(ERROR_FLAG_NAMES) if byte_val & (1 << i)]


def read_shorts(client, address, count, signed=False):
    result = client.read_holding_registers(address, count=count, device_id=DEVICE_ID)
    if result.isError():
        return None
    values = result.registers
    if signed:
        values = [v - 0x10000 if v >= 0x8000 else v for v in values]
    return values


def read_errors(client):
    result = client.read_holding_registers(ERROR, count=3, device_id=DEVICE_ID)
    if result.isError():
        return None
    unpacked = []
    for v in result.registers:
        unpacked.append(v & 0xFF)
        unpacked.append((v >> 8) & 0xFF)
    return unpacked


def write_angles(client, angles) -> None:
    assert len(angles) == NUM_DOF
    client.write_registers(ANGLE_SET, angles, device_id=DEVICE_ID)


def move_and_wait(client, ip: str, target: list, moved_indices: list) -> bool:
    """Write the full 6-DOF target and wait until the joints that just moved
    (moved_indices) reach it. Returns False (with a warning) on timeout or
    if an ERROR flag shows up on a moved joint -- either way, the caller
    decides whether it's still safe to calibrate."""
    write_angles(client, target)
    start = time.time()
    while time.time() - start < MOVE_TIMEOUT_SEC:
        angles = read_shorts(client, ANGLE_ACT, NUM_DOF)
        errors = read_errors(client)
        if angles is None or errors is None:
            print(f"  [{ip}] read error while waiting for move")
            return False

        flagged = [
            (JOINT_NAMES[i], decode_error_bits(errors[i]))
            for i in moved_indices
            if decode_error_bits(errors[i])
        ]
        if flagged:
            for name, flags in flagged:
                print(f"  [{ip}] WARNING: {name} reporting {','.join(flags)} -- possible collision/obstruction")
            return False

        if all(abs(angles[i] - target[i]) <= ANGLE_TOLERANCE for i in moved_indices):
            return True
        time.sleep(0.1)

    print(f"  [{ip}] WARNING: did not confirm target reached within {MOVE_TIMEOUT_SEC}s:")
    for i in moved_indices:
        if abs(angles[i] - target[i]) > ANGLE_TOLERANCE:
            print(f"      {JOINT_NAMES[i]:15s} actual={angles[i]}  target={target[i]}")
    return False


def trigger_calibration(client, ip: str, check_indices: list) -> None:
    client.write_register(GESTURE_FORCE_CLB, 1, device_id=DEVICE_ID)
    print("  calibration triggered, waiting 2s...")
    time.sleep(2.0)

    forces = read_shorts(client, FORCE_ACT, NUM_DOF, signed=True)
    if forces is None:
        print(f"  [{ip}] read error after calibration")
        return
    print("  FORCE_ACT for joint(s) just calibrated (should be near 0):")
    for i in check_indices:
        val = forces[i]
        flag = "  <-- still offset, non-trivial" if abs(val) > FORCE_OFFSET_FLAG_G else ""
        print(f"      {JOINT_NAMES[i]:15s} {val}{flag}")


def calibrate_step(client, ip: str, target: list, indices: list, label: str) -> bool:
    print(f"  opening {label}...")
    ok = move_and_wait(client, ip, target, indices)
    if not ok:
        print(f"  [{ip}] aborting -- {label} did not reach open cleanly, check the hand before retrying")
        return False

    print(f"  {label} open. Confirm it is touching nothing (no object, no other digit).")
    input("  Press Enter to trigger calibration...")
    trigger_calibration(client, ip, indices)
    return True


def tune_close_force(
    client, ip: str, joint_idx: int, force_config: list, target: list,
    start_force: int, step: int, max_force: int,
):
    """Ramp this one joint's FORCE_SET up from start_force until CLOSED_ANGLE
    is actually reached, polling at 10Hz (move_and_wait's 0.1s loop). Leaves
    force_config[joint_idx] at whatever level worked (or max_force if it
    never did) and target[joint_idx] at CLOSED_ANGLE. Returns the working
    force, or None if it never fully closed."""
    name = JOINT_NAMES[joint_idx]
    force = start_force
    target[joint_idx] = CLOSED_ANGLE
    while force <= max_force:
        force_config[joint_idx] = force
        client.write_registers(FORCE_SET, force_config, device_id=DEVICE_ID)
        print(f"    {name} @ {force}g...")
        reached = move_and_wait(client, ip, target, [joint_idx])
        if reached:
            print(f"    {name}: closed fully at {force}g")
            return force

        errors = read_errors(client)
        if errors and decode_error_bits(errors[joint_idx]):
            print(f"    {name}: error flag at {force}g -- stopping (likely mechanical, not just force)")
            return None

        angles = read_shorts(client, ANGLE_ACT, NUM_DOF)
        angle_now = angles[joint_idx] if angles else "?"
        print(f"    {name}: stalled at angle={angle_now} with {force}g, trying more force")
        force += step

    print(f"    {name}: still not fully closed at max test force {max_force}g")
    return None


def calibrate(
    ip: str, speed: int, force: int,
    tune_start: int = FORCE_TUNE_START, tune_step: int = FORCE_TUNE_STEP, tune_max: int = FORCE_TUNE_MAX,
) -> None:
    print(f"=== {ip} ===")
    client = ModbusTcpClient(ip, port=PORT, timeout=3)
    if not client.connect():
        print("  FAILED to connect")
        return
    try:
        client.write_registers(SPEED_SET, [speed] * NUM_DOF, device_id=DEVICE_ID)
        client.write_registers(FORCE_SET, [force] * NUM_DOF, device_id=DEVICE_ID)

        target = [CLOSED_ANGLE] * NUM_DOF
        print("  closing hand to a known starting pose...")
        if not move_and_wait(client, ip, target, list(range(NUM_DOF))):
            print(
                f"  [{ip}] one or more joints did not actually reach closed -- they will NOT be "
                "clear of the thumb/other fingers during the steps below."
            )
            if input("  Continue anyway? [y/N] ").strip().lower() != "y":
                return

        thumb_target = [OPEN_ANGLE if i in THUMB_JOINTS else CLOSED_ANGLE for i in range(NUM_DOF)]
        if not calibrate_step(client, ip, thumb_target, THUMB_JOINTS, "thumb (bend + rotation)"):
            return
        for i in THUMB_JOINTS:
            target[i] = OPEN_ANGLE

        for finger_idx in FINGER_ORDER:
            step_target = list(target)
            step_target[finger_idx] = OPEN_ANGLE
            if not calibrate_step(client, ip, step_target, [finger_idx], JOINT_NAMES[finger_idx]):
                return
            target[finger_idx] = OPEN_ANGLE

        print("  force-sensor calibration done. Sensors are zeroed, so it's now safe to close")
        print("  joints without corrupting an earlier calibration.")
        print(
            f"  Tuning per-joint close force: {tune_start}g, +{tune_step}g steps, "
            f"up to {tune_max}g, checked at 10Hz."
        )
        input("  Press Enter to start (ctrl-c to skip/abort at any point)...")

        force_config = [force] * NUM_DOF
        tuned_forces = {}
        try:
            for joint_idx in range(NUM_DOF):
                print(f"  -- {JOINT_NAMES[joint_idx]} --")
                tuned_forces[JOINT_NAMES[joint_idx]] = tune_close_force(
                    client, ip, joint_idx, force_config, target, tune_start, tune_step, tune_max,
                )
                target[joint_idx] = OPEN_ANGLE
                move_and_wait(client, ip, target, [joint_idx])
        except KeyboardInterrupt:
            print(f"  [{ip}] tuning interrupted, re-opening hand...")
            write_angles(client, [OPEN_ANGLE] * NUM_DOF)

        print("  close-force tuning results:")
        for name, tuned in tuned_forces.items():
            status = f"{tuned}g" if tuned is not None else "FAILED to fully close -- check for a mechanical issue"
            print(f"      {name:15s} {status}")

        print("  all joints open. Final FORCE_ACT:")
        forces = read_shorts(client, FORCE_ACT, NUM_DOF, signed=True)
        for joint, val in zip(JOINT_NAMES, forces or []):
            flag = "  <-- still offset, non-trivial" if abs(val) > FORCE_OFFSET_FLAG_G else ""
            print(f"      {joint:15s} {val}{flag}")
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hands", nargs="*", default=["192.168.123.210", "192.168.123.211"],
        help="Modbus TCP IPs of the hands to calibrate",
    )
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="0-1000, higher=faster")
    parser.add_argument("--force", type=int, default=DEFAULT_FORCE, help="0-3000 grams (used until tuning starts)")
    parser.add_argument("--tune-start-force", type=int, default=FORCE_TUNE_START, help="close-force tuning start, grams")
    parser.add_argument("--tune-force-step", type=int, default=FORCE_TUNE_STEP, help="close-force tuning step, grams")
    parser.add_argument("--tune-max-force", type=int, default=FORCE_TUNE_MAX, help="close-force tuning safety cap, grams")
    args = parser.parse_args()

    for ip in args.hands:
        calibrate(ip, args.speed, args.force, args.tune_start_force, args.tune_force_step, args.tune_max_force)
        print()


if __name__ == "__main__":
    main()
