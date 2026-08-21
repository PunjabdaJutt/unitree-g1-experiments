"""Drive Inspire hand finger joints through a sinusoidal open/close motion
over Modbus TCP. Reuses the connection/register logic from
tools/inspire_hand_modbus_control.py.

Two modes (--mode):
  sequential (default) - one joint at a time runs a full 0 -> amplitude -> 0
                          sweep over `period` seconds while every other
                          joint is held closed; then the next joint (in
                          --joints order) takes its turn, cycling forever.
                          This is what you actually see: finger 1 opens and
                          closes, then finger 2 opens and closes, etc.
  wave                  - every selected joint oscillates continuously and
                          simultaneously, each on its own sinusoid:
                              angle(t) = amplitude/2 * (1 - cos(2*pi*t/period + phase))
                          with --phase-offset staggering successive joints
                          to produce a traveling wave instead of lockstep.

We write one combined 6-register ANGLE_SET update per hand per tick (using
the -1/no-op sentinel where relevant), rather than one write per joint, to
keep the update loop fast enough to hold --rate.

Run:
  .venv/bin/python3 tools/sinusoidal_hands.py
  .venv/bin/python3 tools/sinusoidal_hands.py --amplitude 700 --period 3
  .venv/bin/python3 tools/sinusoidal_hands.py --mode wave --phase-offset 30
  .venv/bin/python3 tools/sinusoidal_hands.py --hands 192.168.123.210 --joints index thumb_bend
"""
import argparse
import math
import sys
import time

from inspire_hand_modbus_control import (
    InspireHand,
    JOINT_NAMES,
    NUM_DOF,
    NO_OP,
    OPEN_ANGLE,
    CLOSED_ANGLE,
)

# Fast follow so the hand's own ramping stays close behind the target
# instead of lagging noticeably behind a moving setpoint.
DEFAULT_SPEED = 100
DEFAULT_FORCE = 700
DEFAULT_AMPLITUDE = 20
DEFAULT_PERIOD = 90
DEFAULT_RATE = 20.0


def joint_angle(t: float, amplitude: float, period: float, phase_deg: float) -> int:
    phase = math.radians(phase_deg)
    value = amplitude / 2.0 * (1 - math.cos(2 * math.pi * t / period + phase))
    return int(round(min(max(value, CLOSED_ANGLE), OPEN_ANGLE)))


def sequential_angles(t: float, joint_indices: list, amplitude: float, period: float) -> list:
    """One joint sweeps 0 -> amplitude -> 0 at a time; the rest stay closed."""
    angles = [CLOSED_ANGLE] * NUM_DOF
    num_joints = len(joint_indices)
    active = int(t // period) % num_joints
    t_local = t % period
    angles[joint_indices[active]] = joint_angle(t_local, amplitude, period, 0.0)
    return angles


def wave_angles(t: float, joint_indices: list, amplitude: float, period: float, phase_offset: float) -> list:
    """Every selected joint oscillates continuously, phase-staggered."""
    angles = [NO_OP] * NUM_DOF
    for order, joint_idx in enumerate(joint_indices):
        angles[joint_idx] = joint_angle(t, amplitude, period, order * phase_offset)
    return angles


def _direction(frac: float) -> str:
    return "opening" if (frac % 1.0) < 0.5 else "closing"


def describe_sequential(t: float, joint_indices: list, angles: list, period: float) -> str:
    active = int(t // period) % len(joint_indices)
    direction = _direction((t % period) / period)
    parts = []
    for i, idx in enumerate(joint_indices):
        name = JOINT_NAMES[idx]
        if i == active:
            parts.append(f"{name}:{direction}({angles[idx]})")
        else:
            parts.append(f"{name}:closed")
    return " | ".join(parts)


def describe_wave(t: float, joint_indices: list, angles: list, period: float, phase_offset: float) -> str:
    parts = []
    for order, idx in enumerate(joint_indices):
        frac = t / period + (order * phase_offset) / 360.0
        parts.append(f"{JOINT_NAMES[idx]}:{_direction(frac)}({angles[idx]})")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hands", nargs="+", default=["192.168.123.210", "192.168.123.211"],
        help="Modbus TCP IPs of the hands to drive (default: both known hands)",
    )
    parser.add_argument(
        "--joints", nargs="+", default=JOINT_NAMES, choices=JOINT_NAMES,
        help="Which joints to move, in turn order for --mode sequential (default: all six)",
    )
    parser.add_argument(
        "--mode", choices=["sequential", "wave"], default="sequential",
        help="sequential: one joint opens+closes at a time (default). "
             "wave: all selected joints oscillate at once, phase-staggered.",
    )
    parser.add_argument(
        "--amplitude", type=int, default=DEFAULT_AMPLITUDE,
        help="Peak angle reached, 0-1000 (motion always starts/ends at 0)",
    )
    parser.add_argument("--period", type=float, default=DEFAULT_PERIOD,
                         help="Seconds for one joint's full open+close cycle")
    parser.add_argument(
        "--phase-offset", type=float, default=0.0,
        help="--mode wave only: degrees of phase stagger between successive --joints",
    )
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE,
                         help="Target-update rate in Hz")
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="0-1000, higher=faster tracking")
    parser.add_argument("--force", type=int, default=DEFAULT_FORCE, help="0-3000 grams")
    args = parser.parse_args()

    if not (CLOSED_ANGLE <= args.amplitude <= OPEN_ANGLE):
        parser.error(f"--amplitude must be within {CLOSED_ANGLE}-{OPEN_ANGLE}")
    if args.period <= 0:
        parser.error("--period must be > 0")
    if args.rate <= 0:
        parser.error("--rate must be > 0")

    joint_indices = [JOINT_NAMES.index(j) for j in args.joints]

    hands = [InspireHand(ip) for ip in args.hands]
    for hand in hands:
        if not hand.connect():
            print(f"FAILED to connect to {hand.ip}")
            sys.exit(1)
        hand.configure(args.speed, args.force)
        print(f"[INFO] connected to {hand.ip}, speed={args.speed} force={args.force}g")

    print(
        f"[INFO] mode={args.mode} joints={args.joints} amplitude={args.amplitude} "
        f"period={args.period}s phase_offset={args.phase_offset}deg rate={args.rate}Hz"
    )
    print("WARNING: Ensure fingers are clear of pinch points before running this script.")
    input("Press Enter to continue...")

    dt = 1.0 / args.rate
    start = time.monotonic()
    try:
        while True:
            t = time.monotonic() - start
            if args.mode == "sequential":
                angles = sequential_angles(t, joint_indices, args.amplitude, args.period)
                status = describe_sequential(t, joint_indices, angles, args.period)
            else:
                angles = wave_angles(t, joint_indices, args.amplitude, args.period, args.phase_offset)
                status = describe_wave(t, joint_indices, angles, args.period, args.phase_offset)
            for hand in hands:
                hand.set_angles(angles)
            print(f"\r[{t:7.1f}s] {status}   ", end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        for hand in hands:
            hand.close()
        print("[INFO] Exiting.")


if __name__ == "__main__":
    main()
