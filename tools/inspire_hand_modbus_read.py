"""Read-only health check for the G1's Inspire RH56DFTP hands over Modbus TCP.

Confirms we can actually talk to the hands and that the register map from
Inspire's official RH56DFTP manual (register descriptions, section 2.6)
lines up with real values, before anything writes/moves a finger.

Network background (see project memory / this session's investigation):
  - The hands are wired via an RJ45-shaped connector at the back of the
    robot's head, but electrically it's RS485/CAN, not real Ethernet -- the
    factory default Modbus TCP endpoint (192.168.11.210:6000) is
    unreachable on this robot. Whoever set this G1 up instead put the two
    hands directly on the robot's own 192.168.123.0/24 network, keeping the
    factory host octet (.210) for one hand and .211 for the other.
  - Port 6000 is open and SSH (22) is refused on both -- consistent with a
    small embedded Modbus TCP server, not a general-purpose Linux box.
  - Run this directly from a machine on 192.168.123.0/24 (e.g. this dev
    box's eno1) -- no need to go through PC1/PC2 for the hands themselves.

Register addresses (from the RH56DFTP manual, section 2.6 -- same numbering
used for RS485/CAN and Modbus TCP):
  1000        HAND_ID          1 register, R/W
  1546        ANGLE_ACT(m)     6 registers, R  (0-1000; 1000 = fully open,
                                                 0 = fully closed/bent --
                                                 confirmed by the manual's
                                                 own FORCE_SET worked example)
  1582        FORCE_ACT(m)     6 registers, R  (-4000..4000 g)
  1606        ERROR(m)         6 registers, R  (nonzero = actuator error)
  1618        TEMP(m)          6 registers, R  (actuator temperature)

DOF order for all (m) register groups: (0) little finger, (1) ring,
(2) middle, (3) index, (4) thumb bend, (5) thumb rotation.

Run:
  .venv/bin/python3 tools/inspire_hand_modbus_read.py 192.168.123.210 192.168.123.211
"""
import sys

from pymodbus.client import ModbusTcpClient

PORT = 6000
DEVICE_ID = 1

JOINT_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]

# ANGLE_ACT/FORCE_ACT are "short" (2 bytes/DOF) -- one register per DOF.
# ERROR/TEMP are "byte" (1 byte/DOF) -- Modbus registers are always 16-bit,
# so two adjacent DOFs pack into one register (low byte = lower address/
# even DOF index, high byte = higher address/odd DOF index -- confirmed by
# unpacking a captured raw TEMP register: 8736 -> 32, 34 (both plausible
# actuator temps), not a single ~8700 degree reading.
REGISTERS = [
    ("ANGLE_ACT", 1546, 6, "0-1000, 1000=open/0=closed", False),
    ("FORCE_ACT", 1582, 6, "grams, signed", False),
    ("ERROR", 1606, 3, "bitmask: bit0=locked-rotor bit1=overtemp bit2=overcurrent bit3=abnormal bit4=comms", True),
    ("TEMP", 1618, 3, "deg C", True),
]


def read_group(
    client: ModbusTcpClient, name: str, address: int, count: int, note: str, packed_bytes: bool
) -> None:
    result = client.read_holding_registers(address, count=count, device_id=DEVICE_ID)
    if result.isError():
        print(f"  {name:10s} @ {address}: ERROR reading ({result})")
        return
    values = result.registers
    if packed_bytes:
        # unpack each 16-bit register into its two 1-byte DOF values
        unpacked = []
        for v in values:
            unpacked.append(v & 0xFF)
            unpacked.append((v >> 8) & 0xFF)
        values = unpacked
    else:
        # FORCE_ACT is signed in the manual (-4000..4000); pymodbus returns
        # raw unsigned 16-bit words.
        values = [v - 0x10000 if v >= 0x8000 else v for v in values]
    print(f"  {name:10s} @ {address} ({note}):")
    for joint, val in zip(JOINT_NAMES, values):
        print(f"      {joint:15s} {val}")


def check_hand(ip: str) -> None:
    print(f"=== {ip}:{PORT} ===")
    client = ModbusTcpClient(ip, port=PORT, timeout=3)
    if not client.connect():
        print("  FAILED to connect")
        return
    try:
        hand_id = client.read_holding_registers(1000, count=1, device_id=DEVICE_ID)
        if hand_id.isError():
            print(f"  HAND_ID    @ 1000: ERROR reading ({hand_id})")
        else:
            print(f"  HAND_ID    @ 1000: {hand_id.registers[0]}")
        for name, address, count, note, packed in REGISTERS:
            read_group(client, name, address, count, note, packed)
    finally:
        client.close()


def main() -> None:
    ips = sys.argv[1:] or ["192.168.123.210", "192.168.123.211"]
    for ip in ips:
        check_hand(ip)
        print()


if __name__ == "__main__":
    main()
