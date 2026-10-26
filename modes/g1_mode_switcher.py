import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

# NOTE on why this file changed:
# LocoClient's convenience wrappers (Damp(), Squat2StandUp(), Lie2StandUp(), ...)
# call self.SetFsmId(id) but never `return` its result, so the RPC error code
# was always discarded -- every mode switch here was flying blind. We now call
# SetFsmId()/SelectMode() directly and print the code (0 = accepted by the
# robot, non-zero = rejected) so failures are visible instead of silent.
#
# Root cause of "only damp/sit/zero-torque/dev mode work":
# per Unitree's G1 sport-services docs, FSM ids split into two families:
#   - "No Balance Control" ids (0 ZeroTorque, 1 Damp, 2 Squat, 3 Sit, 4 Lock
#     Standing) are accepted unconditionally.
#   - "Balance Control Active" ids (706 Squat2StandUp, 702 Lie2StandUp, 500/501
#     walk) are ONLY accepted while the robot's fsm_mode is 0 (static). While
#     fsm_mode is 1 (dynamic/settling), "switching to most modes is not
#     allowed". The old code did SetFsmId(1) then a blind time.sleep(0.5)
#     before firing 706/702 -- not a real check, so on a robot that took
#     longer to settle those transitions were silently rejected.
# We now poll GetFsmMode() (API 7002, added to LocoClient in
# g1_loco_client.py) and wait for static (0) before requesting a
# balance-control transition. Also fixed "locked standing": that is FSM id 4
# directly (a real, unconditional No-Balance-Control state), not
# Damp->706->SetFsmId(500) -- 500 is "walk motion", unrelated to locking.
def describe(label, code):
    status = "ok" if code == 0 else f"REJECTED (code={code})"
    print(f"  -> {label}: {status}")

def to_locked_stand(c):
    # FSM 4 ("Lock Standing") is its own No-Balance-Control state, accepted
    # unconditionally like damp/sit -- no need to route through 706/500.
    code = c.SetFsmId(1)
    describe("Damp (fsm 1)", code)
    time.sleep(0.5)
    code = c.SetFsmId(4)
    describe("Lock Standing (fsm 4)", code)

def to_walk(c):
    # FSM 501 ("walk") is a Balance-Control-Active state -- per the note at
    # the top of this file, those are only accepted while fsm_mode is 0
    # (static); while fsm_mode is 1 (dynamic/settling), the request is
    # silently rejected. Poll GetFsmMode() (API 7002) instead of a blind
    # sleep before firing SetFsmId(501), same fix as the 706/702 transitions.
    for _ in range(10):
        code, fsm_mode = c.GetFsmMode()
        if code == 0 and fsm_mode == 0:
            break
        print(f"  -> waiting for fsm_mode=static (0)... current fsm_mode={fsm_mode} (code={code})")
        time.sleep(0.5)
    else:
        print("  -> fsm_mode never reached static (0) after 5s -- requesting fsm 501 anyway, "
              "expect it to be rejected")
    code = c.SetFsmId(501)
    describe("Walking (fsm 501)", code)

def enter_dev_mode(msc):
    # Development mode = high-level control released, so low-level SDK
    # control (custom DDS commands) is available. Keep releasing until
    # CheckMode reports no owner.
    code, result = msc.CheckMode()
    describe("CheckMode", code)
    while result and result.get("name"):
        code, _ = msc.ReleaseMode()
        describe("ReleaseMode", code)
        time.sleep(1)
        code, result = msc.CheckMode()

def exit_dev_mode(msc):
    # Re-select this unit's actual high-level control mode to hand control
    # back from the low-level/development path. "normal" is NOT valid on
    # this robot -- SelectMode("normal") is rejected with code=7004 (unknown
    # mode). CheckMode always reported the owner as "ai" before dev mode was
    # ever entered here, matching out_of_dev.py / motion_switcher_example.py,
    # whose default (with "normal"/"advanced"/"ai-w" commented out as
    # alternatives for *other* hardware/firmware builds) is "ai". Use that.
    code, _ = msc.SelectMode("ai")
    describe("SelectMode('ai')", code)

    # SelectMode can be accepted (code 0) without the owner flipping over
    # immediately -- poll CheckMode for a few seconds so we can tell apart
    # "server rejected the call" (non-zero code above) from "server accepted
    # it but the mode change hasn't propagated / didn't actually take".
    for _ in range(5):
        time.sleep(1)
        code, result = msc.CheckMode()
        owner = result.get("name") if result else None
        if owner:
            print(f"  -> now owned by: {owner}")
            return True
    print("  -> still no owner after 5s: SelectMode was accepted but the robot "
          "never reclaimed high-level control. This points at a firmware/"
          "hardware precondition (e.g. robot must be damped/stationary, or "
          "the physical remote's mode button (L2+A / L2+R2) is required) "
          "rather than a bug in this script -- the RPC layer isn't "
          "reporting an error code.")
    return False

MODES = {
    "0": ("zero torque (debug)", lambda c, msc: describe("ZeroTorque (fsm 0)", c.SetFsmId(0))),
    "1": ("damp", lambda c, msc: describe("Damp (fsm 1)", c.SetFsmId(1))),
    "2": ("sit", lambda c, msc: describe("Sit (fsm 3)", c.SetFsmId(3))),
    "3": ("locked standing (damp -> fsm 4)", lambda c, msc: to_locked_stand(c)),
    "4": ("enter development mode (release high-level control)", lambda c, msc: enter_dev_mode(msc)),
    "5": ("exit development mode (restore high-level control, mode 'ai')", lambda c, msc: exit_dev_mode(msc)),
    "6": ("walking (fsm 501)", lambda c, msc: to_walk(c)),
}

def print_menu():
    print("\nAvailable modes:")
    for key, (name, _) in MODES.items():
        print(f"  {key}: {name}")
    print("  w: watch fsm_id/fsm_mode/owner live, read-only (no commands sent)")
    print("  q: quit")

def run_one_shot(iface, choice):
    """Non-interactive entry point: perform exactly one mode switch and exit
    with 0 on success, 1 on failure. Used by ROS2/other callers via
    subprocess instead of importing unitree_sdk2py directly in-process --
    see the ROS2-sourced-shell libddsc ABI conflict noted in the mode
    switcher node that drives this."""
    if choice not in MODES:
        print(f"invalid mode {choice!r}; valid: {', '.join(MODES)}")
        return 1

    ChannelFactoryInitialize(0, iface)

    client = LocoClient()
    client.SetTimeout(10.0)
    client.Init()

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    name, action = MODES[choice]

    code, mode_result = msc.CheckMode()
    owner = mode_result.get("name") if mode_result else None
    if not owner and choice != "5":
        print("robot is in development mode -- automatically exiting first...")
        if not exit_dev_mode(msc):
            print("could not reclaim high-level control; aborting")
            return 1

    print(f"switching to: {name}")
    action(client, msc)
    time.sleep(1)

    code, fsm_id = client.GetFsmId()
    code, mode_result = msc.CheckMode()
    owner = mode_result.get("name") if mode_result else None
    print(f"new FSM id: {fsm_id}, high-level control owner: {owner if owner else '(none / development mode)'}")
    return 0

if __name__ == '__main__':
    if "--mode" in sys.argv:
        iface = sys.argv[sys.argv.index("--iface") + 1]
        choice = sys.argv[sys.argv.index("--mode") + 1]
        sys.exit(run_one_shot(iface, choice))

    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        print(f"       python3 {sys.argv[0]} --iface networkInterface --mode N  (non-interactive)")
        sys.exit(-1)

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, sys.argv[1])

    client = LocoClient()
    client.SetTimeout(10.0)
    client.Init()

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    while True:
        code, fsm_id = client.GetFsmId()
        code, mode_result = msc.CheckMode()
        owner = mode_result.get("name") if mode_result else None
        print(f"\ncurrent FSM id: {fsm_id}, high-level control owner: {owner if owner else '(none / development mode)'}")
        print_menu()

        choice = input("select mode: ").strip()
        if choice == "q":
            break
        if choice == "w":
            print("watching fsm_id/fsm_mode/owner every 0.5s -- read-only, no "
                  "commands sent, safe to run while walking the robot with "
                  "the physical remote. Ctrl+C to stop and return to menu.")
            try:
                while True:
                    _, fid = client.GetFsmId()
                    _, fmode = client.GetFsmMode()
                    _, mode_result = msc.CheckMode()
                    owner = mode_result.get("name") if mode_result else None
                    print(f"  fsm_id={fid} fsm_mode={fmode} owner={owner if owner else '(none)'}")
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print()
            continue
        if choice not in MODES:
            print("invalid choice")
            continue

        name, action = MODES[choice]

        # Seamlessly hop out of development mode for the user instead of
        # just refusing -- any FSM-based mode implicitly requires high-level
        # control, so reclaim it first via the same path as choice "5".
        code, mode_result = msc.CheckMode()
        owner = mode_result.get("name") if mode_result else None
        if not owner and choice != "5":
            print("robot is in development mode (high-level control released) "
                  "-- automatically exiting development mode first...")
            if not exit_dev_mode(msc):
                print("could not reclaim high-level control; aborting this "
                      "mode switch. Check the physical remote (L2+A / "
                      "L2+R2) and try again.")
                continue

        print(f"switching to: {name}")
        action(client, msc)
        time.sleep(1)

        code, fsm_id = client.GetFsmId()
        code, mode_result = msc.CheckMode()
        owner = mode_result.get("name") if mode_result else None
        print(f"new FSM id: {fsm_id}, high-level control owner: {owner if owner else '(none / development mode)'}")
