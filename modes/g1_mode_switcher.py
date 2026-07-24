import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

def to_locked_stand(c):
    # Mirrors the documented stand sequence from
    # example/g1/high_level/g1_loco_client_example.py (Damp -> sleep 0.5 ->
    # Squat2StandUp). The extra Start()/FSM 500 step below is NOT exercised
    # by any SDK example -- unverified, experimental.
    c.Damp()
    time.sleep(0.5)
    c.Squat2StandUp()
    time.sleep(2)
    c.Start()

def enter_dev_mode(msc):
    # Development mode = high-level control released, so low-level SDK
    # control (custom DDS commands) is available. Keep releasing until
    # CheckMode reports no owner.
    code, result = msc.CheckMode()
    while result and result.get("name"):
        msc.ReleaseMode()
        time.sleep(1)
        code, result = msc.CheckMode()

def exit_dev_mode(msc):
    # Re-select the normal high-level control mode to hand control back
    # from the low-level/development path.
    msc.SelectMode("normal")

MODES = {
    "0": ("zero torque (debug)", lambda c, msc: c.ZeroTorque()),
    "1": ("damp", lambda c, msc: c.Damp()),
    "2": ("sit", lambda c, msc: c.Sit()),
    "3": ("squat2standup (stand) [recommended]", lambda c, msc: c.Squat2StandUp()),
    "4": ("lie2standup (stand)", lambda c, msc: c.Lie2StandUp()),
    "5": ("locked standing (start, fsm 500) [experimental/unverified]", lambda c, msc: to_locked_stand(c)),
    "6": ("enter development mode (release high-level control)", lambda c, msc: enter_dev_mode(msc)),
    "7": ("exit development mode (restore normal high-level control)", lambda c, msc: exit_dev_mode(msc)),
}

def print_menu():
    print("\nAvailable modes:")
    for key, (name, _) in MODES.items():
        print(f"  {key}: {name}")
    print("  q: quit")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
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
        if choice not in MODES:
            print("invalid choice")
            continue

        name, action = MODES[choice]

        code, mode_result = msc.CheckMode()
        owner = mode_result.get("name") if mode_result else None
        if not owner and choice != "7":
            print("robot is in development mode (high-level control released) -- "
                  "this command will be ignored by the robot. Select 7 to exit "
                  "development mode first.")
            continue

        print(f"switching to: {name}")
        action(client, msc)
        time.sleep(1)

        code, fsm_id = client.GetFsmId()
        code, mode_result = msc.CheckMode()
        owner = mode_result.get("name") if mode_result else None
        print(f"new FSM id: {fsm_id}, high-level control owner: {owner if owner else '(none / development mode)'}")
