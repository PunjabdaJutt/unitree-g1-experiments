import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

FSM_NAMES = {
    0: "zero torque (debug)",
    1: "damp",
    2: "squat",
    3: "sit",
    4: "ready mode",
    706: "standing (squat2standup)",
    702: "standing (lie2standup)",
}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    ChannelFactoryInitialize(0, sys.argv[1])

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    code, result = msc.CheckMode()
    owner = result.get("name") if result else None
    print(f"high-level control owner: {owner if owner else '(none / released, low-level control available)'}")

    loco = LocoClient()
    loco.SetTimeout(5.0)
    loco.Init()
    code, fsm_id = loco.GetFsmId()
    print(f"FSM id: {fsm_id} -> {FSM_NAMES.get(fsm_id, 'unknown')}")
