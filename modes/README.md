# Robot modes

Scripts to check and switch the G1's control mode. Run them from the repo `.venv`, not a ROS2-sourced shell.

| Script | What it does |
|---|---|
| `g1_mode_switcher.py` | Switch modes, from an interactive menu or one-shot. |
| `g1_check_mode.py` | Print the current FSM id and who owns high-level control (read-only). |
| `out_of_dev.py` | Leave development mode by handing control back to `ai`. |

```bash
python3 modes/g1_mode_switcher.py eth0                    # interactive menu
python3 modes/g1_mode_switcher.py --iface eth0 --mode 3   # one-shot
python3 modes/g1_check_mode.py eth0
```

## `g1_mode_switcher.py` options

| Key | Mode |
|---|---|
| 0 | Zero torque (fsm 0, debug) |
| 1 | Damp (fsm 1) |
| 2 | Sit (fsm 3) |
| 3 | Locked standing (damp, then fsm 4). This is the normal state for the `rt/arm_sdk` scripts. |
| 4 | Enter development mode (release high-level control) |
| 5 | Exit development mode (restore mode `ai`) |
| 6 | Walking (fsm 501) |
| w | Watch fsm_id, fsm_mode and owner live (read-only) |

## Safety

The hardware arm scripts expect the robot in locked standing (mode 3) before they start.

**Never switch modes while an arm script is running.** A stale `rt/arm_sdk` command can fire the moment the mode changes.
