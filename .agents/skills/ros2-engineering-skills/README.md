# ros2-engineering-skills

Agent skill for production-grade ROS 2 development — from first workspace to fleet deployment.

`SKILL.md` and `references/` are platform-neutral knowledge documents that work with [Claude Code](https://code.claude.com), [Codex](https://developers.openai.com/codex), [Cursor](https://cursor.sh), [Gemini CLI](https://github.com/google-gemini/gemini-cli), and any agent supporting the [Agent Skills](https://agentskills.io) standard. The hook wiring (automatic validation on tool use / stop) and the opt-in `.skill-runs.log` are **Claude Code-specific**; on other platforms the `scripts/` validators can be run manually (see Installation).

## Before / After

> **Prompt:** *"My ROS 2 subscriber isn't receiving any sensor messages. Help me fix it."*

<table>
<tr>
<th width="50%">Without this skill</th>
<th width="50%">With this skill loaded</th>
</tr>
<tr>
<td>

```python
# "Check your topic name and make sure
#  the publisher is running"
sub = node.create_subscription(
    Image, '/camera/image_raw',
    callback, 10)
```

Generic advice — misses the real issue.

</td>
<td>

```python
# Diagnoses QoS mismatch (DDS RxO rules):
# Sensor drivers default to BEST_EFFORT,
# but queue depth "10" implies RELIABLE.
from rclpy.qos import qos_profile_sensor_data

sub = node.create_subscription(
    Image, '/camera/image_raw',
    callback, qos_profile_sensor_data)
#          ^^^^^^^^^^^^^^^^^^^^^^
# BEST_EFFORT + VOLATILE — matches sensor pub
```

Identifies the root cause, explains DDS Request-vs-Offered semantics, and applies the correct `qos_profile_sensor_data` preset.

</td>
</tr>
</table>

> **Prompt:** *"Create a C++ driver package for my LiDAR sensor."*

<table>
<tr>
<th width="50%">Without this skill</th>
<th width="50%">With this skill loaded</th>
</tr>
<tr>
<td>

```text
my_lidar_driver/
├── src/main.cpp          # rclcpp::Node
├── CMakeLists.txt
└── package.xml
```

Bare `Node`, no lifecycle, no tests, no launch file, `ament_target_dependencies` (deprecated in Kilted).

</td>
<td>

```text
my_lidar_driver/
├── include/my_lidar_driver/
│   └── my_lidar_driver_node.hpp  # LifecycleNode
├── src/
│   ├── my_lidar_driver_node.cpp  # on_configure/activate/…
│   └── main.cpp
├── launch/bringup.launch.py      # auto-configure + activate
├── config/params.yaml
├── test/test_my_lidar_driver.cpp  # gtest
├── CMakeLists.txt                 # target_link_libraries
└── package.xml                    # format 3, Apache-2.0
```

LifecycleNode with managed transitions, launch file with auto-activation, gtest scaffolding, modern CMake (`target_link_libraries` over deprecated `ament_target_dependencies`), and distro-aware defaults.

</td>
</tr>
</table>

## What this is

A `SKILL.md`-based knowledge module that gives AI coding agents deep ROS 2 engineering expertise. Instead of a shallow cheat sheet, it provides:

- **Decision frameworks** — when to use rclcpp vs rclpy, which QoS profile, lifecycle vs plain node
- **Progressive disclosure** — compact routing in `SKILL.md`, detailed patterns in `references/`
- **Full spectrum** — workspace setup through real-time tuning, Nav2, MoveIt 2, ros2_control, DDS configuration, cross-compilation, and CI/CD
- **Distro-aware** — explicit Humble / Jazzy / Kilted / Rolling differences with migration paths
- **Anti-pattern documentation** — what breaks in production and why

## How it differs from existing ROS 2 skills

| Aspect | Typical ROS 2 skill | This project |
|---|---|---|
| Depth | Basic QoS + lifecycle intro | DDS vendor tuning, custom executors, intra-process zero-copy, type adapters |
| Scope | Single SKILL.md file | 23 reference files via progressive disclosure |
| Hardware | Mentioned in passing | ros2_control hardware interface patterns, serial/CAN/EtherCAT, controller chaining |
| Real-time | Not covered | PREEMPT_RT, realtime_tools, memory allocation, callback group strategies |
| Simulation | Mentioned in passing | Gazebo version matrix, gz_ros2_control, Isaac Sim, sim-to-real |
| Security | Not covered | SROS2, DDS security plugins, certificate management, supply chain |
| Embedded | Not covered | micro-ROS, rclc, XRCE-DDS, ESP32/STM32/RP2040 |
| Multi-robot | Not covered | Open-RMF, fleet adapters, DDS discovery at scale, NTP/PTP sync |
| Testing | "Use pytest" | launch_testing, gtest, industrial_ci, simulation-in-the-loop CI |
| Deployment | Not covered | Docker multi-stage, cross-compile, fleet OTA, Zenoh routing |

## Installation

### Claude Code

```bash
# From plugin marketplace (terminal)
claude plugin marketplace add dbwls99706/ros2-engineering-skills
claude plugin install ros2-engineering@ros2-engineering-skills

# Or use slash commands (inside Claude Code)
/plugin marketplace add dbwls99706/ros2-engineering-skills
/plugin install ros2-engineering@ros2-engineering-skills

# Or clone directly
git clone https://github.com/dbwls99706/ros2-engineering-skills.git ~/.claude/skills/ros2-engineering-skills
```

### Codex / Gemini CLI / OpenCode

```bash
git clone https://github.com/dbwls99706/ros2-engineering-skills.git ~/.agents/skills/ros2-engineering-skills
```

On these platforms the knowledge documents load, but the Claude Code hook
block in `SKILL.md` frontmatter is ignored: nothing runs automatically and
`.skill-runs.log` is never created. Run the validators manually when needed
(Python 3.10+; the YAML lint additionally needs PyYAML):

```bash
# Manual workspace validation (launch files, package.xml, Nav2 YAML lint)
SKILL_WORKSPACE=/path/to/your/workspace python3 scripts/skill_stop_hook.py

# Manual anti-pattern / dangerous-command checks.
# The quoted command is inspected only; it is never executed.
python3 scripts/skill_validate_hook.py --file src/my_node.py
python3 scripts/skill_validate_hook.py --command 'rm -rf /'        # status: fail, exit 1
python3 scripts/skill_validate_hook.py --command 'ros2 topic list'  # status: pass, exit 0
```

Without `--file`/`--command`, `skill_validate_hook.py` expects a Claude Code
PreToolUse payload on stdin — it is not a workspace-scanning CLI in that mode.

### Cursor

```bash
git clone https://github.com/dbwls99706/ros2-engineering-skills.git
# Add to .cursor/rules/ros2-engineering-skills
```

### Any project (symlink)

```bash
ln -s /path/to/ros2-engineering-skills .claude/skills/ros2-engineering-skills
```

## Structure

```text
ros2-engineering-skills/
├── SKILL.md                        # Entry point — decision router + core principles
├── references/                     # 23 reference files (15,000+ lines)
│   ├── workspace-build.md          # colcon, ament_cmake, package.xml, overlays
│   ├── nodes-executors.md          # rclcpp/rclpy nodes, executors, callback groups
│   ├── communication.md            # Topics, services, actions, QoS, type adapters, DDS tuning
│   ├── lifecycle-components.md     # Managed nodes, component loading, composition
│   ├── launch-system.md            # Python launch API, conditions, events, large systems
│   ├── tf2-urdf.md                 # Transforms, URDF, xacro, robot_state_publisher
│   ├── hardware-interface.md       # ros2_control, HW interfaces, controller chaining, EtherCAT
│   ├── realtime.md                 # RT kernel, realtime_tools, jitter, deterministic execution
│   ├── navigation.md               # Nav2, SLAM, costmaps, BT navigator, collision monitor
│   ├── manipulation.md             # MoveIt 2, MTC, planning scene, grasp pipelines
│   ├── perception.md               # image_transport, PCL, cv_bridge, depth, Isaac ROS
│   ├── sensor-integration.md       # Sensor drivers, clock sync (PTP), tf2 MessageFilter, extrinsics
│   ├── simulation.md               # Gazebo, Isaac Sim, gz_ros2_control, sim-to-real
│   ├── security.md                 # SROS2, DDS security plugins, certificates, supply chain
│   ├── safety-estop.md             # E-stop architecture, fail-safe heartbeat QoS, command arbitration
│   ├── micro-ros.md                # micro-ROS, rclc, XRCE-DDS, ESP32/STM32/RP2040
│   ├── multi-robot.md              # Fleet management, Open-RMF, DDS discovery at scale
│   ├── testing.md                  # gtest, pytest, launch_testing, industrial_ci, CI/CD
│   ├── debugging.md                # ros2 doctor, tracing, Foxglove, MCAP, rosbag2
│   ├── deployment.md               # Docker, cross-compile, fleet management, Zenoh routing
│   ├── system-bringup.md           # udev rules, systemd boot ordering, watchdogs, health checks
│   ├── message-types.md            # Message conventions, units, covariance, diagnostics
│   └── migration-ros1.md           # ROS 1 → ROS 2 strategy, ros1_bridge
├── scripts/
│   ├── create_package.py           # Scaffold a package (cpp/python/interfaces/hardware_interface), --component, --lifecycle, --sros2, --robots N
│   ├── qos_checker.py              # Verify QoS compatibility between pub/sub pairs with fix suggestions + vendor-specific warnings
│   ├── launch_validator.py         # AST-based static analysis for Python launch files
│   ├── rosbag2_qos_checker.py      # Validate QoS compatibility for rosbag2 playback against subscriber profiles
│   ├── eval_runner.py              # Skills 2.0 eval harness — structural fixture coverage check
│   ├── skill_validate_hook.py      # Skills 2.0 PreToolUse hook — anti-pattern + dangerous-command guard
│   └── skill_stop_hook.py          # Skills 2.0 Stop hook — launch/package.xml validation + Nav2 YAML lint + opt-in .skill-runs.log
├── tests/
│   ├── test_create_package.py      # scaffolding, validation, copyright, lifecycle matcher regression
│   ├── test_launch_validator.py    # AST visitors, patterns, CLI, main()
│   ├── test_qos_checker.py         # parsing, compatibility, presets (incl. SKILL.md cross-check), CLI
│   ├── test_qos_property.py        # Hypothesis property-based DDS RxO verification
│   ├── test_rosbag2_qos_checker.py # bag metadata parsing, playback QoS compatibility
│   ├── test_integration.py         # end-to-end multi-script workflows
│   ├── test_skills2_frontmatter.py # SKILL.md metadata completeness + size budget
│   ├── test_skills2_evals.py       # eval runner, criteria, fixture coverage
│   ├── test_skills2_hooks.py       # pre/post hooks, dangerous-command coverage (bash + PowerShell)
│   └── Dockerfile.ros2-test        # Multi-stage Docker test (build + validate across distros)
├── setup.cfg                       # flake8 + mypy configuration
├── pytest.ini                      # pytest configuration
├── LICENSE
└── README.md
```

## Current status

23 reference files, 15,000+ lines of guidance, 7 utility/harness scripts (4 user-facing + 2 Skills 2.0 hooks + 1 eval harness). The scripts and harness are covered by automated tests and were exercised on live ROS 2 Jazzy environments.

**What the automated checks do — and do not — verify:**

- The hooks perform *lightweight validation only*: launch-file Python syntax
  (including a `generate_launch_description` check), `package.xml` structure,
  and a Nav2 YAML lint for syntax and selected legacy identifiers
  (pre-Humble recovery naming, the pre-Galactic BT navigator parameter).
  They do **not** verify plugin exports, parameter types, BT XML contents,
  CMake exports, real builds, or lifecycle behavior.
- The factual accuracy of the reference documents is **not** established by
  the test suite. A small set of documentation regression tests
  (`tests/test_doc_factuality.py`) pins previously discovered errors so they
  cannot reappear; everything else should be verified against your installed
  distro (source-first procedure: `references/navigation.md` section 6).

| | |
|---|---|
| **Automated tests** | Hooks, frontmatter, eval definitions, script behavior (unit + property-based Hypothesis + CLI + integration), and selected documentation regressions |
| **Quality gates** | `pytest --cov=scripts --cov-fail-under=90`; flake8 + mypy clean |
| **Real-world Evals** | **Validated empirically on WSL (Ubuntu 24.04 + ROS 2 Jazzy)** for SROS2, micro-ROS `rclc`, and Multi-robot fleet scenarios. The `eval_runner.py` performs *structural* checks on prompt/expected fixtures (keyword coverage of declared criteria); model-output quality is evaluated outside this runner. |
| **5 CI jobs** | Lint (flake8 + mypy + pip-audit), unit-tests (py 3.10/3.11/3.12 matrix), ros2-integration (humble/jazzy/kilted/lyrical/rolling Docker matrix), markdown-lint, lint-scripts |

## Supported ROS 2 distributions

This skill is designed to work **on a complete, internally consistent ROS 2
installation**. The matrix below describes what "complete" means per distro,
and which combinations are CI-verified end-to-end.

Use **Lyrical** on its primary Ubuntu 26.04 platform. Use **Jazzy** for
Ubuntu 24.04 deployments. **Humble** remains supported for Ubuntu 22.04.

| Distro | Status | CI verification | Notes |
|---|---|---|---|
| **Lyrical Luth** (LTS, May 2026) | Recommended for Ubuntu 26.04 | Full pipeline (lint → unit → docker build → colcon test → smoke) | EventsCBGExecutor, ros2_control 6.x — references document the deltas |
| **Jazzy Jalisco** (LTS) | Recommended for Ubuntu 24.04 | Full pipeline (lint → unit → docker build → colcon test → smoke) | Scripts and scaffolds are CI-verified against Jazzy idioms |
| **Humble Hawksbill** (LTS) | Fully supported (Ubuntu 22.04) | Full pipeline | Distro-aware code paths handle 22.04 / older rosidl / pre-`HardwareComponentInterfaceParams` API |
| **Kilted Kaiju** (non-LTS, May 2025, EOL Dec 2026) | Reference-supported | Full pipeline | Zenoh Tier 1, experimental EventsExecutor rclpy port — references document the deltas |
| **Rolling Ridley** | CI-verified via source overlay of broken upstream packages | Full pipeline (rolling-only stage in Dockerfile) | See **Rolling caveat** below |
| **Foxy Fitzroy** (LTS, EOL June 2023) | Migration reference only | Not built | Documented for upgrade paths only |

### Rolling caveat

`Rolling` is, by ROS 2 policy, an upstream development distribution with no
ABI guarantees. During active refactors (e.g., the in-progress `rosidl`
split into `rosidl_buffer`, `rosidl_buffer_backend`, …), `packages.ros.org`
periodically enters states where freshly-rebuilt binary `.deb` files
(e.g. `control_msgs`, `hardware_interface`) declare CMake link-interface
targets whose providing packages have not yet propagated as standalone
debs — making `find_package(hardware_interface)` fail at CMake generate
time with `"target was not found"`.

This project handles that with a **source overlay** in
[tests/Dockerfile.ros2-test](tests/Dockerfile.ros2-test): on rolling
only, after the broken binaries are apt-installed, we clone every
upstream repo whose source tree currently underpins the broken Config
chain (every `rosidl*` repo from the canonical `ros2.repos` manifest,
plus `ros-controls/control_msgs` and `ros-controls/ros2_control`) and
`colcon build --merge-install --install-base /opt/ros/rolling` them as
one workspace. The regenerated `Config.cmake` files reference each
other consistently and overlay onto `/opt/ros/rolling`, replacing the
broken binary configs in place. A throwaway `find_package(hardware_interface)`
verification at the end of the overlay step fails fast if anything
remains unresolved.

For your own deployments on rolling, the same approach applies — clone
the `rosidl*`/`control_msgs`/`ros2_control` source trees and overlay
them with `colcon build --merge-install`.

For production work, **pick an LTS** — Lyrical on Ubuntu 26.04, Jazzy on
Ubuntu 24.04, Humble on Ubuntu 22.04. Use rolling only when you
specifically need a feature that has not yet landed in an LTS.

## Contributing

Contributions welcome. Please:

1. Keep `SKILL.md` under 500 lines — add depth in `references/`
2. Include working code examples, not pseudocode
3. Document anti-patterns alongside correct patterns
4. Note which ROS 2 distros your change applies to
5. Install the dev dependencies (`pip install -r requirements-dev.txt`),
   then run `flake8 scripts/ tests/` and `mypy scripts/` before submitting
6. Ensure `pytest tests/ --cov=scripts --cov-fail-under=90` passes
7. Test with at least one agent (Claude Code, Codex, etc.)

## License

Apache-2.0 — see [LICENSE](LICENSE).
