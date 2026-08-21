# unitree-g1-experiments

Experiments with the Unitree G1 humanoid: raw DDS control scripts (`high_level/`,
`low_level/`, `modes/`) built on the vendored `unitree_sdk2py` SDK
(`.unitree_sdk2_python_src/`), a Port-Hamiltonian finger controller
(`porthamiltonian/`), and a Dockerized ROS2 stack (`ros2_ws/`, `docker/`) for
inspecting the robot's live topics and visualizing its pose in RViz or MuJoCo.

## ROS2 / Docker setup

### What this fixes

The host already had a partial ROS2 setup (`~/unitree_ros2`, outside this
repo) that could see the robot's raw DDS topics via `ros2 topic list`, but
`ros2 node list` came back empty and CycloneDDS logged
`Failed to parse type hash ... USER_DATA (null)` warnings. Root cause: that
workspace's `setup.sh` sourced ROS2 Jazzy while its own `setup_default.sh`/
`setup_local.sh` still referenced Foxy -- an inconsistent setup, on top of
which the raw SDK scripts in this repo (`unitree_sdk2py`) talk CycloneDDS
directly and never go through `rclpy`, so they never appear as real ROS2
nodes regardless of distro.

This setup vendors Unitree's official ROS2 message packages
(`unitree_go`, `unitree_hg`, `unitree_api`) into `ros2_ws/src/unitree`,
builds them inside a container pinned to ROS2 Jazzy (matching the host), and
adds a small bridge node that turns the robot's raw `/lowstate` into a
proper `sensor_msgs/JointState` so RViz can actually show the robot moving.

**`ros2 node list` will still come back empty** -- confirmed against the
live robot, this is unaffected by distro or container. The robot's onboard
software publishes over raw CycloneDDS without ROS2's type-hash `USER_DATA`
(hence the `Failed to parse type hash ... USER_DATA (null)` warnings on
every topic, from firmware we don't control), so it never appears as a real
ROS2 node no matter what talks to it. `ros2 topic list`/`topic echo` work
fine regardless -- DDS-level topic discovery doesn't need that metadata,
only the node-graph view does. The one *real* ROS2 node in this stack is
`g1_bridge`'s own `joint_state_bridge` -- run it and it *will* show up in
`ros2 node list`, because unlike the robot's firmware it's an actual rclpy
node.

### Layout

```
docker/
  Dockerfile              osrf/ros:jazzy-desktop + colcon build of ros2_ws/src
  entrypoint.sh            sources ROS2 + the workspace, renders cyclonedds.xml
  cyclonedds.xml.template  DDS config pinned to the robot's NIC (ROBOT_NET_IFACE)
  install_docker.sh        one-time host setup (needs sudo, run yourself)
docker-compose.yml
ros2_ws/src/
  unitree/                 vendored unitree_go / unitree_hg / unitree_api (message defs only)
  g1_description/          G1 29dof + Inspire-hand URDF, meshes, RViz config
  g1_bridge/               /lowstate -> /joint_states, MuJoCo mirror, Inspire hand Modbus bridge
.unitree_mujoco_src/       vendored unitreerobotics/unitree_mujoco (git submodule) -- isolated
                           G1 simulator for sim-side controller dev, see "MuJoCo" below
simulation/mujoco_menagerie/unitree_g1/  vendored google-deepmind/mujoco_menagerie G1 MJCF (reference
                                          asset -- unitree_mujoco ships its own compatible copy, used instead)
```

### Network

The robot is wired to `eno1` (192.168.123.150/24, same subnet as the G1).
`docker-compose.yml` runs with `network_mode: host` (required -- DDS
multicast discovery does not cross a bridge network without manual unicast
peer config) and pins CycloneDDS to that one interface so discovery traffic
doesn't also leak onto other host interfaces (this machine also has a
second, unrelated network on `enx0a4428e05ec5`). Override with
`ROBOT_NET_IFACE=<iface> docker compose ...` if your cabling differs.

### MuJoCo

Two independent, separately-runnable pieces:

- **Live mirror** (`g1_bridge`'s `mujoco_mirror` node, launched via
  `ros2 launch g1_bridge bridge.launch.py mujoco:=true`): kinematics-only,
  drives a MuJoCo scene's `qpos` straight from `/joint_states` every
  message (no `mj_step`, no gravity/contacts) so it always shows the real
  robot's current pose. `/joint_states` has two independent publishers --
  `joint_state_bridge` (29 body joints, from the robot's DDS `/lowstate`)
  and `inspire_hand_modbus_bridge` (24 Inspire hand joints, launched with
  `hands:=true` -- see below) -- `mujoco_mirror` just copies by joint name,
  it doesn't care which one a name came from.
- **Isolated sim** (`.unitree_mujoco_src/`, Unitree's own simulator):
  makes MuJoCo act as a drop-in DDS substitute for the real robot -- it
  subscribes `rt/lowcmd` and publishes `rt/lowstate` on the *same* topics
  and IDL (`unitree_hg`) the physical G1 uses, just on an isolated
  `domain_id`/interface (`1`/`lo`, set in
  `.unitree_mujoco_src/simulate_python/config.py`) so it can never collide
  with the real robot's `domain_id 0` traffic on the robot-facing NIC. Any
  `unitree_sdk2py` controller script (e.g. the ones under
  `.unitree_sdk2_python_src/example/`) can be pointed at this sim by
  initializing its DDS channel with `(1, "lo")` instead of `(0, <robot NIC>)`
  -- same code path, no real robot required. Run it (not inside the ROS2
  container -- it only needs `unitree_sdk2py` + `mujoco`, both already in
  `.venv/`, and must run outside a ROS2-sourced shell, see "Troubleshooting"):
  ```bash
  cd .unitree_mujoco_src/simulate_python
  LD_LIBRARY_PATH=/home/ammar/.local/cyclonedds/lib ../../.venv/bin/python3 unitree_mujoco.py
  ```

#### Inspire hands: Modbus, not DDS

Unlike the rest of the robot, the Inspire hands don't appear on this G1's
DDS bus at all (`rt/inspire/state` never publishes) -- they're reachable
directly as two Modbus TCP devices on the robot's `192.168.123.0/24`
network (see `tools/inspire_hand_modbus_read.py` for the full writeup).
Confirmed by observation against the physical robot: `192.168.123.210` is
the anatomical **left** hand, `192.168.123.211` is **right** (the reverse
of the factory host-octet order most other tools in this repo list them
in). `inspire_hand_modbus_bridge` polls both directly and publishes their
12-joint-per-hand angles (6 actuated + 6 mechanically-mimicked, see
`g1_bridge/inspire_hand_kinematics.py`) onto `/joint_states`. Override the
IPs with `ros2 launch g1_bridge bridge.launch.py hands:=true
right_hand_ip:=... left_hand_ip:=...` if a hand is ever swapped/re-cabled.

### Install and build

See [INSTALL.md](INSTALL.md) for the full step-by-step setup (installing
Docker, building the image, and running/verifying against the robot).

### Troubleshooting

Issues hit (and fixed) while building and validating this against the real
robot -- listed in case any resurface after a distro bump or a fresh host:

- **`ros2 topic list`/`ros2 daemon start` hangs forever inside the
  container**, or reports `The daemon is already running` in a container
  that just started. Because `network_mode: host` shares the host's
  loopback interface, a `ros2 daemon` process left running *on the host*
  (started by any earlier bare `ros2 ...` invocation) answers the
  container's daemon liveness check -- but the container runs as root
  (`HOME=/root`) while the host daemon runs as your user, so the auth
  token paths never match and the RPC call hangs indefinitely rather than
  failing fast. Fix: `ros2 daemon stop` on the *host* (or
  `pkill -f ros2cli.daemon`) before using the container, or just don't run
  bare `ros2` commands on the host and in the container interchangeably.
- **`rmw_cyclonedds_cpp: eno1: does not match an available interface`**
  even though `network_mode: host` makes the interface visible inside the
  container (`ip -4 addr` / `/proc/net/dev` show it fine). CycloneDDS's
  interface enumeration needs `NET_ADMIN`/`NET_RAW`, which Docker doesn't
  grant by default even under host networking -- `cap_add` for both is
  already in `docker-compose.yml`.
- **`osrf/ros:jazzy-desktop` vs `ros:jazzy-desktop`**: the official
  `library/ros` Docker Hub image doesn't publish a `*-desktop` (GUI/RViz)
  tag for Jazzy, only `*-ros-base` and the bare `*-jazzy` tag. `osrf/ros`
  mirrors it with the desktop variants included -- the Dockerfile uses that.
- **`rosdep install` fails with `Unable to locate package
  ros-jazzy-xacro`** even right after `apt-get update`: caused by deleting
  `/var/lib/apt/lists/*` in an earlier `RUN` layer, which rosdep's own
  internal `apt-get install` then hits with an empty index. Only clean up
  apt lists once, in the final layer, after rosdep/colcon are done with apt.
- **`unitree_hg`/`unitree_go`/`unitree_api` fail to `colcon build` with
  `Could not find ... rosidl_generator_dds_idl`**: these packages generate
  both normal ROS2 interfaces and Connext DDS-IDL bindings side by side
  (`rosidl_generate_dds_interfaces` in their `CMakeLists.txt`), which needs
  `ros-jazzy-rosidl-generator-dds-idl` and `libyaml-cpp-dev` installed --
  neither is pulled in by the base desktop image. Already in the
  Dockerfile.

### Known gaps / next steps

- No IMU-based orientation broadcast (`imu_state.quaternion` in
  `LowState`) -- the model's root (`pelvis`) is fixed at the origin in
  RViz/MuJoCo rather than tilting/rotating with the robot.
- `inspire_hand_modbus_bridge`'s mimic-joint ratios
  (`g1_bridge/inspire_hand_kinematics.py`) are copied by hand from the
  URDF's own `<mimic>` tags, since this MuJoCo version's URDF compiler
  doesn't turn them into equality constraints (`model.neq == 0`). If
  `g1_29dof_with_inspire_hand.urdf`'s hand geometry/ratios ever change,
  that file needs updating to match.
