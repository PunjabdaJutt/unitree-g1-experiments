# Installation

Step-by-step setup for the ROS2 stack in this repo (`ros2_ws/`, `docker/`) on
a fresh Ubuntu machine. The ROS2 environment itself (ROS2 Jazzy + all
`rosdep` dependencies) is installed **inside a Docker image**, not on the
host -- the only thing you need to install on the host is Docker. See
[README.md](README.md) for the rationale and troubleshooting notes.

Tested on Ubuntu 24.04 LTS (noble).

## 1. Clone the repo

This repo vendors the Unitree SDK as a git submodule, so clone with
`--recurse-submodules` (or run the `submodule update` step below if you
already cloned without it):

```bash
git clone --recurse-submodules https://github.com/PunjabdaJutt/unitree-g1-experiments.git
cd unitree-g1-experiments

# if you already cloned without --recurse-submodules:
git submodule update --init --recursive
```

## 2. Install Docker

Nothing else needs to be installed on the host -- ROS2 Jazzy and every
package it depends on (`colcon`, `rosdep`, CycloneDDS, `xacro`, etc.) are
installed inside the container image in step 3.

```bash
bash docker/install_docker.sh
```

This installs Docker Engine + the compose plugin and adds your user to the
`docker` group. Log out and back in (or run `newgrp docker`) for the group
change to take effect, then sanity-check:

```bash
docker run hello-world
```

## 3. Build the ROS2 image

```bash
docker compose build
```

This pulls `osrf/ros:jazzy-desktop`, installs the ROS2 dependencies listed
in `docker/Dockerfile` plus everything `rosdep` resolves for the packages in
`ros2_ws/src` (`unitree_go`, `unitree_hg`, `unitree_api`, `g1_description`,
`g1_bridge`), and runs an initial `colcon build`. Expect this to take a few
minutes the first time.

## 4. Network configuration (only needed with the physical robot)

The container uses `network_mode: host` and pins CycloneDDS to one network
interface (default `eno1`) so DDS discovery reaches the robot without
leaking onto other interfaces. If the G1 is wired to a different interface
on your machine, override it:

```bash
ROBOT_NET_IFACE=<your-iface> docker compose run --rm g1 ros2 topic list
```

## 5. Verify

With the robot connected and powered on:

```bash
docker compose run --rm g1 ros2 topic list
docker compose run --rm g1 ros2 topic echo /lowstate --once
```

To visualize the robot's pose in RViz (needs an X server on the host):

```bash
xhost +local:docker
docker compose run --rm g1 ros2 launch g1_bridge bridge.launch.py
```

Or view the URDF without the robot connected (slider GUI instead of live
data):

```bash
docker compose run --rm g1 ros2 launch g1_description display.launch.py
```

## Troubleshooting

See the "Troubleshooting" section of [README.md](README.md) for issues hit
(and fixed) while validating this setup against the real robot.
