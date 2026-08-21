#!/usr/bin/env bash
# Builds and installs the Inspire hand driver (DFX_inspire_service, vendored
# at .dfx_inspire_service_src/) plus its one real dependency, unitree_sdk2 --
# the C++ SDK, NOT unitree_sdk2py (already vendored as a submodule at
# .unitree_sdk2_python_src/).
#
# unitree_sdk2 vendors prebuilt CycloneDDS (ddsc) and CycloneDDS-CXX (ddscxx)
# under its own thirdparty/ dir and installs their headers/libs to
# /usr/local on `sudo make install`. DFX_inspire_service's CMakeLists.txt
# hardcodes /usr/local/include/ddscxx, so unitree_sdk2 MUST be installed at
# the default prefix (/usr/local) -- do not pass -DCMAKE_INSTALL_PREFIX.
#
# This only builds the binaries. It does NOT run them -- see the printed
# next steps at the end for the hardware-specific bits (serial port names,
# udev rules, sudo/dialout permissions) a script can't safely guess.
#
# Run this on whichever machine is physically wired to the two Inspire hand
# USB-serial adapters (likely the robot's onboard computer, not necessarily
# this dev machine).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER_SRC="$REPO_ROOT/.dfx_inspire_service_src"

if [ ! -d "$DRIVER_SRC" ]; then
  echo "error: $DRIVER_SRC not found (expected the vendored DFX_inspire_service checkout)" >&2
  exit 1
fi

echo "==> Installing build dependencies (matches unitree_sdk2's own README)"
sudo apt-get update
sudo apt-get install -y \
  cmake g++ build-essential \
  libyaml-cpp-dev libeigen3-dev libfmt-dev \
  libboost-all-dev libspdlog-dev

echo "==> Cloning + building unitree_sdk2 (C++ SDK)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2 "$WORKDIR/unitree_sdk2"
cmake -S "$WORKDIR/unitree_sdk2" -B "$WORKDIR/unitree_sdk2/build" -DBUILD_EXAMPLES=OFF
cmake --build "$WORKDIR/unitree_sdk2/build" -j"$(nproc)"

echo "==> Installing unitree_sdk2 to /usr/local (sudo required)"
sudo cmake --install "$WORKDIR/unitree_sdk2/build"
sudo ldconfig

echo "==> Building DFX_inspire_service ($DRIVER_SRC)"
cmake -S "$DRIVER_SRC" -B "$DRIVER_SRC/build"
cmake --build "$DRIVER_SRC/build" -j"$(nproc)"

cat <<EOF

Built:
  $DRIVER_SRC/build/inspire_g1
  $DRIVER_SRC/build/hand_example

Next steps (hardware-specific -- do these by hand):

1. Find the real device paths for the two hand USB-serial adapters:
     ls /dev/serial/by-id/          # stable names, if the kernel exposes them
     dmesg | grep -i ttyUSB         # or watch this while plugging each in
   inspire_g1.cpp currently hardcodes /dev/ttyUSB1 (right hand) and
   /dev/ttyUSB2 (left hand) at 115200 baud. If yours differ, edit
   $DRIVER_SRC/inspire_g1.cpp and re-run
   'cmake --build $DRIVER_SRC/build'.

2. If enumeration order isn't stable across reboots/replugs, add udev rules
   pinning each adapter (by vendor/serial ID) to a fixed /dev/ttyUSBn or a
   symlink, then update the paths in step 1 accordingly.

3. Run the driver (needs root for raw serial + DDS socket access, same as
   Unitree's own README):
     sudo $DRIVER_SRC/build/inspire_g1 --network \${ROBOT_NET_IFACE:-eno1}
   (--network matches the same NIC pinning used by docker-compose.yml's
   ROBOT_NET_IFACE for the ROS2 container -- run this on the host, not
   inside that container: it links its own CycloneDDS build, so mixing it
   into the ROS2 container's process risks the same libddsc ABI conflict
   already seen with unitree_sdk2py.)

4. Verify, in another terminal:
     $DRIVER_SRC/build/hand_example
   or, with the ROS2 container already running (network_mode: host means
   it shares the DDS domain with the driver process above):
     docker compose run --rm g1 ros2 topic echo /inspire/state --once
EOF
