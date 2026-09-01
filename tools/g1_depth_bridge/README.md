# G1 depth camera DDS bridge

Streams the G1's onboard D435i depth camera (color + aligned depth) off the
robot's onboard Jetson as a raw DDS topic, for
`depth_camera_client.py` (or anything else) to subscribe to from the
workstation. See `depth_relay_onboard.py`'s module docstring for the full
rationale (why raw DDS, not ROS2 or a custom socket).

## Prerequisites (one-time, already done on this specific Jetson as of 2026-08-31)

1. **librealsense + pyrealsense2, built from source, RSUSB backend** (no
   kernel patch needed on this Jetson's stock L4T kernel):
   ```bash
   sudo apt install -y git libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev
   git clone https://github.com/realsenseai/librealsense.git
   cd librealsense && mkdir build && cd build
   cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python3) \
            -DFORCE_RSUSB_BACKEND=true -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false \
            -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)   # drop to -j2 if it OOMs partway -- make resumes
   sudo make install && sudo ldconfig
   ```
2. **Known segfault-on-import fix**: `pyrealsense2`'s own `.so` embeds a
   duplicate copy of Easylogging++'s global logger state, which crashes
   `librealsense2.so`'s static initializer on plain `import pyrealsense2`
   (confirmed via `gdb` backtrace). `depth_relay_onboard.py` already works
   around this itself (`reexec_with_clean_env()` sets `LD_PRELOAD` to the
   real `librealsense2.so.2.58`) -- you don't need to do anything extra for
   this once the file is in place, but if you ever test `pyrealsense2`
   standalone, prefix with:
   ```bash
   LD_PRELOAD=/usr/local/lib/librealsense2.so.2.58 python3 -c "import pyrealsense2"
   ```

## Deploy

From the workstation, copy this whole directory to the Jetson (not just
one file -- `depth_relay_onboard.py` needs `g1_depth_msg.py` next to it):

```bash
scp -r tools/g1_depth_bridge unitree@192.168.123.164:~/
```

Then, on the Jetson:

```bash
cd ~/g1_depth_bridge
./install_jetson.sh   # installs unitree_sdk2py if not already present
```

## Run

**Manually** (for testing/iterating):
```bash
cd ~/g1_depth_bridge
python3 depth_relay_onboard.py
```
Prints a warmup message, then `[INFO] publishing on DDS topic '...'`.
Ctrl+C to stop.

**As a systemd service** (once validated, so it survives reboots/crashes
without being started by hand) -- see `g1-depth-relay.service` for install
steps and how to check logs.

## Test from the workstation

```bash
.venv/bin/python simulation/Port-Hamiltonian-G1-Finger-Control/scripts/g1_control/hardware/depth_camera_client.py eno1
```
Should show a live window: color on the left, colorized depth on the
right.

## If you change `g1_depth_msg.py`

It must stay byte-for-byte identical between this directory (published
from) and `depth_camera_client.py`'s side (subscribed from) -- CycloneDDS
matches readers/writers by type signature. Re-copy it to the Jetson (`scp
tools/g1_depth_bridge/g1_depth_msg.py unitree@192.168.123.164:~/g1_depth_bridge/`)
any time it changes.
