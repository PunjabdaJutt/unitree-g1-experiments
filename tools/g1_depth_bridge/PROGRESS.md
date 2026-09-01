# G1 depth camera -> object detection: progress

Tracks getting real depth data off the G1's onboard D435i and into a
usable form for object/proximity detection. Same "each stage depends on
the previous being verified" spirit as
`simulation/Port-Hamiltonian-G1-Finger-Control/scripts/g1_control/hardware/PIPELINE.md`.

## Stage 0 -- Feasibility (done)

Goal: object detection with real proximity info ("what's close in front of
the robot"), starting from `hand_detector.py`'s existing monocular-camera
pattern.

Checked and ruled out:
- The G1's front camera (`VideoClient.GetImageSample`) is monocular JPEG,
  no depth channel, no verified calibration -- confirmed in
  `hand_detector.py`'s own module docstring.
- The D435i has **no DDS/ROS2 topic at all** -- it's a USB device local to
  the robot's onboard Jetson only. Confirmed no depth client exists
  anywhere in the vendored `unitree_sdk2py` SDK.
- Bridging the onboard Jetson's ROS1 Noetic to the workstation's ROS2
  Jazzy (`ros1_bridge`) was considered and rejected -- would need both ROS
  generations co-installed somewhere, more infrastructure than needed.

Decision: get code running directly on the onboard Jetson (SSH access
confirmed available), stream frames to the workstation ourselves.

## Stage 1 -- pyrealsense2 on the Jetson (done)

Jetson facts: Jetson-class (Xavier NX/AGX), `aarch64`, L4T R35.3.1
(`5.10.104-tegra`), Ubuntu 20.04. Already had `ros-noetic-librealsense2`
2.50.0 and `ros-noetic-realsense2-camera` installed; no Python bindings.

- Built `librealsense` (now `realsenseai/librealsense`, Intel spun out the
  business -- old `IntelRealSense` URL still redirects) from source, latest
  release (not pinned to the old 2.50.0 already on the box -- no reason to,
  separate install prefix/SONAME) with `-DFORCE_RSUSB_BACKEND=true`
  (libusb-direct backend, no kernel UVC metadata patch needed on this
  Jetson's stock kernel) and Python bindings on.
- Hit a real segfault on bare `import pyrealsense2`: duplicate
  Easylogging++ (`el::`) global logger state across `librealsense2.so` and
  `pyrealsense2`'s own `.so` (the newer `rsutils` split embeds its own
  copy). Root-caused via `gdb -batch -ex run -ex bt` -- crash was in
  `librealsense2.so`'s static initializer calling into `pyrealsense2.so`'s
  not-yet-constructed copy. Fixed with
  `LD_PRELOAD=/usr/local/lib/librealsense2.so.2.58` before importing.
- Chased one red herring during later hardware debugging: kernel
  `uvcvideo` staying bound to the D435i's USB interfaces in `lsusb -t`
  looked suspicious but turned out to be normal/expected, unrelated to any
  real problem.

## Stage 2 -- Architecture: raw DDS relay, not ROS2 (done)

Considered and declined a "real" ROS2 node (`ros-foxy-realsense2-camera`,
confirmed available in apt on this box) in favor of a raw-CycloneDDS relay
via `unitree_sdk2py`, because:
- `unitree_sdk2py` only vendors a `sensor_msgs/PointCloud2` IDL type, not
  `Image` -- consuming a real ROS2 topic would still mean hand-writing an
  IDL type, no shortcut there.
- Whether the workstation's ROS2 Jazzy can even discover topics from this
  Jetson's ROS2 Foxy is untested, different-distro-generation territory --
  a real risk of becoming its own debugging saga.
- Found `~/cyclonedds_ws` on the Jetson: Unitree's own custom-built
  `rmw_cyclonedds_cpp` against CycloneDDS 0.10.2 from source -- the exact
  version `unitree_sdk2py` pins. Confirms raw CycloneDDS at that version is
  this platform's own intended low-level interop mechanism, not an
  improvised workaround.

This is the same mechanism `camera_grabber.py` already uses for the
Mid-360 lidar, and this repo's own confirmed-working alternative to
subscribing over ROS2's rmw layer (which is specifically what fails for
Unitree's `/frontvideostream`).

## Stage 3 -- Build and deploy (done)

Final structure:
- `tools/g1_depth_bridge/g1_depth_msg.py` -- shared DDS/IDL message type
  (`DepthColorFrame_`: JPEG color + PNG depth + intrinsics), must be
  byte-for-byte identical on both ends.
- `tools/g1_depth_bridge/depth_relay_onboard.py` -- publisher, runs on the
  Jetson. Aligns depth to color (RealSense's built-in alignment), 30-frame
  auto-exposure warmup, publishes on DDS topic
  `g1_depth_camera/color_depth_frame`.
- `tools/g1_depth_bridge/install_jetson.sh` -- one-time setup:
  `unitree_sdk2py` wasn't installed on the Jetson at all (confirmed, not a
  redundant reinstall). Its `cyclonedds` dependency needs the native
  CycloneDDS C library findable via `CYCLONEDDS_HOME` -- fixed by pointing
  at Unitree's own pre-built `~/cyclonedds_ws/install/cyclonedds` rather
  than building a second copy.
- `tools/g1_depth_bridge/g1-depth-relay.service` -- optional systemd unit
  for once this is stable enough to want auto-restart.
- `.../hardware/depth_camera_client.py` (workstation side, next to
  `hand_detector.py`) -- `DepthCameraClient`, subscribes over DDS, keeps
  only the latest frame (bounded, drop-old, matching
  `.agents/skills/robot-perception/SKILL.md`'s `CameraStream` pattern),
  exposes `get_latest() -> (color_bgr, depth_m, intrinsics)`.

Deployed as a directory (not loose files in the Jetson's `$HOME`, which is
how the first iteration went and drew a fair complaint) -- see this
directory's own `README.md` for the exact deploy/run steps.

## Stage 4 -- Verified working end-to-end (done, 2026-09-01)

`RuntimeError: Frame didn't arrive within 5000` on the very first run,
after `install_jetson.sh` succeeded. Isolated systematically rather than
guessing: `rs-enumerate-devices` saw the device fine; color-only stream
worked; depth-only stream worked; both streams together worked; the full
sequence (both streams + `get_depth_scale()` + `rs.align()` +
`get_intrinsics()` + `align.process()`) worked. Every piece worked in
isolation -- the original failure was apparently a transient/stale device
state (likely cleared by the USB replugs done while chasing the `uvcvideo`
red herring above), not a real bug. A plain retry then worked cleanly.

Confirmed: `depth_relay_onboard.py` publishing on the Jetson,
`depth_camera_client.py` on the workstation showing a live color +
colorized depth feed, intrinsics printed matching `rs-enumerate-devices`'
own numbers (`640x480`, principal point ~`[326, 247]`, focal lengths
~`606`).

## Stage 5 -- Object detection layer (not started)

Goal (the original ask): object detection with real proximity ("what's
close in front of the robot," "detect changes in the environment"), not
just raw depth streaming.

Plan: 2D detection on the color stream, backproject the detected
object's pixel location through `depth_m` using the intrinsics
`DepthCameraClient.get_latest()` already returns, following
`.agents/skills/robot-perception/SKILL.md`'s `RobotObjectDetector` /
`DepthProcessor` patterns:
- Sample a depth patch and take the median, never a single pixel (holes,
  noise).
- Validate `z > 0` before trusting any depth read.
- Use the D435i's own factory intrinsics (already flowing through from the
  relay) rather than guessing a `K` matrix.
- Decimate frames for the detector model itself rather than running it
  every published frame.

Not yet decided: which detector model (a MediaPipe object-detector task,
matching the existing `HandLandmarker` pattern in `hand_detector.py`, vs.
something else). No code written for this stage yet.
