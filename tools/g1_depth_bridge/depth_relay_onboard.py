# depth_relay_onboard.py
"""Publishes the G1's D435i depth camera (color + aligned depth) as a raw
DDS topic, for depth_camera_client.py (or anything else) to subscribe to
from elsewhere on the robot's network.

Deployment: this whole directory (g1_depth_bridge/) is meant to be copied
onto the onboard Jetson as a unit and run from there -- see README.md for
the one-time setup (install_jetson.sh) and how to run this either directly
or as the provided systemd service. Do not copy this single file alone;
g1_depth_msg.py must sit next to it, and depth_camera_client.py on the
workstation side must stay byte-for-byte identical to that same file.

Why raw DDS instead of a ROS topic: the D435i has no DDS/ROS2 path off this
onboard Jetson by default -- it's a USB device local to this box only, same
class of problem this repo already hit with /frontvideostream. That
problem was specifically ROS2's rmw_cyclonedds_cpp failing to deserialize
Unitree's multi-sequence video message; raw unitree_sdk2py DDS (no ROS2
rmw involved) is this repo's own confirmed-working alternative for exactly
this kind of byte-sequence camera payload, and is the same mechanism
camera_grabber.py already uses successfully for the Mid-360 lidar. This
also sidesteps the ROS1(Noetic, this box)/ROS2(Jazzy, the workstation)
version mismatch entirely -- there is no ROS involved on either end here.
(This box also has a ROS2 Foxy + custom-built rmw_cyclonedds_cpp under
~/cyclonedds_ws, version-matched to unitree_sdk2py's own cyclonedds==0.10.2
-- confirms raw CycloneDDS at this version is the platform's own intended
low-level interop mechanism, not an improvised workaround.)

Run this ON THE ONBOARD JETSON (not the workstation), from inside this
directory (see README.md for one-time setup):
  python3 depth_relay_onboard.py [network_interface]
"""
import os
import sys
import time

LIBREALSENSE_SO = "/usr/local/lib/librealsense2.so.2.58"
WIDTH, HEIGHT, FPS = 640, 480, 30
WARMUP_FRAMES = 30  # let auto-exposure settle before publishing


def reexec_with_clean_env():
    """Re-exec with LD_PRELOAD pinned to the real librealsense2.so.

    pyrealsense2's own .so embeds a second copy of Easylogging++'s global
    logger state; without preloading the "real" librealsense2.so first, the
    dynamic linker routes librealsense2.so's static initializer into
    pyrealsense2's not-yet-constructed copy and segfaults on import, before
    any Python code runs. Confirmed via gdb backtrace on this box
    (2026-08-31) -- LD_PRELOAD is the fix.
    """
    if os.environ.get('G1_DEPTH_RELAY_REEXECED') == '1':
        return
    env = dict(os.environ)
    env['G1_DEPTH_RELAY_REEXECED'] = '1'
    env['LD_PRELOAD'] = LIBREALSENSE_SO
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


reexec_with_clean_env()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import pyrealsense2 as rs
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher

from g1_depth_msg import DepthColorFrame_, TOPIC_NAME


class DepthCameraPipeline:
    """Opens the D435i, aligns depth to color, and yields (color_bgr,
    depth_raw_uint16) pairs plus the intrinsics/scale needed to interpret
    them. Depth is aligned into the color image's pixel grid (RealSense's
    built-in alignment) so a single intrinsics set covers both."""

    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        profile = self.pipeline.start(config)

        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.fx, self.fy, self.cx, self.cy = intr.fx, intr.fy, intr.ppx, intr.ppy

        print(f"[INFO] warming up ({WARMUP_FRAMES} frames, letting auto-exposure settle)...")
        for _ in range(WARMUP_FRAMES):
            self.pipeline.wait_for_frames()

    def read(self):
        """Returns (color_bgr uint8 HxWx3, depth_raw uint16 HxW), or None
        if either stream dropped a frame this cycle."""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return color, depth

    def close(self):
        self.pipeline.stop()


def publish(iface: str = None):
    ChannelFactoryInitialize(0, iface)
    pub = ChannelPublisher(TOPIC_NAME, DepthColorFrame_)
    pub.Init()

    cam = DepthCameraPipeline()
    print(f"[INFO] publishing on DDS topic '{TOPIC_NAME}'...")

    try:
        while True:
            result = cam.read()
            if result is None:
                continue
            color, depth = result
            ok1, color_jpeg = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 85])
            ok2, depth_png = cv2.imencode('.png', depth)  # lossless, keeps uint16
            if not ok1 or not ok2:
                continue

            msg = DepthColorFrame_(
                time_frame=int(time.time() * 1e6),
                width=WIDTH, height=HEIGHT,
                fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
                depth_scale=cam.depth_scale,
                color_jpeg=color_jpeg.tobytes(),
                depth_png=depth_png.tobytes(),
            )
            pub.Write(msg)
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        pub.Close()
        print("[INFO] exiting")


if __name__ == '__main__':
    iface = sys.argv[1] if len(sys.argv) > 1 else None
    publish(iface)
