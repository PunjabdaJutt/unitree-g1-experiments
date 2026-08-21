"""Live view + recording of the G1's front camera and lidar in one window.

Left panel: front camera JPEG frames polled from the Unitree videohub API
(VideoClient.GetImageSample via rt/api/videohub/request) -- used instead of
subscribing to /frontvideostream, whose firmware encoding cannot be
deserialized by ROS2's rmw_cyclonedds (known upstream issue).

Right panel: Livox Mid-360 lidar (rt/utlidar/cloud_livox_mid360,
sensor_msgs PointCloud2 over raw DDS) rendered as a live top-down
bird's-eye view, colored by height.

Keys in the window:
  r  start/stop recording (saves output/camera_<timestamp>.mp4 of the
     combined view)
  q  quit

Run from the repo root (any shell -- the script re-execs itself with a clean
environment, so it is immune to ROS2 being sourced in ~/.bashrc):
  .venv/bin/python tools/camera_grabber.py eno1
"""
import collections
import os
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CYCLONEDDS_LIB = os.path.expanduser('~/.local/cyclonedds/lib')


def reexec_with_clean_env():
    """Re-exec with LD_LIBRARY_PATH pinned to the SDK-compatible libddsc.

    ~/.bashrc sources ROS2 Jazzy in every shell, putting ROS's libddsc 0.10.5
    ahead of the 0.10.2 the pip cyclonedds binding needs -- mixing them
    corrupts memory. The loader only reads LD_LIBRARY_PATH at process start,
    so fixing it requires an exec, not os.environ.
    """
    # Sentinel rather than comparing LD_LIBRARY_PATH: importing cv2 mutates
    # LD_LIBRARY_PATH in os.environ, which would make a value comparison fail
    # forever and exec-loop if anything imports cv2 before this module.
    if os.environ.get('G1_GRABBER_REEXECED') == '1':
        return
    env = dict(os.environ)
    env['G1_GRABBER_REEXECED'] = '1'
    env['LD_LIBRARY_PATH'] = CYCLONEDDS_LIB
    env.pop('CYCLONEDDS_URI', None)
    env.pop('RMW_IMPLEMENTATION', None)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


reexec_with_clean_env()

sys.path.insert(0, os.path.join(REPO_ROOT, '.unitree_sdk2_python_src'))

import cv2
import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelSubscriber,
)
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')

PANEL_H = 480                  # display height of all panels (3 side by side)
LIDAR_RANGE_M = 15.0           # bird's-eye view covers +/- this many meters
LIDAR_Z_MIN, LIDAR_Z_MAX = -1.0, 2.0  # height range for coloring
LIDAR_TOPIC = 'rt/utlidar/cloud_livox_mid360'

# Third panel: perspective view of the point cloud from a virtual camera
# behind/above the robot looking forward.
VIEW3D_BACK_M = 2.0       # virtual camera distance behind the lidar
VIEW3D_UP_M = 1.5         # ... and above it
VIEW3D_PITCH_DEG = 20.0   # pitched down toward the ground
VIEW3D_FOV_DEG = 75.0     # virtual camera field of view

LIDAR_ACCUM_SWEEPS = 5    # pool this many sweeps (~0.5s) for a denser cloud

# Livox Mid-360 point layout (verified against the live robot):
# x/y/z float32 at offsets 0/4/8, point_step 22.
LIDAR_POINT_DTYPE = np.dtype({
    'names': ['x', 'y', 'z'],
    'formats': ['<f4', '<f4', '<f4'],
    'offsets': [0, 4, 8],
    'itemsize': 22,
})

# Approximate projection of lidar points into the camera image (toggle with
# 'l'). There is no factory lidar->camera calibration available, so these are
# hand-tunable: adjust CAM_HFOV_DEG if points spread too wide/narrow, and the
# OFFSET terms (meters, in the lidar frame) if the overlay is shifted
# relative to the image.
CAM_HFOV_DEG = 90.0     # assumed camera horizontal field of view
CAM_OFFSET_FWD = 0.0    # camera position relative to lidar: forward (+x)
CAM_OFFSET_RIGHT = 0.0  # ... to the robot's right
CAM_OFFSET_UP = -0.05   # ... up (camera sits slightly below the lidar)
OVERLAY_RANGE_M = 10.0  # color scale saturates at this distance


class LidarPanel:
    """Subscribes to the lidar over raw DDS and renders a top-down view."""

    def __init__(self):
        self._lock = threading.Lock()
        # The Mid-360 scans a non-repetitive pattern, so one sweep is sparse;
        # pooling the last few gives a much denser, steadier picture.
        self._sweeps = collections.deque(maxlen=LIDAR_ACCUM_SWEEPS)
        self._count = 0
        self.sub = ChannelSubscriber(LIDAR_TOPIC, PointCloud2_)
        self.sub.Init(self._on_cloud, 5)

    def _on_cloud(self, msg: PointCloud2_):
        if msg.point_step != LIDAR_POINT_DTYPE.itemsize:
            return
        pts = np.frombuffer(bytes(msg.data), dtype=LIDAR_POINT_DTYPE)
        with self._lock:
            self._sweeps.append(pts)
            self._count += 1

    def _points(self):
        with self._lock:
            if not self._sweeps:
                return None, self._count
            return np.concatenate(list(self._sweeps)), self._count

    def render(self):
        pts, count = self._points()
        img = np.zeros((PANEL_H, PANEL_H, 3), dtype=np.uint8)
        if pts is None:
            cv2.putText(img, 'waiting for lidar...', (PANEL_H // 4, PANEL_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            return img

        x, y, z = pts['x'], pts['y'], pts['z']
        scale = PANEL_H / (2 * LIDAR_RANGE_M)
        # Robot at center: +x forward -> up on screen. Screen left/right is
        # mirrored relative to the sensor's +y so the panel matches what you
        # see standing behind the robot (object on the robot's right appears
        # on the right of the panel), consistent with the camera panel.
        px = (PANEL_H / 2 + y * scale).astype(np.int32)
        py = (PANEL_H / 2 - x * scale).astype(np.int32)
        keep = (px >= 0) & (px < PANEL_H) & (py >= 0) & (py < PANEL_H)
        px, py, z = px[keep], py[keep], z[keep]

        zn = np.clip((z - LIDAR_Z_MIN) / (LIDAR_Z_MAX - LIDAR_Z_MIN), 0, 1)
        colors = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[py, px] = colors[:, 0, :]

        # Robot marker + range rings
        c = PANEL_H // 2
        cv2.drawMarker(img, (c, c), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
        for r_m in (5, 10, 15):
            r_px = int(r_m * scale)
            cv2.circle(img, (c, c), r_px, (60, 60, 60), 1)
            cv2.putText(img, f'{r_m}m', (c + r_px - 34, c - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        cv2.putText(img, f'lidar #{count}', (10, PANEL_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return img

    def render_3d(self):
        """Perspective view of the point cloud from behind/above the robot."""
        pts, _ = self._points()
        img = np.zeros((PANEL_H, PANEL_H, 3), dtype=np.uint8)
        if pts is None:
            cv2.putText(img, 'waiting for lidar...', (PANEL_H // 4, PANEL_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            return img

        # Virtual camera 2m behind and 1.5m above the lidar, pitched down,
        # looking forward. Same left/right convention as the other panels.
        x = pts['x'] + VIEW3D_BACK_M
        y = pts['y']
        z = pts['z'] - VIEW3D_UP_M
        pitch = np.radians(VIEW3D_PITCH_DEG)
        xc = x * np.cos(pitch) - z * np.sin(pitch)   # depth along view axis
        zc = x * np.sin(pitch) + z * np.cos(pitch)   # up in view frame
        keep = xc > 0.2
        xc, y, zc, zorig = xc[keep], y[keep], zc[keep], pts['z'][keep]
        if xc.size == 0:
            return img

        f = (PANEL_H / 2) / np.tan(np.radians(VIEW3D_FOV_DEG) / 2)
        u = (PANEL_H / 2 + f * y / xc).astype(np.int32)
        v = (PANEL_H / 2 - f * zc / xc).astype(np.int32)
        keep = (u >= 0) & (u < PANEL_H) & (v >= 0) & (v < PANEL_H)
        u, v, zorig = u[keep], v[keep], zorig[keep]

        zn = np.clip((zorig - LIDAR_Z_MIN) / (LIDAR_Z_MAX - LIDAR_Z_MIN), 0, 1)
        colors = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, :]
        # 2x2 dots so sparse points stay readable in perspective
        for du in (0, 1):
            for dv in (0, 1):
                img[np.clip(v + dv, 0, PANEL_H - 1), np.clip(u + du, 0, PANEL_H - 1)] = colors
        cv2.putText(img, '3d view', (10, PANEL_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return img


class Recorder:
    def __init__(self):
        self.frames = []
        self.start_time = None

    @property
    def active(self):
        return self.start_time is not None

    def start(self):
        self.frames = []
        self.start_time = time.time()
        print('[INFO] Recording started')

    def stop(self):
        elapsed = time.time() - self.start_time
        n = len(self.frames)
        path = os.path.join(OUTPUT_DIR, f'camera_{int(self.start_time)}.mp4')
        self.start_time = None
        if n == 0 or elapsed <= 0:
            print('[WARN] No frames captured, nothing saved')
            return
        fps = n / elapsed
        h, w = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in self.frames:
            writer.write(f)
        writer.release()
        self.frames = []
        print(f'[INFO] Saved {n} frames ({elapsed:.1f}s @ {fps:.1f}fps) to {path}')

    def add(self, frame):
        if self.active:
            self.frames.append(frame)


def main():
    iface = sys.argv[1] if len(sys.argv) > 1 else 'eno1'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ChannelFactoryInitialize(0, iface)
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    lidar = LidarPanel()

    print("[INFO] Streaming camera + lidar -- 'r' to start/stop recording, 'q' to quit")
    rec = Recorder()
    failures = 0
    cam_panel = None
    try:
        while True:
            code, data = client.GetImageSample()
            if code == 0 and data:
                failures = 0
                frame = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
                if frame is not None:
                    h, w = frame.shape[:2]
                    cam_panel = cv2.resize(frame, (int(w * PANEL_H / h), PANEL_H))
            else:
                failures += 1
                if failures % 30 == 1:
                    print(f'[WARN] GetImageSample failed (code={code}) x{failures}')
                time.sleep(0.1)

            if cam_panel is None:
                cam_panel = np.zeros((PANEL_H, int(PANEL_H * 16 / 9), 3), dtype=np.uint8)
                cv2.putText(cam_panel, 'waiting for camera...',
                            (cam_panel.shape[1] // 3, PANEL_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

            canvas = np.hstack([cam_panel, lidar.render(), lidar.render_3d()])
            rec.add(canvas)

            display = canvas
            if rec.active:
                display = canvas.copy()
                cv2.circle(display, (25, 25), 10, (0, 0, 255), -1)
                cv2.putText(display, 'REC', (45, 33), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
            cv2.imshow('G1 camera + lidar', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                rec.stop() if rec.active else rec.start()
    except KeyboardInterrupt:
        pass
    finally:
        if rec.active:
            rec.stop()
        cv2.destroyAllWindows()
        print('[INFO] Exiting.')


if __name__ == '__main__':
    main()
