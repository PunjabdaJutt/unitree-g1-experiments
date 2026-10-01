import collections
import os
import sys
import threading
import time

# ---------------------------------------------------------------------
# Unitree SDK environment
#
# ~/.bashrc sources ROS2 Jazzy in every shell, putting ROS's libddsc
# 0.10.5 ahead of the 0.10.2 the pip cyclonedds binding was built
# against -- mixing them corrupts memory and aborts with
# "*** buffer overflow detected ***" inside ChannelFactoryInitialize.
# The loader only reads LD_LIBRARY_PATH at process start, so the fix
# is to re-exec with a pinned environment before importing anything.
# ---------------------------------------------------------------------
CYCLONEDDS_LIB = os.path.expanduser('~/.local/cyclonedds/lib')

# Outer unitree-g1-experiments repo root (this file lives in the
# Port-Hamiltonian submodule under simulation/.../hardware/Detection/),
# where the vendored .unitree_sdk2_python_src checkout lives.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTER_REPO_ROOT = os.path.abspath(os.path.join(_HERE, *[".."] * 6))


def reexec_with_clean_env():
    # Sentinel rather than comparing LD_LIBRARY_PATH: importing cv2
    # mutates LD_LIBRARY_PATH, which would make a value comparison fail
    # forever and exec-loop.
    if os.environ.get('G1_YOLO_REEXECED') == '1':
        return
    env = dict(os.environ)
    env['G1_YOLO_REEXECED'] = '1'
    env['LD_LIBRARY_PATH'] = CYCLONEDDS_LIB
    env.pop('CYCLONEDDS_URI', None)
    env.pop('RMW_IMPLEMENTATION', None)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


reexec_with_clean_env()

sys.path.insert(0, os.path.join(OUTER_REPO_ROOT, ".unitree_sdk2_python_src"))

import cv2
import numpy as np
from ultralytics import YOLO

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL_PATH = os.path.join(_HERE, "cube_yolo26_seg.pt")
CONF = 0.35
IMGSZ = 640

# Network interface connected to the G1
INTERFACE = sys.argv[1] if len(sys.argv) > 1 else "eno1"

# ---------------------------------------------------------------------
# Lidar side panel -- ported inline from tools/camera_grabber.py's
# LidarPanel (same pattern hand_detector.py already uses for reusing that
# file's code: it isn't structured as an importable library -- it re-execs
# and imports cv2 unconditionally at module scope -- and lives in the
# outer repo, not this submodule). Same topic/dtype/rendering as there;
# only the 3D perspective panel was left out for simplicity (add it back
# from camera_grabber.py's LidarPanel.render_3d if you want it too).
# ---------------------------------------------------------------------
LIDAR_TOPIC = 'rt/utlidar/cloud_livox_mid360'
LIDAR_POINT_DTYPE = np.dtype({
    'names': ['x', 'y', 'z'], 'formats': ['<f4', '<f4', '<f4'],
    'offsets': [0, 4, 8], 'itemsize': 22,
})
PANEL_H = 480                  # display height of both panels, side by side
LIDAR_RANGE_M = 15.0           # bird's-eye view covers +/- this many meters
LIDAR_Z_MIN, LIDAR_Z_MAX = -1.0, 2.0  # height range for coloring
LIDAR_ACCUM_SWEEPS = 5         # pool this many sweeps (~0.5s) for a denser cloud


class LidarPanel:
    """Subscribes to the Mid-360 over raw DDS and renders a top-down view."""

    def __init__(self):
        self._lock = threading.Lock()
        # The Mid-360 scans a non-repetitive pattern, so one sweep is
        # sparse; pooling the last few gives a denser, steadier picture.
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
        # Robot at center: +x forward -> up on screen, mirrored on +y so
        # the panel matches what you see standing behind the robot.
        px = (PANEL_H / 2 + y * scale).astype(np.int32)
        py = (PANEL_H / 2 - x * scale).astype(np.int32)
        keep = (px >= 0) & (px < PANEL_H) & (py >= 0) & (py < PANEL_H)
        px, py, z = px[keep], py[keep], z[keep]

        zn = np.clip((z - LIDAR_Z_MIN) / (LIDAR_Z_MAX - LIDAR_Z_MIN), 0, 1)
        colors = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[py, px] = colors[:, 0, :]

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

    def close(self):
        self.sub.Close()


def main():
    # -------------------------------------------------------------
    # Load YOLO26x
    # -------------------------------------------------------------
    print(f"[INFO] Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # -------------------------------------------------------------
    # Connect to Unitree
    # -------------------------------------------------------------
    print(f"[INFO] Initializing Unitree on {INTERFACE}...")
    ChannelFactoryInitialize(0, INTERFACE)

    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    lidar = LidarPanel()

    print("[INFO] Starting camera + YOLO + lidar")
    print("[INFO] Press 'q' to quit")

    failures = 0

    try:
        while True:

            # ---------------------------------------------------------
            # Get JPEG frame from Unitree VideoClient
            # ---------------------------------------------------------
            code, data = client.GetImageSample()

            if code != 0 or not data:
                failures += 1

                if failures % 30 == 1:
                    print(
                        f"[WARN] GetImageSample failed "
                        f"(code={code}) x{failures}"
                    )

                time.sleep(0.05)
                continue

            failures = 0

            # ---------------------------------------------------------
            # Decode JPEG -> OpenCV BGR image
            # ---------------------------------------------------------
            frame = cv2.imdecode(
                np.frombuffer(bytes(data), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                continue

            # ---------------------------------------------------------
            # YOLO inference
            #
            # Ultralytics accepts the OpenCV BGR numpy array directly.
            # ---------------------------------------------------------
            results = model.predict(
                source=frame,
                conf=CONF,
                imgsz=IMGSZ,
                verbose=False,
            )

            # ---------------------------------------------------------
            # Draw detections
            # ---------------------------------------------------------
            annotated = results[0].plot()

            # ---------------------------------------------------------
            # Composite with the lidar top-down panel, same layout as
            # tools/camera_grabber.py (camera panel | lidar panel)
            # ---------------------------------------------------------
            h, w = annotated.shape[:2]
            cam_panel = cv2.resize(annotated, (int(w * PANEL_H / h), PANEL_H))
            canvas = np.hstack([cam_panel, lidar.render()])
            cv2.imshow("G1 YOLO26x + lidar", canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        lidar.close()
        cv2.destroyAllWindows()
        print("[INFO] Exiting.")


if __name__ == "__main__":
    main()
