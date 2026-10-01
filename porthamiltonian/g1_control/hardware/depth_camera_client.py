# depth_camera_client.py
"""Usage:
    client = DepthCameraClient(network_interface='eno1')
    client.connect()
    result = client.get_latest()
    if result is not None:
        color_bgr, depth_m, intrinsics = result  # depth_m: float32 meters, 0 = invalid
    client.close()
"""
import os
import sys
import threading
import time
from typing import Optional, Tuple

CYCLONEDDS_LIB = os.path.expanduser('~/.local/cyclonedds/lib')


def reexec_with_clean_env():
    """Re-exec with LD_LIBRARY_PATH pinned to the SDK-compatible libddsc.

    Same fix as tools/camera_grabber.py and hand_detector.py: a ROS2-sourced
    shell puts ROS's libddsc ahead of the one the pip cyclonedds binding
    needs, corrupting memory if both load.
    """
    if os.environ.get('G1_DEPTH_CLIENT_REEXECED') == '1':
        return
    env = dict(os.environ)
    env['G1_DEPTH_CLIENT_REEXECED'] = '1'
    env['LD_LIBRARY_PATH'] = CYCLONEDDS_LIB
    env.pop('CYCLONEDDS_URI', None)
    env.pop('RMW_IMPLEMENTATION', None)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


reexec_with_clean_env()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 5)))
sys.path.insert(0, os.path.join(REPO_ROOT, '.unitree_sdk2_python_src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'tools', 'g1_depth_bridge'))

import cv2
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

from g1_depth_msg import DepthColorFrame_, TOPIC_NAME


class DepthCameraClient:
    """Subscribes to depth_relay_onboard.py's DDS topic and exposes the
    latest (color_bgr, depth_m, intrinsics) frame, non-blocking."""

    def __init__(self, network_interface: Optional[str] = None, domain_id: int = 0):
        self.network_interface = network_interface
        self.domain_id = domain_id
        self.intrinsics = None
        self._sub: Optional[ChannelSubscriber] = None
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def connect(self):
        ChannelFactoryInitialize(self.domain_id, self.network_interface)
        self._sub = ChannelSubscriber(TOPIC_NAME, DepthColorFrame_)
        self._sub.Init(self._on_frame, 0)

    def _on_frame(self, msg: DepthColorFrame_):
        color = cv2.imdecode(np.frombuffer(bytes(msg.color_jpeg), dtype=np.uint8), cv2.IMREAD_COLOR)
        depth_raw = cv2.imdecode(np.frombuffer(bytes(msg.depth_png), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if color is None or depth_raw is None:
            return
        depth_m = depth_raw.astype(np.float32) * msg.depth_scale
        intrinsics = {
            'width': msg.width, 'height': msg.height,
            'fx': msg.fx, 'fy': msg.fy, 'cx': msg.cx, 'cy': msg.cy,
            'depth_scale': msg.depth_scale,
        }
        with self._lock:
            self._latest = (color, depth_m)
            self.intrinsics = intrinsics

    def get_latest(self):
        """Returns (color_bgr, depth_m, intrinsics), or None if no frame
        has arrived yet. depth_m is float32 meters, 0 where invalid."""
        with self._lock:
            if self._latest is None:
                return None
            color, depth_m = self._latest
            intrinsics = self.intrinsics
        return color, depth_m, intrinsics

    def close(self):
        if self._sub is not None:
            self._sub.Close()


def main():
    """Standalone viewer: subscribe and show color + colorized depth side
    by side. Usage: python3 depth_camera_client.py [network_interface]"""
    iface = sys.argv[1] if len(sys.argv) > 1 else None

    client = DepthCameraClient(network_interface=iface)
    print(f"[INFO] subscribing to '{TOPIC_NAME}'...")
    client.connect()

    try:
        while True:
            result = client.get_latest()
            if result is None:
                time.sleep(0.01)
                continue
            color, depth_m, _ = result
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_m, alpha=255.0 / 4.0), cv2.COLORMAP_TURBO)
            cv2.imshow('G1 depth camera (color | depth)', np.hstack([color, depth_vis]))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
