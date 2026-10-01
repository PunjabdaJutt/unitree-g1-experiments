# lidar_object_distance.py
"""Distance-to-cube estimate from the G1's own Mid-360 lidar, as a second,
independent measurement alongside the D435i depth path (Detection/).

There's no existing pixel<->lidar projection in this repo to reuse: this
was expected to exist (tools/camera_grabber.py declares CAM_HFOV_DEG /
CAM_OFFSET_* constants for exactly this, with a comment about a toggleable
overlay), but grepping that file and the ros2_ws/g1_bridge/camera_view.py
it half-references turned up no actual projection function or overlay
implementation, only the unused constants. This file is a from-scratch,
explicitly-approximate implementation:

  1. Run YOLO26-seg (the same CubeSegmenter as Detection/detection.py) on
     the front camera (VideoClient.GetImageSample -- a plain monocular
     JPEG feed, same source as yolo_app.py/hand_detector.py) to get the
     cube's bbox center pixel.
  2. Convert that pixel to a bearing angle (degrees left/right of the
     camera's forward axis) using an ASSUMED horizontal FOV
     (CAM_HFOV_DEG, same 90 deg guess as camera_grabber.py -- not a
     factory spec, tune this if bearings look off).
  3. Accumulate a few Mid-360 sweeps (rt/utlidar/cloud_livox_mid360, same
     topic/dtype as camera_grabber.LidarPanel), compute each point's
     azimuth = atan2(y, x) and range = hypot(x, y) in the lidar frame.
  4. Take the nearest lidar point whose azimuth falls within
     BEARING_TOLERANCE_DEG of the camera bearing -- "nearest", not
     "median", because within a many-degree cone the first real surface
     the beam hits is the most likely to actually be the cube (a median
     would get dragged toward whatever's most common in the cone, e.g. a
     wall behind it).

Known inaccuracies (TODO, robot-specific, needs bench validation):
  - CAM_HFOV_DEG is a guess (see camera_grabber.py's own docstring
    admitting the same thing), not measured against this camera.
  - No lidar<->camera extrinsic offset is applied (assumes both sensors
    share an origin) -- camera_grabber.py's CAM_OFFSET_FWD/RIGHT/UP exist
    for this but were never wired into an actual transform either; they're
    small (~5cm) so likely a minor error source next to the FOV guess.
  - The Mid-360 has a well-known blind cone straight ahead at very close
    range and a non-repeating scan pattern -- a small/close cube may not
    get many (or any) returns; this is a coarse cross-check against the
    D435i, not a replacement for it.

Usage:
    python3 lidar_object_distance.py [network_interface]
"""
import collections
import os
import sys
import time

CYCLONEDDS_LIB = os.path.expanduser('~/.local/cyclonedds/lib')


def reexec_with_clean_env():
    if os.environ.get('G1_LIDAR_DIST_REEXECED') == '1':
        return
    env = dict(os.environ)
    env['G1_LIDAR_DIST_REEXECED'] = '1'
    env['LD_LIBRARY_PATH'] = CYCLONEDDS_LIB
    env.pop('CYCLONEDDS_URI', None)
    env.pop('RMW_IMPLEMENTATION', None)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


reexec_with_clean_env()

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTER_REPO_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 5)))
sys.path.insert(0, os.path.join(_OUTER_REPO_ROOT, '.unitree_sdk2_python_src'))
sys.path.insert(0, os.path.join(_HERE, 'Detection'))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.go2.video.video_client import VideoClient  # noqa: E402
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_  # noqa: E402

from detection import CubeSegmenter, DEFAULT_MODEL_PATH  # noqa: E402

LIDAR_TOPIC = 'rt/utlidar/cloud_livox_mid360'
LIDAR_POINT_DTYPE = np.dtype({
    'names': ['x', 'y', 'z'], 'formats': ['<f4', '<f4', '<f4'],
    'offsets': [0, 4, 8], 'itemsize': 22,
})
LIDAR_ACCUM_SWEEPS = 8  # pool several sparse sweeps into one denser cloud

# TODO: same hand-tunable guesses as tools/camera_grabber.py -- see this
# file's module docstring for why these aren't calibrated.
CAM_HFOV_DEG = 90.0
BEARING_TOLERANCE_DEG = 8.0
MIN_RANGE_M = 0.15   # ignore returns closer than this (self/near-robot noise)
MAX_RANGE_M = 6.0


class LidarAccumulator:
    def __init__(self):
        self._sweeps = collections.deque(maxlen=LIDAR_ACCUM_SWEEPS)
        self.sub = ChannelSubscriber(LIDAR_TOPIC, PointCloud2_)
        self.sub.Init(self._on_cloud, 5)

    def _on_cloud(self, msg: PointCloud2_):
        if msg.point_step != LIDAR_POINT_DTYPE.itemsize:
            return
        pts = np.frombuffer(bytes(msg.data), dtype=LIDAR_POINT_DTYPE)
        self._sweeps.append(pts)

    def points(self):
        if not self._sweeps:
            return None
        return np.concatenate(list(self._sweeps))

    def close(self):
        self.sub.Close()


def bearing_to_pixel(u: float, frame_width: int, hfov_deg: float = CAM_HFOV_DEG) -> float:
    """Pixel x-coordinate -> bearing in degrees, positive = right of
    the camera's forward axis (matches the lidar's +y = robot's right
    convention used in camera_grabber.py's top-down panel)."""
    return (u - frame_width / 2.0) / (frame_width / 2.0) * (hfov_deg / 2.0)


def nearest_in_bearing(points, bearing_deg: float,
                        tolerance_deg: float = BEARING_TOLERANCE_DEG):
    """Returns (range_m, azimuth_deg, n_matched) for the closest lidar
    point whose azimuth falls within tolerance_deg of bearing_deg, or None
    if no lidar points matched."""
    if points is None or len(points) == 0:
        return None
    x, y = points['x'], points['y']
    rng = np.hypot(x, y)
    az_deg = np.degrees(np.arctan2(y, x))
    keep = ((np.abs(az_deg - bearing_deg) < tolerance_deg)
            & (rng > MIN_RANGE_M) & (rng < MAX_RANGE_M))
    if not np.any(keep):
        return None
    idx = np.argmin(rng[keep])
    matched_rng = rng[keep]
    matched_az = az_deg[keep]
    return float(matched_rng[idx]), float(matched_az[idx]), int(keep.sum())


def main():
    iface = sys.argv[1] if len(sys.argv) > 1 else None
    if iface:
        ChannelFactoryInitialize(0, iface)
    else:
        ChannelFactoryInitialize(0)

    video = VideoClient()
    video.SetTimeout(3.0)
    video.Init()
    seg = CubeSegmenter(DEFAULT_MODEL_PATH)
    lidar = LidarAccumulator()

    print("[INFO] streaming -- printing lidar-based cube distance each frame with a "
          "detection (Ctrl+C to stop)")
    try:
        n = 0
        while n < 200:  # bounded run for a non-interactive smoke test
            code, data = video.GetImageSample()
            if code != 0 or not data:
                time.sleep(0.05)
                continue
            frame = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            det = seg.detect(frame)
            if det is None:
                time.sleep(0.05)
                continue

            u, _ = det.centroid_px
            frame_w = frame.shape[1]
            bearing_deg = bearing_to_pixel(u, frame_w)

            result = nearest_in_bearing(lidar.points(), bearing_deg)
            if result is None:
                print(f"score={det.score:.2f} bearing={bearing_deg:+.1f}deg "
                      f"-- no lidar returns in that bearing window yet")
            else:
                range_m, matched_az_deg, n_matched = result
                print(f"score={det.score:.2f} bearing={bearing_deg:+.1f}deg "
                      f"lidar_range_m={range_m:.2f} matched_azimuth={matched_az_deg:+.1f}deg "
                      f"n_matched_points={n_matched}")
            n += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        lidar.close()


if __name__ == '__main__':
    main()
