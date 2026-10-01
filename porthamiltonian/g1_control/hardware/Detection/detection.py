# detection.py
"""RealSense D435i capture + YOLO26-seg grey-cube detection.

RealSenseCamera below is self-contained pyrealsense2 (no unitree_sdk2py
dependency) so this file runs unmodified on either machine that might host
the D435i:
  - Official PyPI pyrealsense2 wheels exist for x86_64 Linux (a normal
    workstation) -- confirmed clean bare import here, no segfault, no
    LD_PRELOAD needed.
  - The onboard Jetson (aarch64) has no such wheel; per this repo's own
    finding in tools/g1_depth_bridge/depth_relay_onboard.py, a
    source-built pyrealsense2 there segfaults on import unless
    LD_PRELOAD points at the real librealsense2.so first (confirmed via
    gdb backtrace on that box 2026-08-31). _maybe_preload_librealsense()
    below applies that fix ONLY if LIBREALSENSE_SO actually exists on
    this machine, so it's a no-op (and no unnecessary re-exec) here.

If your compute (YOLO) instead runs on a workstation off the Jetson with
the D435i still physically on the Jetson, use
depth_camera_client.DepthCameraClient (DDS relay, in the parent hardware/
directory) as the frame source instead of RealSenseCamera below -- both
expose the same (color_bgr, depth_m, intrinsics) shape, so
CubeSegmenter.detect() doesn't care which one fed it.

Usage:
    cam = RealSenseCamera()
    model = CubeSegmenter("cube_yolo26_seg.pt")  # trained weights, same dir as this file
    color, depth_m, intr = cam.read()
    det = model.detect(color)
    if det is not None:
        print(det.bbox, det.score, det.centroid_px)
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Same fix, same path, as tools/g1_depth_bridge/depth_relay_onboard.py --
# only relevant on the Jetson's source-built pyrealsense2. Checked BEFORE
# importing pyrealsense2 (LD_PRELOAD only takes effect if set pre-import,
# hence the re-exec) and skipped entirely when the .so isn't present, so
# this never fires on a box using the official wheel.
_LIBREALSENSE_SO = "/usr/local/lib/librealsense2.so.2.58"


def _maybe_preload_librealsense():
    if os.environ.get("G1_DETECTION_REEXECED") == "1":
        return
    if not os.path.isfile(_LIBREALSENSE_SO):
        return
    env = dict(os.environ)
    env["G1_DETECTION_REEXECED"] = "1"
    env["LD_PRELOAD"] = _LIBREALSENSE_SO
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_maybe_preload_librealsense()

import cv2  # noqa: E402
import pyrealsense2 as rs  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# TODO: your trained/fine-tuned grey-cube segmentation weights (see the
# dataset/training notes accompanying this skeleton). yolo26x-seg.pt (the
# stock COCO-seg checkpoint at the repo root) has no "cube" class and will
# not detect a plain grey cube well -- it's only a placeholder for wiring.
DEFAULT_MODEL_PATH = os.path.join(_HERE, "cube_yolo26_seg.pt")
DEFAULT_CLASS_NAME = "item"  # single_cls=True training collapses all classes to this name (see
                              # ultralytics/engine/trainer.py's get_dataset()) -- not "cube"
DEFAULT_CONF = 0.2  # lowered from 0.35 -- cube_yolo26_seg.pt is trained only on saturated
                     # colored cubes (see Cube-segmentation-1/data.yaml) and runs marginal
                     # on the real grey target, so 0.35 was dropping good detections
DEFAULT_IMGSZ = 640

WIDTH, HEIGHT, FPS = 640, 480, 30
WARMUP_FRAMES = 30  # let auto-exposure settle before the first real read


class RealSenseCamera:
    """Opens the D435i, aligns depth to color, and exposes
    (color_bgr, depth_m, intrinsics) -- depth_m is float32 meters, 0 where
    invalid; intrinsics matches depth_camera_client.DepthCameraClient's
    dict shape so localize_3d.py doesn't care which source fed it."""

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

        for _ in range(WARMUP_FRAMES):
            self.pipeline.wait_for_frames()

    @property
    def intrinsics(self) -> dict:
        return {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "depth_scale": self.depth_scale,
        }

    def read(self) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
        """Returns (color_bgr, depth_m, intrinsics), or None on a dropped
        frame pair."""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None
        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        return color, depth_m, self.intrinsics

    def close(self):
        self.pipeline.stop()


class DDSCubeCamera:
    """Depth camera source for the common case: the D435i is physically on
    the robot's onboard Jetson (~/g1_depth_bridge/depth_relay_onboard.py
    running there -- the one with the systemd service and README, NOT the
    stray duplicate that was also found loose at ~/depth_relay_onboard.py
    on that box; that loose copy speaks a different, TCP-socket protocol
    and should be deleted/ignored to stop this exact confusion happening
    again), and this process runs on a workstation with no camera of its
    own -- e.g. this box, whose USB bus has no RealSense device at all
    (confirmed via lsusb).

    Thin adapter around depth_camera_client.DepthCameraClient (this repo's
    established DDS path for the depth camera) so it exposes the exact
    same (color_bgr, depth_m, intrinsics)-returning .read()/.close()
    interface as RealSenseCamera above; CubeSegmenter/localize_3d don't
    care which one fed them. No SSH into the robot needed for this
    process -- just the workstation's network interface on the same LAN as
    the Jetson (matches yolo_app.py's own network_interface convention,
    e.g. 'eno1').

    Requires depth_relay_onboard.py to already be running on the Jetson
    (directly from ~/g1_depth_bridge/, or as the g1-depth-relay.service
    unit sitting right next to it there) publishing the
    'g1_depth_camera/color_depth_frame' DDS topic this subscribes to.
    """

    def __init__(self, network_interface: Optional[str] = None,
                 connect_timeout_s: float = 10.0):
        # depth_camera_client.py lives one level up (hardware/); imported
        # here (not at module scope) so a --local run of this file never
        # triggers its module-level cyclonedds re-exec or needs
        # unitree_sdk2py importable at all.
        _parent_dir = os.path.abspath(os.path.join(_HERE, os.pardir))
        if _parent_dir not in sys.path:
            sys.path.insert(0, _parent_dir)
        from depth_camera_client import DepthCameraClient

        self._client = DepthCameraClient(network_interface=network_interface)
        print(f"[INFO] connecting to depth camera DDS topic "
              f"(iface={network_interface!r}) -- requires depth_relay_onboard.py "
              f"running on the robot's Jetson (~/g1_depth_bridge/, not the loose "
              f"copy in the Jetson's home directory)...")
        self._client.connect()

        import time
        t0 = time.time()
        while self._client.get_latest() is None:
            if time.time() - t0 > connect_timeout_s:
                raise TimeoutError(
                    f"no frame received on the depth camera DDS topic within "
                    f"{connect_timeout_s}s -- is depth_relay_onboard.py actually "
                    f"running on the Jetson (from ~/g1_depth_bridge/), and is "
                    f"{network_interface!r} the right NIC (on the same LAN as the robot)?"
                )
            time.sleep(0.05)
        print("[INFO] receiving depth camera frames over DDS")

    def read(self) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
        return self._client.get_latest()

    def close(self):
        self._client.close()


def open_camera(network_interface: Optional[str] = None, local: bool = False):
    """local=True: D435i is physically plugged into this machine (uses
    RealSenseCamera directly). local=False (default): subscribe to the
    Jetson's DDS relay instead (DDSCubeCamera) -- the common case, and the
    only one that needs no camera hardware on this machine at all."""
    return RealSenseCamera() if local else DDSCubeCamera(network_interface=network_interface)


@dataclass
class CubeDetection:
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels
    score: float
    mask: np.ndarray          # bool, HxW, full frame resolution
    centroid_px: Tuple[float, float]  # (u, v) mask centroid in pixels


class CubeSegmenter:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 class_name: str = DEFAULT_CLASS_NAME,
                 conf: float = DEFAULT_CONF, imgsz: int = DEFAULT_IMGSZ):
        self.model = YOLO(model_path)
        self.class_name = class_name
        self.conf = conf
        self.imgsz = imgsz
        # id->name map so a single-class "cube" model or a multi-class model
        # (e.g. cube treated as one class among several "box" classes) both
        # work without editing this file.
        self._names = self.model.names

    def detect(self, frame_bgr: np.ndarray) -> Optional[CubeDetection]:
        """Runs seg inference and returns the highest-confidence cube
        instance, or None if none cleared self.conf."""
        results = self.model.predict(
            source=frame_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False,
        )[0]
        if results.masks is None or len(results.boxes) == 0:
            return None

        h, w = frame_bgr.shape[:2]
        best_idx, best_score = None, -1.0
        for i in range(len(results.boxes)):
            cls_id = int(results.boxes.cls[i])
            if self._names.get(cls_id, "") != self.class_name:
                continue
            score = float(results.boxes.conf[i])
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx is None:
            return None

        bbox = tuple(float(v) for v in results.boxes.xyxy[best_idx].tolist())
        # Ultralytics masks are at model input resolution -- resize to the
        # original frame so mask pixels line up with the depth image below.
        mask_small = results.masks.data[best_idx].cpu().numpy()
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5

        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        centroid_px = (float(xs.mean()), float(ys.mean()))

        return CubeDetection(bbox=bbox, score=best_score, mask=mask, centroid_px=centroid_px)


def main():
    """Standalone viewer: depth camera -> YOLO26-seg -> annotated window.

    Default: subscribes to the Jetson's depth_relay_onboard.py over DDS
    (no camera needed on this machine, no SSH into the robot -- just the
    network interface on the robot's LAN). Pass --local if the D435i is
    physically plugged into this machine instead.

    Usage: python3 detection.py [network_interface] [--local] [--model PATH]
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("network_interface", nargs="?", default=None,
                         help="NIC on the robot's LAN (e.g. eno1) -- ignored with --local")
    parser.add_argument("--local", action="store_true",
                         help="D435i is physically plugged into this machine")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    cam = open_camera(network_interface=args.network_interface, local=args.local)
    seg = CubeSegmenter(args.model)
    try:
        while True:
            result = cam.read()
            if result is None:
                continue
            color, depth_m, _ = result
            det = seg.detect(color)
            vis = color.copy()
            if det is not None:
                x1, y1, x2, y2 = (int(v) for v in det.bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                overlay = np.zeros_like(vis)
                overlay[det.mask] = (0, 255, 0)
                vis = cv2.addWeighted(vis, 1.0, overlay, 0.4, 0)
                u, v = det.centroid_px
                cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(vis, f"{det.score:.2f}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("detection", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
