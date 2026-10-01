# hand_detector.py
"""Camera-based human-hand presence detector for the G1's front camera.

Deliberately does NOT estimate metric 3D hand position or distance -- the
G1's front camera (VideoClient.GetImageSample) is a plain monocular JPEG
feed with no depth channel and no verified camera calibration (see
ros2_ws/src/g1_bridge/g1_bridge/camera_view.py's own comments on the
lidar<->camera projection being hand-tuned, not factory-calibrated). This
still answers "is a hand present in frame right now", debounced against
noise, for g1_hw_handshake.py to use as a trigger -- the handshake arm pose
itself is fixed (see HANDSHAKE_POSE in that file).

HandPresenceDetector.hand_center_and_size() additionally exposes the last
detected hand's normalized image-plane centroid and apparent size. This is
NOT a calibration claim -- it's still just pixels, with no known mapping to
metric distance -- but it is enough for image-based visual servoing (a
proportional controller that steers toward the hand and closes in as it
appears bigger), which only needs consistent sign/scale, not a calibrated
camera. See g1_hw_handshake_visual.py, which is the only consumer of this.

Uses mediapipe's Tasks API (mediapipe.tasks.python.vision.HandLandmarker),
not the older mp.solutions.hands.Hands -- the installed mediapipe==1.0.1
does not expose mp.solutions at all, only the Tasks API. Needs the
hand_landmarker.task model asset at models/hand_landmarker.task (download
with the curl command in HandPresenceDetector's FileNotFoundError below).
"""
import collections
import os
import sys
import time
from typing import Optional, Tuple

CYCLONEDDS_LIB = os.path.expanduser('~/.local/cyclonedds/lib')


def reexec_with_clean_env():
    """Re-exec with LD_LIBRARY_PATH pinned to the SDK-compatible libddsc.

    Same fix as tools/camera_grabber.py in the parent repo: a ROS2-sourced
    shell puts ROS's libddsc ahead of the one the pip cyclonedds binding
    needs, corrupting memory if both load. Ported inline here (not
    imported from camera_grabber.py) since that file isn't structured as
    an importable library -- it re-execs and imports cv2 unconditionally
    at module scope -- and lives in the outer repo, not this submodule.
    """
    if os.environ.get('G1_HAND_DETECTOR_REEXECED') == '1':
        return
    env = dict(os.environ)
    env['G1_HAND_DETECTOR_REEXECED'] = '1'
    env['LD_LIBRARY_PATH'] = CYCLONEDDS_LIB
    env.pop('CYCLONEDDS_URI', None)
    env.pop('RMW_IMPLEMENTATION', None)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


reexec_with_clean_env()

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import RunningMode
from unitree_sdk2py.go2.video.video_client import VideoClient

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(_MODELS_DIR, "hand_landmarker.task")
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")

# The 21-landmark MediaPipe hand topology (thumb/index/middle/ring/pinky chains
# + palm base), hardcoded because mediapipe==1.0.1's Tasks API doesn't expose
# mp.solutions.drawing_utils to draw this for us (see module docstring).
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandPresenceDetector:
    """Polls the G1's front camera and reports debounced hand presence.

    `hand_present()` is a bool, not a position -- `min_detections` of the
    last `window` polled frames must see a hand before it returns True, and
    after a True result it stays False for `cooldown_s` seconds so a
    handshake doesn't immediately re-trigger on the same hand still in
    frame. For image-plane position/size (not metric 3D -- see module
    docstring), use `hand_center_and_size()`.
    """

    def __init__(self, video_client: VideoClient, model_path: str = DEFAULT_MODEL_PATH,
                 window: int = 8, min_detections: int = 5, min_confidence: float = 0.6,
                 cooldown_s: float = 5.0, show: bool = False):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"hand landmarker model not found at {model_path!r} -- download it with:\n"
                f"  curl -sL {_MODEL_URL} -o {model_path}"
            )
        self._client = video_client
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._recent = collections.deque(maxlen=window)
        self._min_detections = min_detections
        self._cooldown_s = cooldown_s
        self._cooldown_until = 0.0
        self._last_landmarks = None  # most recent frame's landmarks, or None if no hand seen
        self._frame_failures = 0  # consecutive GetImageSample failures, see _grab_frame_bgr()
        self._show = show
        self._window_name = "G1 hand detector"

    def _grab_frame_bgr(self):
        code, data = self._client.GetImageSample()
        if code != 0 or not data:
            self._frame_failures += 1
            if self._frame_failures % 30 == 1:
                print(f"\n[WARN] GetImageSample failed (code={code}) x{self._frame_failures} -- "
                      "no hand can be detected while this keeps happening, this is NOT the same "
                      "as 'no hand in frame'")
            return None
        if self._frame_failures:
            print(f"\n[INFO] GetImageSample recovered after {self._frame_failures} failure(s)")
        self._frame_failures = 0
        return cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)

    def poll(self) -> bool:
        """Poll one camera frame, push into the debounce window, and return
        whether a hand was seen in THIS frame (raw, not debounced -- use
        hand_present() for the debounced/cooldown-gated trigger signal)."""
        frame_bgr = self._grab_frame_bgr()
        if frame_bgr is None:
            self._recent.append(False)
            if self._show:
                self._show_frame(None, None)
            return False
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)
        detected = len(result.hand_landmarks) > 0
        self._last_landmarks = result.hand_landmarks[0] if detected else None
        self._recent.append(detected)
        if self._show:
            self._show_frame(frame_bgr, self._last_landmarks)
        return detected

    def _show_frame(self, frame_bgr, landmarks):
        """Draw the detected hand (if any) on frame_bgr and display it in a
        cv2 window -- needs a display (X11 forwarding or similar over SSH).
        Only called when show=True. Non-fatal: a missing frame just skips
        the draw rather than raising, since this is a debug/visualization
        aid, not part of the control path."""
        if frame_bgr is None:
            return
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]
        if landmarks:
            pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
            for a, b in _HAND_CONNECTIONS:
                cv2.line(vis, pts[a], pts[b], (0, 255, 0), 2)
            for x, y in pts:
                cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
            cx = int(sum(x for x, _ in pts) / len(pts))
            cy = int(sum(y for _, y in pts) / len(pts))
            cv2.drawMarker(vis, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(vis, "HAND DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "searching...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)
        cv2.imshow(self._window_name, vis)
        cv2.waitKey(1)

    def hand_center_and_size(self) -> Optional[Tuple[float, float, float]]:
        """Normalized (cx, cy, size) for the hand seen on the last poll(),
        or None if that poll saw no hand.

        cx, cy: centroid of the 21 hand landmarks, in [0, 1] image
        coordinates (0,0 = top-left).
        size: wrist-to-middle-fingertip normalized distance (landmarks 0
        and 12) -- grows as the hand gets closer to the camera. Scale-
        invariant only in the loose sense of "bigger means closer"; not a
        metric distance (see module docstring).
        """
        lm = self._last_landmarks
        if not lm:
            return None
        cx = sum(p.x for p in lm) / len(lm)
        cy = sum(p.y for p in lm) / len(lm)
        wrist, mid_tip = lm[0], lm[12]
        size = float(((mid_tip.x - wrist.x) ** 2 + (mid_tip.y - wrist.y) ** 2) ** 0.5)
        return (float(cx), float(cy), size)

    def hand_present(self) -> bool:
        """True once >= min_detections of the last `window` polls saw a
        hand AND the post-trigger cooldown has elapsed."""
        if time.time() < self._cooldown_until:
            return False
        return sum(self._recent) >= self._min_detections

    def arm_cooldown(self):
        """Call after acting on a hand_present()==True trigger so the same
        hand still in frame doesn't immediately re-trigger the next poll."""
        self._cooldown_until = time.time() + self._cooldown_s
        self._recent.clear()

    def close(self):
        self._landmarker.close()
        if self._show:
            cv2.destroyWindow(self._window_name)


def main():
    """Standalone dry-run: detection only, no robot motion. Prints when a
    debounced hand-present trigger would fire. Run this before wiring the
    detector into g1_hw_handshake.py, per safety_config.json's verification
    notes.

    Usage: python3 hand_detector.py [network_interface] [--show]
    """
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    show = "--show" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    iface = positional[0] if positional else None
    if iface:
        ChannelFactoryInitialize(0, iface)
    else:
        ChannelFactoryInitialize(0)

    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()

    detector = HandPresenceDetector(client, show=show)
    print("[INFO] watching for a hand in frame (Ctrl+C to stop, no robot motion sent)...")
    try:
        while True:
            detector.poll()
            if detector.hand_present():
                print(f"[TRIGGER] hand detected and debounced at t={time.time():.1f}")
                detector.arm_cooldown()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        detector.close()


if __name__ == "__main__":
    main()
