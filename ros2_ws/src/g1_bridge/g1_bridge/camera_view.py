"""Live view + optional recording of the G1's front camera (/frontvideostream).

Decodes the JPEG-compressed frame from each Go2FrontVideoData message and
shows it in a live OpenCV window. Press 'r' at any time to start recording
to an MP4; press it again to stop and save. Recordings are timestamped by
wall-clock arrival time (not a fixed fps), so playback speed matches real
time regardless of how fast frames actually arrive.

Needs an X server on the host (same xhost/DISPLAY setup as RViz -- see
INSTALL.md). Output directory must be a host-mounted volume (see
docker-compose.yml's ./output:/output mount) or recordings vanish when the
container exits.

Run (inside the ROS2 container/workspace):
  ros2 run g1_bridge camera_view
  ros2 run g1_bridge camera_view --ros-args -p resolution:=360p -p output_dir:=/output
"""
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from unitree_go.msg import Go2FrontVideoData

RESOLUTION_FIELDS = {
    '720p': 'video720p',
    '360p': 'video360p',
    '180p': 'video180p',
}


class CameraView(Node):

    def __init__(self):
        super().__init__('g1_camera_view')

        self.declare_parameter('topic', '/frontvideostream')
        self.declare_parameter('resolution', '720p')
        self.declare_parameter('output_dir', '/output')

        topic = self.get_parameter('topic').value
        resolution = self.get_parameter('resolution').value
        self.output_dir = self.get_parameter('output_dir').value
        if resolution not in RESOLUTION_FIELDS:
            raise ValueError(f'resolution must be one of {list(RESOLUTION_FIELDS)}')
        self.field = RESOLUTION_FIELDS[resolution]

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(Go2FrontVideoData, topic, self._on_frame, qos)
        self.latest_frame = None

        self.recording = False
        self._rec_frames = []
        self._rec_start_time = None

        self.get_logger().info(
            f"Viewing {topic} ({resolution}) -- 'r' to start/stop recording, 'q' to quit")

    def _on_frame(self, msg: Go2FrontVideoData):
        jpeg_bytes = bytes(getattr(msg, self.field))
        if not jpeg_bytes:
            return
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self.latest_frame = frame
        if self.recording:
            self._rec_frames.append(frame.copy())

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self._rec_frames = []
            self._rec_start_time = time.time()
            self.get_logger().info('Recording started')
        else:
            self.recording = False
            self._save_recording()

    def _save_recording(self):
        elapsed = time.time() - self._rec_start_time
        n = len(self._rec_frames)
        if n == 0 or elapsed <= 0:
            self.get_logger().warn('Recording stopped -- no frames captured, nothing saved')
            return

        fps = n / elapsed
        h, w = self._rec_frames[0].shape[:2]
        path = f'{self.output_dir}/camera_{int(self._rec_start_time)}.mp4'
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in self._rec_frames:
            writer.write(f)
        writer.release()
        self._rec_frames = []
        self.get_logger().info(
            f'Recording stopped -- saved {n} frames ({elapsed:.1f}s @ {fps:.1f}fps) to {path}')


def main(args=None):
    rclpy.init(args=args)
    node = CameraView()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.latest_frame is not None:
                display = node.latest_frame
                if node.recording:
                    display = display.copy()
                    cv2.circle(display, (25, 25), 10, (0, 0, 255), -1)
                    cv2.putText(display, 'REC', (45, 33), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)
                cv2.imshow('G1 front camera', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                node.toggle_recording()
    except KeyboardInterrupt:
        pass
    finally:
        if node.recording:
            node.recording = False
            node._save_recording()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
