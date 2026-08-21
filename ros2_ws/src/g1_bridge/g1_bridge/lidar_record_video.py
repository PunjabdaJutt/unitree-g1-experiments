"""Render the G1's Livox Mid-360 lidar (/utlidar/cloud_livox_mid360, PointCloud2) to an MP4.

Point clouds aren't natively "video" -- each incoming frame is rendered as a
top-down (bird's-eye) scatter plot, colored by height, and encoded into an
MP4 with OpenCV. This is for offline inspection, not live viewing: matplotlib
rendering is far slower than the lidar's actual publish rate, so frames are
dropped rather than queued. Use RViz (add /utlidar/cloud_livox_mid360 as a
PointCloud2 display) for a live 3D view instead.

Run (inside the ROS2 container/workspace):
  ros2 run g1_bridge lidar_record_video --ros-args \
      -p output:=/tmp/lidar.mp4 -p max_frames:=300 -p range_m:=15.0
"""
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

FRAME_PX = 800  # square output resolution, pixels


class LidarVideoRecorder(Node):

    def __init__(self):
        super().__init__('g1_lidar_video_recorder')

        self.declare_parameter('output', f'/output/lidar_{int(time.time())}.mp4')
        self.declare_parameter('topic', '/utlidar/cloud_livox_mid360')
        self.declare_parameter('max_frames', 300)
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('range_m', 15.0)

        self.output_path = self.get_parameter('output').value
        topic = self.get_parameter('topic').value
        self.max_frames = self.get_parameter('max_frames').value
        self.fps = self.get_parameter('fps').value
        self.range_m = self.get_parameter('range_m').value

        self.frame_count = 0
        self._finished = False
        self.writer = cv2.VideoWriter(
            self.output_path, cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, (FRAME_PX, FRAME_PX))

        self.fig, self.ax = plt.subplots(figsize=(FRAME_PX / 100, FRAME_PX / 100), dpi=100)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(PointCloud2, topic, self._on_cloud, qos)

        self.get_logger().info(
            f'Recording {topic} -> {self.output_path} '
            f'(max {self.max_frames} frames @ {self.fps} fps)')

    def _on_cloud(self, msg: PointCloud2):
        if self._finished:
            return

        points = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if points.shape[0] > 0:
            self.ax.clear()
            self.ax.set_xlim(-self.range_m, self.range_m)
            self.ax.set_ylim(-self.range_m, self.range_m)
            self.ax.set_facecolor('black')
            self.ax.scatter(points['x'], points['y'], c=points['z'], s=1,
                             cmap='viridis', vmin=-1.0, vmax=2.0)
            self.ax.set_aspect('equal')
            self.ax.axis('off')

            self.fig.canvas.draw()
            buf = np.asarray(self.fig.canvas.buffer_rgba())
            frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
            frame_bgr = cv2.resize(frame_bgr, (FRAME_PX, FRAME_PX))
            self.writer.write(frame_bgr)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'{self.frame_count}/{self.max_frames} frames')

        if self.frame_count >= self.max_frames:
            self._finish()
            if rclpy.ok():
                rclpy.shutdown()

    def _finish(self):
        if self._finished:
            return
        self._finished = True
        self.writer.release()
        plt.close(self.fig)
        self.get_logger().info(f'Done -- wrote {self.frame_count} frames to {self.output_path}')


def main(args=None):
    rclpy.init(args=args)
    node = LidarVideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
