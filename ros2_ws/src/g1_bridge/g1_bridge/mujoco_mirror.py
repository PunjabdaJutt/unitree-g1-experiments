"""Live MuJoCo visualization of the real G1, driven by /joint_states.

Kinematics-only mirror: on every /joint_states message this writes the
reported joint angles into MuJoCo's qpos and calls mj_forward -- no
mj_step, no gravity, no contacts. The scene always shows the robot's actual
current pose, not a simulated one. This is a MuJoCo-rendered alternative to
bridge.launch.py's RViz view.

/joint_states has two independent publishers, both consumed here purely by
joint name so it doesn't matter which one a given name came from:
joint_state_bridge.py (29 body joints, from the robot's DDS /lowstate) and
inspire_hand_modbus_bridge.py (24 Inspire hand joints, read directly from
the hands over Modbus TCP -- they don't appear on this robot's DDS bus).

Run (inside the ROS2 container, after colcon build + source install/setup.bash):
  ros2 run g1_bridge mujoco_mirror
"""
import os
import re
import tempfile
from pathlib import Path

import mujoco
import mujoco.viewer
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState

# g1_description's URDF references meshes as package://g1_description/meshes/...,
# which ROS resolves but MuJoCo's URDF compiler does not -- it only honors the
# <mujoco><compiler meshdir="meshes"/> hint already declared in the URDF,
# relative to the URDF's own directory. Rewrite both to point at an absolute
# meshdir so the patched copy can be compiled from a tempfile.
_PACKAGE_URI_RE = re.compile(r'package://g1_description/meshes/')
_MESHDIR_RE = re.compile(r'meshdir="meshes"')


def _load_model(urdf_path: Path) -> mujoco.MjModel:
    meshes_dir = urdf_path.parent.parent / 'meshes'
    text = _PACKAGE_URI_RE.sub('', urdf_path.read_text())
    text = _MESHDIR_RE.sub(f'meshdir="{meshes_dir}"', text)

    fd, tmp_path = tempfile.mkstemp(suffix='.urdf')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        return mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)


class MujocoMirror(Node):

    def __init__(self):
        super().__init__('g1_mujoco_mirror')

        default_urdf = str(Path(get_package_share_directory('g1_description'))
                            / 'urdf' / 'g1_29dof_with_inspire_hand.urdf')
        self.declare_parameter('urdf_path', default_urdf)
        urdf_path = Path(self.get_parameter('urdf_path').value)

        self.model = _load_model(urdf_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Resolved lazily per joint name on first sighting, since JointState
        # doesn't guarantee its name order matches MuJoCo's qpos layout.
        self._qpos_adr = {}

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.create_subscription(JointState, 'joint_states', self._on_joint_states, 10)

        self.get_logger().info('MuJoCo mirror ready, waiting for /joint_states')

    def _qpos_index(self, name: str):
        if name not in self._qpos_adr:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._qpos_adr[name] = int(self.model.jnt_qposadr[jid]) if jid >= 0 else None
        return self._qpos_adr[name]

    def _on_joint_states(self, msg: JointState):
        if not self.viewer.is_running():
            return
        for name, pos in zip(msg.name, msg.position):
            adr = self._qpos_index(name)
            if adr is not None:
                self.data.qpos[adr] = pos
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoMirror()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
