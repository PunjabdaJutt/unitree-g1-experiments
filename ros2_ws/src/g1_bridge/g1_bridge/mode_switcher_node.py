"""ROS2 front-end for FSM/mode switching on the G1.

Deliberately does NOT import unitree_sdk2py in this process: importing it
inside a ROS2-sourced shell hits a libddsc ABI conflict between ROS2's
CycloneDDS RMW and the SDK's own bundled CycloneDDS, crashing at
ChannelFactoryInitialize. Instead this node shells out to
modes/g1_mode_switcher.py's one-shot mode (`--iface IFACE --mode N`) in a
subprocess with a clean, non-ROS2-sourced environment, and reports the
result back over ROS2.
"""
import os
import subprocess

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

# modes/g1_mode_switcher.py MODES keys -> service name suffix. Keep this in
# sync with that file's MODES dict -- it was previously stale (e.g.
# 'locked_stand' pointed at key '5', which is actually exit_dev_mode) after
# that script's modes were trimmed down to 0-5 only.
MODE_KEYS = {
    'zero_torque': '0',
    'damp': '1',
    'sit': '2',
    'locked_stand': '3',
    'enter_dev_mode': '4',
    'exit_dev_mode': '5',
}

SWITCHER_SCRIPT = os.environ.get(
    'G1_MODE_SWITCHER_SCRIPT',
    os.path.expanduser('~/unitree-g1-experiments/.unitree_sdk2_python_src/modes/g1_mode_switcher.py'),
)


class ModeSwitcherNode(Node):

    def __init__(self):
        super().__init__('g1_mode_switcher_node')
        self.declare_parameter('network_interface', 'eth0')
        self.iface = self.get_parameter('network_interface').get_parameter_value().string_value

        # Strip ROS2's env vars (AMENT_PREFIX_PATH, CMAKE_PREFIX_PATH,
        # RMW_IMPLEMENTATION, ...) so the subprocess's unitree_sdk2py import
        # doesn't see the ROS2-sourced CycloneDDS.
        self.clean_env = {
            k: v for k, v in os.environ.items()
            if not k.startswith(('ROS_', 'AMENT_', 'CMAKE_', 'RMW_', 'CYCLONEDDS_'))
        }

        self._mode_services = [
            self.create_service(Trigger, f'/g1/mode/{name}',
                                 self._make_handler(key))
            for name, key in MODE_KEYS.items()
        ]
        self.get_logger().info(
            f"g1_mode_switcher_node ready on iface={self.iface}; "
            f"services: {', '.join(f'/g1/mode/{n}' for n in MODE_KEYS)}"
        )

    def _make_handler(self, mode_key):
        def handler(request, response):
            cmd = ['python3', SWITCHER_SCRIPT, '--iface', self.iface, '--mode', mode_key]
            try:
                result = subprocess.run(
                    cmd, env=self.clean_env, capture_output=True, text=True, timeout=30,
                )
                response.success = (result.returncode == 0)
                response.message = (result.stdout + result.stderr)[-800:]
            except subprocess.TimeoutExpired:
                response.success = False
                response.message = 'mode switch timed out after 30s'
            return response
        return handler


def main():
    rclpy.init()
    node = ModeSwitcherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
