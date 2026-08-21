"""Live G1 visualization: joint_state_bridge (reads the real robot) + robot_state_publisher + RViz.

Requires the container to actually be on the robot's network (see
docker-compose.yml, network_mode: host + cyclonedds.xml pinned to the
robot-facing interface).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_rviz = LaunchConfiguration('rviz')
    start_mujoco = LaunchConfiguration('mujoco')
    start_hands = LaunchConfiguration('hands')
    right_hand_ip = LaunchConfiguration('right_hand_ip')
    left_hand_ip = LaunchConfiguration('left_hand_ip')

    urdf_path = PathJoinSubstitution([
        FindPackageShare('g1_description'), 'urdf', 'g1_29dof_with_inspire_hand.urdf',
    ])
    robot_description = Command(['cat ', urdf_path])

    rviz_config = PathJoinSubstitution([
        FindPackageShare('g1_description'), 'rviz', 'g1.rviz',
    ])

    # network_mode: host puts every node in this launch file on the robot's
    # full DDS domain (not just the topics they subscribe to), and the
    # robot's onboard firmware publishes hundreds of raw DDS topics that
    # never went through rclpy -- rmw_cyclonedds_cpp logs one WARN per such
    # topic it discovers ("Failed to parse type hash ... USER_DATA '(null)'"),
    # which is harmless but drowns out real output. Quiet that logger only.
    quiet_rmw_args = ['--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error']

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz alongside the bridge'),
        DeclareLaunchArgument('mujoco', default_value='false', description='Launch the MuJoCo mirror alongside the bridge'),
        DeclareLaunchArgument('hands', default_value='false',
                               description='Poll the Inspire hands over Modbus TCP and publish their joints '
                                            '(the hands are not on this robot\'s DDS bus)'),
        DeclareLaunchArgument('right_hand_ip', default_value='192.168.123.211',
                               description='Confirmed by observation against the physical robot -- see inspire_hand_modbus_bridge.py'),
        DeclareLaunchArgument('left_hand_ip', default_value='192.168.123.210',
                               description='Confirmed by observation against the physical robot -- see inspire_hand_modbus_bridge.py'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            arguments=quiet_rmw_args,
        ),
        Node(
            package='g1_bridge',
            executable='joint_state_bridge',
            name='g1_joint_state_bridge',
            output='screen',
            arguments=quiet_rmw_args,
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config] + quiet_rmw_args,
            output='screen',
            condition=IfCondition(start_rviz),
        ),
        Node(
            package='g1_bridge',
            executable='mujoco_mirror',
            name='g1_mujoco_mirror',
            output='screen',
            arguments=quiet_rmw_args,
            condition=IfCondition(start_mujoco),
        ),
        Node(
            package='g1_bridge',
            executable='inspire_hand_modbus_bridge',
            name='g1_inspire_hand_modbus_bridge',
            output='screen',
            parameters=[{'right_hand_ip': right_hand_ip, 'left_hand_ip': left_hand_ip}],
            arguments=quiet_rmw_args,
            condition=IfCondition(start_hands),
        ),
    ])
