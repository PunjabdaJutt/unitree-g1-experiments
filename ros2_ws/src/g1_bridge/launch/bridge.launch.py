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

    urdf_path = PathJoinSubstitution([
        FindPackageShare('g1_description'), 'urdf', 'g1_29dof_with_inspire_hand.urdf',
    ])
    robot_description = Command(['cat ', urdf_path])

    rviz_config = PathJoinSubstitution([
        FindPackageShare('g1_description'), 'rviz', 'g1.rviz',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz alongside the bridge'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='g1_bridge',
            executable='joint_state_bridge',
            name='g1_joint_state_bridge',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
            condition=IfCondition(start_rviz),
        ),
    ])
