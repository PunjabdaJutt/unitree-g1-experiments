"""Standalone URDF viewer: robot_state_publisher + RViz, with sliders for manual joint testing.

For live robot data instead of sliders, use g1_bridge/launch/bridge.launch.py.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_joint_state_gui = LaunchConfiguration('use_joint_state_gui')

    urdf_path = PathJoinSubstitution([
        FindPackageShare('g1_description'), 'urdf', 'g1_29dof_with_inspire_hand.urdf',
    ])
    robot_description = Command(['cat ', urdf_path])

    rviz_config = PathJoinSubstitution([
        FindPackageShare('g1_description'), 'rviz', 'g1.rviz',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_joint_state_gui', default_value='true',
            description='Launch joint_state_publisher_gui sliders (disable when a bridge node supplies /joint_states)',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            condition=IfCondition(use_joint_state_gui),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])
