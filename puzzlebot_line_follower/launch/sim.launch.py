#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command

def generate_launch_description():
    # Paquetes
    desc_pkg     = get_package_share_directory('puzzlebot_description')
    follower_pkg = get_package_share_directory('puzzlebot_line_follower')

    # Rutas
    xacro_file       = os.path.join(desc_pkg, 'urdf', 'puzzlebot.urdf.xacro')
    controllers_file = os.path.join(follower_pkg, 'config', 'controllers.yaml')
    params_file      = os.path.join(follower_pkg, 'params', 'line_follower_params.yaml')
    world_file       = os.path.join(follower_pkg, 'worlds', 'track.world')

    # Generar robot_description
    robot_desc = {'robot_description': Command(['xacro ', xacro_file])}

    return LaunchDescription([
        # 1) Gazebo con world
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', world_file],
            output='screen'
        ),

        # 2) Joint + TF publishers
        Node(package='joint_state_publisher',
             executable='joint_state_publisher',
             name='joint_state_publisher'),
        Node(package='robot_state_publisher',
             executable='robot_state_publisher',
             name='robot_state_publisher',
             parameters=[robot_desc]),

        # 3) Spawn Puzzlebot
        Node(package='gazebo_ros',
             executable='spawn_entity.py',
             arguments=['-topic', 'robot_description',
                        '-entity', 'puzzlebot'],
             output='screen'),

        # 4) ros2_control_node
        Node(package='controller_manager',
             executable='ros2_control_node',
             parameters=[robot_desc, controllers_file],
             output='screen'),

        # 5) Spawners
        Node(package='controller_manager',
             executable='spawner',
             arguments=['joint_state_broadcaster',
                        '--controller-manager', '/controller_manager']),
        Node(package='controller_manager',
             executable='spawner',
             arguments=['diff_drive_controller',
                        '--controller-manager', '/controller_manager']),

        # 6) Tus nodos visión & control
        Node(package='puzzlebot_line_follower',
             executable='line_light_detector',
             name='line_light_detector',
             parameters=[params_file],
             output='screen'),
        Node(package='puzzlebot_line_follower',
             executable='yolov8_detector',
             name='yolov8_detector',
             parameters=[params_file],
             output='screen'),
        Node(package='puzzlebot_line_follower',
             executable='line_follower_controller',
             name='line_follower_controller',
             parameters=[params_file],
             output='screen'),
    ])
