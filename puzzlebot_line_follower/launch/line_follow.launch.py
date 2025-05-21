#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('puzzlebot_line_follower')
    params_file = os.path.join(pkg_share, 'params', 'line_follower_params.yaml')

    return LaunchDescription([
        # Nodo de visión: solo detección de línea
        Node(
            package='puzzlebot_line_follower',
            executable='line_light_detector',
            name='line_light_detector',
            output='screen',
            parameters=[params_file]
        ),
        # Nodo de control: PID básico sin semáforo
        Node(
            package='puzzlebot_line_follower',
            executable='line_follower_controller',
            name='line_follower_controller',
            output='screen',
            parameters=[params_file]
        ),
    ])
