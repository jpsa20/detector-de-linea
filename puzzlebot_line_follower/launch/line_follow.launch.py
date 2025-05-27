from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('puzzlebot_line_follower')
    params = os.path.join(pkg, 'params', 'line_follower_params.yaml')

    return LaunchDescription([
        Node(
            package='puzzlebot_line_follower',
            executable='line_light_detector',
            name='line_light_detector',
            parameters=[params],
            output='screen'
        ),
        Node(
            package='puzzlebot_line_follower',
            executable='yolov8_detector',
            name='yolov8_detector',
            parameters=[params],
            output='screen'
        ),
        Node(
            package='puzzlebot_line_follower',
            executable='line_follower_controller',
            name='line_follower_controller',
            parameters=[params],
            output='screen'
        ),
    ])
