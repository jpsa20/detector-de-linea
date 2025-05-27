from setuptools import setup, find_packages

package_name = 'puzzlebot_line_follower'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # 1) Para que ROS2 descubra el package
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # 2) package.xml
        ('share/' + package_name, ['package.xml']),
        # 3) launch files
        ('share/' + package_name + '/launch', [
            'launch/line_follow.launch.py',
            'launch/sim.launch.py'
        ]),
        # 4) parámetros
        ('share/' + package_name + '/params', [
            'params/line_follower_params.yaml'
        ]),
        # 5) controladores ROS2_Control
        ('share/' + package_name + '/config', [
            'config/controllers.yaml'
        ]),
        # 6) mundo de Gazebo
        ('share/' + package_name + '/worlds', [
            'worlds/track.world'
        ]),
        # 7) modelo YOLOv8
        ('share/' + package_name + '/models', [
            'models/lastfinal.pt'
        ]),
        # 8) scripts adicionales (semáforo dinámico)
        ('share/' + package_name + '/scripts', [
            'scripts/traffic_light_controller.py'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jp',
    maintainer_email='jp@todo.todo',
    description='Line follower for Puzzlebot using ROS 2, OpenCV and YOLOv8',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'line_light_detector       = puzzlebot_line_follower.line_light_detector:main',
            'yolov8_detector           = puzzlebot_line_follower.yolov8_detector:main',
            'line_follower_controller  = puzzlebot_line_follower.line_follower_controller:main',
            'traffic_light_controller  = puzzlebot_line_follower.traffic_light_controller:main',
        ],
    },
)
