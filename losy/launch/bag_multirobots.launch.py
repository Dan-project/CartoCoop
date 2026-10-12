import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    bag_path_scout = '/home/multirobots/multirobots_ws/bag_scout'
    bag_path_ranger = '/home/multirobots/multirobots_ws/bag'
    qos_path = '/home/multirobots/multirobots_ws/qos_override.yaml'

    play_scout_bag = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', bag_path_scout,
            '--clock',
            # '--start-offset', '60.0',  # si on veut differe
            '--qos-profile-overrides-path', qos_path,
            '--remap',
            '/camera/camera/color/image_raw:=/scout/camera/color/image_raw',
            '/ouster/points:=/scout/ouster/points',
            '/ouster/imu:=/scout/ouster/imu',
            '/fix:=/scout/fix',
            '/tf:=/scout/tf_raw',
            '/tf_static:=/scout/tf_static_raw',
        ],
        output='screen',
    )

    play_ranger_bag = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', bag_path_ranger,
            # '--clock',
            '--qos-profile-overrides-path', qos_path,
            '--remap',
            '/ugv1/camera/color/image_raw:=/ranger/camera/color/image_raw',
            '/ugv1/ouster/points:=/ranger/ouster/points',
            '/ugv1/ouster/imu:=/ranger/ouster/imu',
            '/ugv1/fix:=/ranger/fix',
            '/tf:=/ranger/tf_raw',
            '/tf_static:=/ranger/tf_static_raw',
        ],
        output='screen',
    )

    tf_prefix_scout = Node(
        package='losy',
        executable='tf_prefixer',
        name='tf_prefixer_scout',
        parameters=[{
            'prefix': 'scout',
            'input_tf_topic': '/scout/tf_raw',
            'input_tf_static_topic': '/scout/tf_static_raw',
        }],
        output='screen',
    )

    tf_prefix_ranger = Node(
        package='losy',
        executable='tf_prefixer',
        name='tf_prefixer_ranger',
        parameters=[{
            'prefix': 'ranger',
            'input_tf_topic': '/ranger/tf_raw',
            'input_tf_static_topic': '/ranger/tf_static_raw',
        }],
        output='screen',
    )

    return LaunchDescription([
        play_scout_bag,
        play_ranger_bag,
        tf_prefix_scout,
        tf_prefix_ranger,
    ])