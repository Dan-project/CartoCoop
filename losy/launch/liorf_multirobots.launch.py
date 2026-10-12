import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace
from ament_index_python.packages import get_package_share_directory

# Liste des robots à lancer — il suffit de modifier cette liste
ROBOTS = ['scout','ranger']

def generate_launch_description():
    pkg_share = get_package_share_directory('liorf')

    robot_groups = []
    for robot in ROBOTS:
        launch_path = os.path.join(pkg_share, 'launch', f'run_{robot}_multirobots.launch.py')

        robot_groups.append(
            GroupAction(
                actions=[
                    PushRosNamespace(robot),
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(launch_path)
                    )
                ]
            )
        )

    return LaunchDescription(robot_groups)

