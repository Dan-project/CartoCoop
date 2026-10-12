"""
Lance un noeud de perception par robot (namespace = nom du robot) et un
seul noeud de fusion qui les agrege.

Pour du multi-bag : lancer ce launch, puis dans deux terminaux separes :
    ros2 bag play bag_scout/  --clock
    ros2 bag play bag_ranger/ --clock --remap ...  (si besoin de remapper
    les topics du bag vers le namespace /ranger)

Pour du reel : les memes noeuds de perception tournent embarques sur
chaque robot (memes parametres, meme namespace), le noeud de fusion peut
tourner sur une base au sol tant que le domaine DDS est partage.
"""
import os 
from launch import LaunchDescription
from launch_ros.actions import Node


ROBOTS = ['scout','ranger']


def generate_launch_description():
    nodes = []

    for robot in ROBOTS:
        nodes.append(Node(
            package='losy',
            executable='perception',
            name='perception',
            namespace=robot,
            output='screen',
            parameters=[{
                'robot': robot,
                'acquisition_hz': 2.0,
                'range': 20,
                'model_path': 'yolo26n-sem.pt',
                'output_topic': 'semantic_points',
                'loop': False,
            }],
        ))

    nodes.append(Node(
        package='losy',
        executable='map_fusion',
        name='map_fusion',
        output='screen',
        parameters=[{
            'robots': ROBOTS,
            'dim': '2d',
            'cell_size': 1,
            'write_interval_sec': 30.0,
            'loop': False,

        }],
    ))

    return LaunchDescription(nodes)