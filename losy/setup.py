from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'losy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='multirobots',
    maintainer_email='multirobots@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
           'tf_fusion = losy.lidar_camera_fusion_tf:main',
           'odom_fusion = losy.lidar_camera_fusion_odom:main',
           'parseur =  losy.parseur:main',
           'projection_kitti = losy.projection_kitti:main',
           'projection_ranger = losy.projection_ranger:main',
           'projection_scout = losy.projection_scout:main',
           'calib_ranger = losy.calibration_geo_ranger:main',
           'calib_scout = losy.calibration_geo_scout:main',
           'projection_map = losy.projection_map_apriori:main',
           'multi = losy.projection_multi:main',
           'perception = losy.perception:main',
           'map_fusion = losy.map_fusion:main',
           'tf_prefixer =  losy.tf_prefixer:main',


        ],
    },
)
