from glob import glob

from setuptools import find_packages, setup

package_name = 'weld_teleop'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'leader_state_node = weld_teleop.leader_state_node:main',
            'leader_fk_node = weld_teleop.leader_fk_node:main',
            'ee_mapping_node = weld_teleop.ee_mapping_node:main',
            'follower_ik_node = weld_teleop.follower_ik_node:main',
        ],
    },
)
