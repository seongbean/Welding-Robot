"""리더(SO-101) → opti3 엔드이펙터 텔레오퍼레이션

예)
  # 하드웨어 없이 전체 파이프라인 확인 (mock 리더 + loopback 팔로워)
  ros2 launch weld_teleop teleop.launch.py leader_profile:=5dof test_mode:=true \
      leader_urdf:=/path/to/so101.urdf

  # 실기 (dry_run은 follower_opti3.yaml에서 false로 바꾼 뒤)
  ros2 launch weld_teleop teleop.launch.py leader_profile:=7axis   # urdf/so102.urdf 자동 사용
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    share = get_package_share_directory('weld_teleop')
    cfg = lambda f: os.path.join(share, 'config', f)  # noqa: E731
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731

    leader_yaml = cfg(f'leader_so101_{lc("leader_profile")}.yaml')
    test = lc('test_mode').lower() == 'true'

    profile = lc('leader_profile')
    leader_urdf = lc('leader_urdf') or (
        os.path.join(share, 'urdf', 'so102.urdf') if profile == '7axis' else '')
    if not leader_urdf:
        raise RuntimeError('leader_profile:=5dof 에서는 leader_urdf:=.../so101_new_calib.urdf 를 지정하세요')

    follower_urdf = lc('follower_urdf') or os.path.join(
        get_package_share_directory('indy_description'), 'urdf_files', 'opti3.urdf')

    return [
        Node(package='weld_teleop', executable='leader_state_node',
             parameters=[leader_yaml] + ([{'driver': 'mock'}] if test else [])),
        Node(package='weld_teleop', executable='leader_fk_node',
             parameters=[leader_yaml, {'urdf_path': leader_urdf}]),
        Node(package='weld_teleop', executable='ee_mapping_node',
             parameters=[cfg('mapping.yaml')] + ([{'auto_clutch': True}] if test else [])),
        Node(package='weld_teleop', executable='follower_ik_node',
             parameters=[cfg('follower_opti3.yaml'), {'urdf_path': follower_urdf}]
             + ([{'loopback': True}] if test else [])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('leader_profile', default_value='7axis',
                              description='5dof | 7axis'),
        DeclareLaunchArgument('leader_urdf', default_value='',
                              description='리더 URDF 절대경로 (7axis는 비우면 패키지의 so102.urdf)'),
        DeclareLaunchArgument('follower_urdf', default_value='',
                              description='비우면 indy_description의 opti3.urdf'),
        DeclareLaunchArgument('test_mode', default_value='false',
                              description='true: mock 리더 + 자동 클러치 + loopback 팔로워'),
        OpaqueFunction(function=_setup),
    ])
