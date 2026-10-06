"""config/*.yaml 정합성: 관절 수·서보 ID·URDF 관절 이름이 서로 맞는지."""
import os

import pytest
import yaml
from conftest import PKG_ROOT, SO102_URDF

from weld_teleop.kinematics import ChainKinematics

CFG = os.path.join(PKG_ROOT, 'config')


def load(name):
    with open(os.path.join(CFG, name), encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize('name', ['leader_so101_7axis.yaml', 'leader_so101_5dof.yaml'])
def test_leader_profile_lengths_match(name):
    c = load(name)
    st = c['leader_state_node']['ros__parameters']
    n = len(st['joint_names'])
    assert len(st['servo_ids']) == len(st['offsets']) == len(st['signs']) == n
    assert len(set(st['servo_ids'])) == n
    assert st['aux_servo_id'] not in st['servo_ids']
    assert c['leader_fk_node']['ros__parameters']['joint_names'] == st['joint_names']


def test_so102_profile_matches_urdf_and_probe_result():
    c = load('leader_so101_7axis.yaml')
    st = c['leader_state_node']['ros__parameters']
    # 2026-10-06 실기 확인: pan=1 … roll=6, gripper=7
    assert dict(zip(st['joint_names'], st['servo_ids'])) == {
        'shoulder_pan': 1, 'shoulder_lift': 2, 'elbow_flex': 3,
        'wrist_flex': 4, 'wrist_yaw': 5, 'wrist_roll': 6}
    assert st['aux_servo_id'] == 7
    assert st['driver'] != 'lerobot'   # lerobot SO101Leader는 wrist_yaw가 없음
    fk = c['leader_fk_node']['ros__parameters']
    ChainKinematics(SO102_URDF, fk['joint_names'], fk['ee_frame'])  # 이름이 URDF에 있어야 함
