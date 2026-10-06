"""teleop_loop.TeleopLoop: ROS 없이 리더 관절 → 팔로워 관절 명령 (팔로워도 so102.urdf로 대역)."""
import os

import numpy as np
import pytest
import yaml
from conftest import PKG_ROOT, SO102_URDF

from weld_teleop.follower_control import FollowerStepper
from weld_teleop.kinematics import ChainKinematics
from weld_teleop.mapping import EeMapper
from weld_teleop.teleop_loop import TeleopLoop, build_loop, decode, encode

ARM = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll']
QF = np.array([0.0, -0.5, 0.8, 0.3, 0.0, 0.0])   # 팔로워 시작 자세 (특이점 회피)
QL = np.array([0.1, -0.3, 0.4, 0.2, -0.1, 0.3])
DT = 0.01


@pytest.fixture(scope='module')
def kin():
    return ChainKinematics(SO102_URDF, ARM, 'gripper_frame_link')


def make(kin):
    return TeleopLoop(kin, kin, EeMapper(max_linear_step=0.01, max_angular_step=0.1,
                                         workspace_min=(-1, -1, -1), workspace_max=(1, 1, 1)),
                      FollowerStepper(kin, max_joint_step=0.05, input_timeout=0.2))


def run(loop, q_leader_fn, steps, t0=0.0, clutch=True):
    """팔로워가 명령을 그대로 따라간다고 보고(완벽 추종) steps 주기 실행."""
    out = None
    for i in range(steps):
        t = t0 + i * DT
        q_meas = loop.stepper.q_cmd if loop.stepper.q_cmd is not None else QF
        loop.on_measured(q_meas, t)
        q = loop.tick(t, q_leader_fn(i), clutch)
        out = q if q is not None else out
    return out


def test_clutch_off_sends_nothing(kin):
    loop = make(kin)
    loop.on_measured(QF, 0.0)
    assert loop.tick(0.0, QL, clutch=False) is None
    assert loop.status == 'clutch_off'


def test_waits_for_measured(kin):
    loop = make(kin)
    assert loop.tick(0.0, QL, clutch=True) is None
    assert loop.status == 'waiting'


def test_follower_copies_leader_relative_motion(kin):
    loop = make(kin)
    dq = np.array([0.0, 0.0, 0.0, 0.15, 0.0, 0.0])   # 리더 손목만 움직임
    run(loop, lambda i: QL if i < 3 else QL + dq, 200)
    leader_delta = kin.fk(QL + dq).translation - kin.fk(QL).translation
    follower_delta = kin.fk(loop.stepper.q_cmd).translation - kin.fk(QF).translation
    assert np.linalg.norm(leader_delta) > 0.01
    assert np.allclose(follower_delta, leader_delta, atol=2e-3)


def test_clutch_reengage_does_not_jump(kin):
    loop = make(kin)
    run(loop, lambda i: QL, 5)
    q_before = loop.stepper.q_cmd.copy()
    run(loop, lambda i: QL + 0.3, 20, t0=1.0, clutch=False)   # 클러치 OFF 동안 리더 크게 이동
    run(loop, lambda i: QL + 0.3, 20, t0=2.0)                  # 다시 ON: 그 자세가 새 기준
    assert np.allclose(loop.stepper.q_cmd, q_before, atol=1e-6)


def test_leader_lost_stops_and_reanchors(kin):
    loop = make(kin)
    run(loop, lambda i: QL, 5)
    loop.on_measured(loop.stepper.q_cmd, 1.0)
    assert loop.tick(1.0, None, clutch=True) is None
    assert loop.status == 'leader_lost' and loop.mapper.anchor is None


def test_stale_measurement_stops(kin):
    loop = make(kin)
    run(loop, lambda i: QL, 5)
    assert loop.tick(5.0, QL + 0.1, clutch=True) is None   # 실측이 5초 전
    assert loop.status == 'joint_state_timeout'
    assert loop.mapper.anchor is None


def test_udp_message_roundtrip_and_rejects_bad_packets():
    q = np.array([0.1, -0.2, 0.3, 0.0, 1.0, -1.0])
    assert np.allclose(decode(encode(q), 6)['q'], q)
    assert decode(encode(q), 7) is None
    assert decode(b'not json', 6) is None
    assert decode(b'{"x": 1}', 6) is None
    assert decode(encode([0, 0, 0, 0, 0, float('nan')]), 6) is None


def test_build_loop_from_package_configs(tmp_path):
    cfg = os.path.join(PKG_ROOT, 'config')
    fo = yaml.safe_load(open(os.path.join(cfg, 'follower_opti3.yaml'), encoding='utf-8'))
    p = fo['follower_ik_node']['ros__parameters']
    p['joint_names'], p['ee_frame'] = ARM, 'gripper_frame_link'   # opti3.urdf 대신 so102로 대역
    f = tmp_path / 'follower.yaml'
    f.write_text(yaml.safe_dump(fo))
    loop = build_loop(os.path.join(cfg, 'leader_so101_7axis.yaml'),
                      os.path.join(cfg, 'mapping.yaml'), str(f), SO102_URDF, SO102_URDF)
    assert loop.leader_kin.nq == 6 and loop.stepper.timeout == p['input_timeout']
