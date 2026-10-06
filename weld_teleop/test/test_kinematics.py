"""kinematics.ChainKinematics: so102.urdf(리더, 팔 6관절)로 FK/IK 검증."""
import numpy as np
import pinocchio as pin
import pytest
from conftest import SO102_URDF

from weld_teleop.kinematics import (ChainKinematics, se3_from_xyz_rpy, se3_to_xyzquat,
                                    xyzquat_to_se3)

ARM = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll']
EE = 'gripper_frame_link'


@pytest.fixture(scope='module')
def kin():
    return ChainKinematics(SO102_URDF, ARM, EE)


def test_chain_locks_gripper(kin):
    # 그리퍼 관절은 잠기고 팔 6관절만 남는다
    assert kin.nq == 6
    assert kin.model.nq == 6
    assert np.all(kin.lower < kin.upper)


def test_config_order_is_respected():
    # config 순서가 URDF 순서와 달라도 같은 FK가 나와야 한다
    q = np.array([0.3, -0.4, 0.5, -0.2, 0.6, 0.1])
    a = ChainKinematics(SO102_URDF, ARM, EE).fk(q)
    perm = [5, 0, 3, 1, 4, 2]
    b = ChainKinematics(SO102_URDF, [ARM[i] for i in perm], EE).fk(q[perm])
    assert a.isApprox(b, 1e-12)


def test_tool_offset_applied_in_flange_frame():
    tool = se3_from_xyz_rpy([0.0, 0.0, 0.05], [0.0, 0.0, 0.0])
    k = ChainKinematics(SO102_URDF, ARM, EE, tool)
    q = np.array([0.2, 0.3, -0.4, 0.5, -0.3, 0.7])
    flange = k.fk_flange(q)
    tcp = k.fk(q)
    assert np.isclose(np.linalg.norm(tcp.translation - flange.translation), 0.05)
    assert np.allclose(tcp.translation, flange.act(np.array([0.0, 0.0, 0.05])))


def test_unknown_joint_or_frame_raises():
    with pytest.raises(ValueError):
        ChainKinematics(SO102_URDF, ARM[:-1] + ['nope'], EE)
    with pytest.raises(ValueError):
        ChainKinematics(SO102_URDF, ARM, 'nope_frame')


@pytest.mark.parametrize('seed', range(5))
def test_ik_recovers_reachable_pose(kin, seed):
    # 리밋 안의 임의 자세 → FK → 근처 seed에서 IK → 같은 포즈에 도달
    rng = np.random.default_rng(seed)
    q_true = rng.uniform(0.6 * kin.lower, 0.6 * kin.upper)
    target = kin.fk(q_true)
    q_seed = q_true + rng.normal(0.0, 0.05, kin.nq)
    q, ok, err = kin.ik(target, q_seed, iters=200)
    assert ok, f'IK 미수렴 err={err}'
    assert np.all(q >= kin.lower - 1e-9) and np.all(q <= kin.upper + 1e-9)
    assert kin.fk(q).isApprox(target, 1e-3)


def test_ik_respects_limits_for_unreachable_target(kin):
    # 작업영역 밖 목표: 수렴 실패를 알리고 리밋을 넘지 않아야 한다
    far = pin.SE3(np.eye(3), np.array([5.0, 0.0, 0.0]))
    q, ok, err = kin.ik(far, np.zeros(kin.nq), iters=50)
    assert not ok and err > 1.0
    assert np.all(q >= kin.lower - 1e-9) and np.all(q <= kin.upper + 1e-9)


def test_xyzquat_roundtrip():
    T = se3_from_xyz_rpy([0.1, -0.2, 0.3], [0.4, -0.5, 0.6])
    assert xyzquat_to_se3(se3_to_xyzquat(T)).isApprox(T, 1e-12)


def test_matches_independent_numpy_fk():
    # so102_urdf_check.py의 numpy FK(URDF 직접 파싱)와 pinocchio FK가 같은 TCP 위치를 내야 한다
    import so102_urdf_check as chk
    u = chk.Urdf(SO102_URDF)
    k = ChainKinematics(SO102_URDF, ARM, EE)
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.uniform(k.lower, k.upper)
        p_np = u.fk(dict(zip(ARM, q)))
        assert np.allclose(k.fk(q).translation, p_np, atol=1e-9)
