"""mapping.EeMapper: 클러치 재기준, 스케일, 축 정렬, 속도 제한, 작업공간 제한."""
import math

import numpy as np
import pinocchio as pin
import pytest

from weld_teleop.mapping import EeMapper

F0 = pin.SE3(pin.rpy.rpyToMatrix(math.pi, 0.0, 0.0), np.array([0.4, 0.0, 0.3]))
L0 = pin.SE3(np.eye(3), np.array([0.2, 0.0, 0.1]))


def moved(T, dxyz=(0, 0, 0), drpy=(0, 0, 0)):
    """베이스 기준으로 이동/회전한 포즈"""
    return pin.SE3(pin.rpy.rpyToMatrix(*drpy) @ T.rotation, T.translation + np.array(dxyz))


def run(m, leader, n):
    cmd = None
    for _ in range(n):
        cmd = m.step(leader, F0)
    return cmd


def anchored(**kw):
    kw.setdefault('max_linear_step', 1.0)
    kw.setdefault('max_angular_step', 10.0)
    m = EeMapper(**kw)
    assert m.step(L0, F0) is None   # 첫 주기는 재기준만
    return m


def test_first_step_anchors_and_holds_follower():
    m = anchored()
    cmd = m.step(L0, F0)
    assert cmd.isApprox(F0, 1e-12)


def test_translation_is_scaled_relative_motion():
    m = anchored(position_scale=2.0)
    cmd = m.step(moved(L0, (0.01, -0.02, 0.03)), F0)
    assert np.allclose(cmd.translation, F0.translation + 2.0 * np.array([0.01, -0.02, 0.03]))
    assert np.allclose(cmd.rotation, F0.rotation)


def test_base_alignment_rotates_leader_motion():
    # 리더 베이스가 팔로워 대비 z축 90° 돌아가 있으면 리더 +x 이동 = 팔로워 +y 이동
    m = anchored(leader_to_follower_rpy=(0.0, 0.0, math.pi / 2))
    cmd = m.step(moved(L0, (0.01, 0.0, 0.0)), F0)
    assert np.allclose(cmd.translation - F0.translation, [0.0, 0.01, 0.0], atol=1e-12)


def test_full_mode_applies_relative_rotation():
    m = anchored()
    d = (0.0, 0.0, 0.1)
    cmd = m.step(moved(L0, drpy=d), F0)
    assert np.allclose(cmd.rotation, pin.rpy.rpyToMatrix(*d) @ F0.rotation)


def test_position_only_keeps_follower_orientation():
    m = anchored(orientation_mode='position_only')
    cmd = m.step(moved(L0, (0.01, 0, 0), drpy=(0.3, 0.2, 0.1)), F0)
    assert np.allclose(cmd.rotation, F0.rotation)
    assert np.allclose(cmd.translation, F0.translation + [0.01, 0, 0])


def test_invalid_orientation_mode_rejected():
    with pytest.raises(ValueError):
        EeMapper(orientation_mode='positon_only')


def test_linear_step_is_limited_and_converges():
    m = anchored(max_linear_step=0.002)
    target = F0.translation + [0.05, 0.0, 0.0]
    leader = moved(L0, (0.05, 0.0, 0.0))
    prev = F0.translation.copy()
    for _ in range(10):
        cmd = m.step(leader, F0)
        assert np.linalg.norm(cmd.translation - prev) <= 0.002 + 1e-12
        prev = cmd.translation.copy()
    assert np.isclose(prev[0] - F0.translation[0], 0.02)
    cmd = run(m, leader, 30)
    assert np.allclose(cmd.translation, target)


def test_angular_step_is_limited():
    m = anchored(max_angular_step=0.02)
    cmd = m.step(moved(L0, drpy=(0.0, 0.0, 0.5)), F0)
    ang = np.linalg.norm(pin.log3(F0.rotation.T @ cmd.rotation))
    assert np.isclose(ang, 0.02)


def test_workspace_clips_target():
    m = anchored(workspace_min=(-0.5, -0.5, 0.2), workspace_max=(0.5, 0.5, 0.8))
    cmd = run(m, moved(L0, (1.0, 0.0, -1.0)), 5)
    assert np.allclose(cmd.translation, [0.5, 0.0, 0.2])


def test_reset_reanchors_without_jump():
    # 클러치 OFF 후 리더를 옮겨 놓고 다시 ON → 팔로워가 튀지 않아야 한다
    m = anchored()
    run(m, moved(L0, (0.02, 0, 0)), 3)
    m.reset()
    far_leader = moved(L0, (0.1, 0.1, 0.1))
    held = pin.SE3(F0.rotation, F0.translation + [0.02, 0, 0])
    assert m.step(far_leader, held) is None
    cmd = m.step(far_leader, held)
    assert cmd.isApprox(held, 1e-12)


def test_base_alignment_rotates_leader_rotation_axis():
    # 리더 x축 회전은 z축 90° 정렬 후 팔로워 y축 회전이 되어야 한다
    m = anchored(leader_to_follower_rpy=(0.0, 0.0, math.pi / 2))
    cmd = m.step(moved(L0, drpy=(0.1, 0.0, 0.0)), F0)
    assert np.allclose(cmd.rotation, pin.rpy.rpyToMatrix(0.0, 0.1, 0.0) @ F0.rotation)
