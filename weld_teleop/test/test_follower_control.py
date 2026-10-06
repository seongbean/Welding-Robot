"""follower_control.FollowerStepper: 추종 오차·입력 끊김 시 명령 정지 (so102.urdf를 대역으로 사용)."""
import numpy as np
import pytest
from conftest import SO102_URDF

from weld_teleop.follower_control import FollowerStepper
from weld_teleop.kinematics import ChainKinematics

ARM = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll']
Q0 = np.array([0.1, -0.3, 0.4, 0.2, -0.1, 0.3])
DT = 0.01


@pytest.fixture(scope='module')
def kin():
    return ChainKinematics(SO102_URDF, ARM, 'gripper_frame_link')


def make(kin, **kw):
    s = FollowerStepper(kin, max_joint_step=0.01, max_track_error=0.1, input_timeout=0.2, **kw)
    return s


def target_near(kin, dq=0.05):
    return kin.fk(Q0 + dq)


def test_waits_without_inputs(kin):
    s = make(kin)
    assert s.step(0.0) is None and s.status == 'waiting'
    s.on_measured(Q0, 0.0)
    assert s.step(0.0) is None and s.status == 'waiting'


def test_steps_toward_target_with_step_limit(kin):
    s = make(kin)
    s.on_measured(Q0, 0.0)
    s.on_target(target_near(kin), 0.0)
    q = s.step(DT)
    assert s.status == 'ok'
    assert np.max(np.abs(q - Q0)) <= 0.01 + 1e-12


def test_step_limit_keeps_joint_space_direction(kin):
    s = make(kin)
    s.on_measured(Q0, 0.0)
    s.on_target(kin.fk(Q0 + np.array([0.0, 0.2, -0.1, 0.0, 0.05, 0.0])), 0.0)
    dq = s.step(DT) - Q0
    sol = s.q_ik - Q0
    assert np.max(np.abs(dq)) == pytest.approx(0.01)
    assert np.allclose(dq / np.linalg.norm(dq), sol / np.linalg.norm(sol), atol=1e-9)


def test_ik_seed_follows_previous_solution_and_resets_on_resync(kin):
    s = make(kin)
    s.on_measured(Q0, 0.0)
    s.on_target(target_near(kin, dq=0.3), 0.0)
    s.step(DT)
    assert s.q_ik is not None and kin.fk(s.q_ik).isApprox(target_near(kin, dq=0.3), 1e-3)
    s.on_measured(Q0 + 0.5, DT)   # 큰 추종 오차 → 재동기화
    s.step(DT)
    assert s.status == 'tracking_error' and s.q_ik is None


def test_converges_when_robot_follows(kin):
    s = make(kin)
    target = target_near(kin)
    s.on_measured(Q0, 0.0)
    t = 0.0
    for _ in range(50):
        t += DT
        s.on_target(target, t)
        q = s.step(t)
        s.on_measured(q, t)   # 로봇이 명령대로 따라옴
    assert kin.fk(s.q_cmd).isApprox(target, 1e-3)


def test_tracking_error_stops_and_resyncs(kin):
    # 로봇이 멈춰 있는데(실측 고정) 명령만 앞서 나가면 정지하고 실측으로 재동기화
    s = make(kin)
    far = kin.fk(Q0 + 0.5)
    t = 0.0
    sent = []
    for _ in range(30):
        t += DT
        s.on_measured(Q0, t)
        s.on_target(far, t)
        q = s.step(t)
        if q is None:
            assert s.status == 'tracking_error'
            assert np.allclose(s.q_cmd, Q0)
        else:
            sent.append(q.copy())
    # 보낸 명령은 어떤 것도 실측에서 max_track_error + 1스텝 넘게 벗어나지 않는다
    assert sent
    assert max(np.max(np.abs(q - Q0)) for q in sent) <= 0.1 + 0.01 + 1e-12


def test_joint_state_timeout_stops(kin):
    s = make(kin)
    s.on_measured(Q0, 0.0)
    s.on_target(target_near(kin), 0.5)
    assert s.step(0.5) is None and s.status == 'joint_state_timeout'


def test_target_timeout_stops(kin):
    # 클러치 OFF 등으로 목표가 끊기면 정지
    s = make(kin)
    s.on_target(target_near(kin), 0.0)
    s.on_measured(Q0, 0.5)
    assert s.step(0.5) is None and s.status == 'target_timeout'
    s.on_target(target_near(kin), 0.5)
    assert s.step(0.5) is not None


def test_ik_failure_stops(kin):
    import pinocchio as pin
    s = make(kin)
    s.on_measured(Q0, 0.0)
    s.on_target(pin.SE3(np.eye(3), np.array([5.0, 0.0, 0.0])), 0.0)
    assert s.step(DT) is None and s.status == 'ik_failed'


def test_loopback_skips_measured_checks(kin):
    s = make(kin, use_measured=False)
    s.set_initial(Q0)
    s.on_target(target_near(kin), 10.0)
    assert s.step(10.0) is not None and s.status == 'ok'


def test_stops_when_measured_outside_urdf_limits(kin):
    s = make(kin)
    q = Q0.copy()
    q[4] = kin.upper[4] + 0.3   # 시뮬/실기 관절이 URDF 리밋 밖
    s.on_measured(q, 0.0)
    s.on_target(target_near(kin), 0.0)
    assert s.step(0.0) is None
    assert s.status == 'out_of_limits' and 'wrist_yaw' in s.detail
    assert s.out_of_limits(Q0) == ''


def test_approx_solution_within_accept_error_still_moves(kin):
    import pinocchio as pin
    s = FollowerStepper(kin, max_joint_step=0.01, input_timeout=0.2, ik_accept_error=10.0)
    s.on_measured(Q0, 0.0)
    s.on_target(pin.SE3(np.eye(3), np.array([5.0, 0.0, 0.0])), 0.0)   # 도달 불가지만 허용 오차 안
    q = s.step(DT)
    assert q is not None and s.status == 'approx'
    assert 0 < np.max(np.abs(q - Q0)) <= 0.01 + 1e-12

