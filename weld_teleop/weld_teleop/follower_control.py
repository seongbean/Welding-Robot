"""팔로워 관절 명령 1주기 계산 + 안전 장치 (ROS 비의존, 단위 테스트용으로 분리)

안전 장치
- 추종 오차: 실측 관절각과 직전 명령의 차이(관절별 최대)가 max_track_error를 넘으면
  명령을 보내지 않고 명령 기준을 실측값으로 재동기화한다.
  (비상정지·충돌 등으로 로봇이 멈춘 동안 명령만 앞서 나가다가, 풀리는 순간 튀는 것을 방지)
- 입력 끊김: 목표 포즈나 관절 상태가 input_timeout 초 넘게 갱신되지 않으면 명령을 보내지 않는다.
  (클러치 OFF로 목표가 끊겨도 같은 방식으로 정지)
loopback(실측 없음)에서는 관절 상태 관련 검사를 건너뛴다.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from .kinematics import ChainKinematics


class FollowerStepper:
    def __init__(self, kin: ChainKinematics, max_joint_step: float = 0.01,
                 ik_tolerance: float = 1e-3, max_track_error: float = 0.1,
                 input_timeout: float = 0.2, use_measured: bool = True):
        self.kin = kin
        self.max_step = float(max_joint_step)
        self.tol = float(ik_tolerance)
        self.max_track_error = float(max_track_error)
        self.timeout = float(input_timeout)
        self.use_measured = use_measured

        self.q_cmd: np.ndarray | None = None
        self.q_meas: np.ndarray | None = None
        self.t_meas: float | None = None
        self.target: pin.SE3 | None = None
        self.t_target: float | None = None
        self.status = 'waiting'   # 마지막 step()의 결과 코드 (로그용)
        self.detail = ''

    def set_initial(self, q) -> None:
        self.q_cmd = np.array(q, dtype=float)

    def on_measured(self, q, t: float) -> None:
        self.q_meas = np.array(q, dtype=float)
        self.t_meas = t
        if self.q_cmd is None:
            self.q_cmd = self.q_meas.copy()

    def on_target(self, T: pin.SE3, t: float) -> None:
        self.target = T
        self.t_target = t

    def step(self, t: float) -> np.ndarray | None:
        """보낼 관절 명령, 또는 보내지 않을 때 None (사유는 self.status)."""
        if self.q_cmd is None or self.target is None:
            self.status = 'waiting'
            return None
        if self.use_measured:
            if self.t_meas is None or t - self.t_meas > self.timeout:
                self.status = 'joint_state_timeout'
                return None
            err = float(np.max(np.abs(self.q_cmd - self.q_meas)))
            if err > self.max_track_error:
                self.q_cmd = self.q_meas.copy()   # 재동기화: 다음 주기는 실측에서 다시 시작
                self.status, self.detail = 'tracking_error', f'{err:.3f} rad'
                return None
        if t - self.t_target > self.timeout:
            self.status = 'target_timeout'
            return None

        q_sol, ok, err = self.kin.ik(self.target, self.q_cmd)
        if not ok and err > self.tol:
            self.status, self.detail = 'ik_failed', f'err={err:.4f}'
            return None
        dq = np.clip(q_sol - self.q_cmd, -self.max_step, self.max_step)
        self.q_cmd = self.q_cmd + dq
        self.status = 'ok'
        return self.q_cmd
