"""팔로워 관절 명령 1주기 계산 + 안전 장치 (ROS 비의존, 단위 테스트용으로 분리)

안전 장치
- 추종 오차: 실측 관절각과 직전 명령의 차이(관절별 최대)가 max_track_error를 넘으면
  명령을 보내지 않고 명령 기준을 실측값으로 재동기화한다.
  (비상정지·충돌 등으로 로봇이 멈춘 동안 명령만 앞서 나가다가, 풀리는 순간 튀는 것을 방지)
- 리밋 밖 실측: 실측 관절각이 URDF 리밋을 벗어나면 명령을 보내지 않는다(IK가 수렴할 수 없음).
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
                 ik_accept_error: float | None = None,
                 input_timeout: float = 0.2, use_measured: bool = True):
        self.kin = kin
        self.max_step = float(max_joint_step)
        self.tol = float(ik_tolerance)
        # 근사 해 허용 한계: tol < 오차 <= accept면 IK 최소제곱 해(가장 가까운 자세)로 명령을 보내고
        # status='approx'. 도달 범위 경계·특이점 근처에서 멈추지 않고 경계를 따라 움직이게 한다.
        self.accept = max(self.tol, float(ik_accept_error)) if ik_accept_error is not None else self.tol
        self.max_track_error = float(max_track_error)
        self.timeout = float(input_timeout)
        self.use_measured = use_measured

        self.q_cmd: np.ndarray | None = None
        self.q_ik: np.ndarray | None = None      # 직전 IK 해 (다음 IK 시드)
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

    def out_of_limits(self, q, tol: float = 1e-3) -> str:
        """URDF 리밋을 벗어난 관절 설명 ('' = 모두 리밋 안)."""
        k = self.kin
        bad = [f'{n}={v:.2f}' for n, v, lo, hi in zip(k.joint_names, q, k.lower, k.upper)
               if v < lo - tol or v > hi + tol]
        return ', '.join(bad)

    def step(self, t: float) -> np.ndarray | None:
        """보낼 관절 명령, 또는 보내지 않을 때 None (사유는 self.status)."""
        if self.q_cmd is None or self.target is None:
            self.status = 'waiting'
            return None
        if self.use_measured:
            if self.t_meas is None or t - self.t_meas > self.timeout:
                self.status = 'joint_state_timeout'
                return None
            out = self.out_of_limits(self.q_meas)
            if out:
                # URDF 리밋 밖 실측값에서는 IK가 리밋으로 잘려 수렴하지 못함 (시뮬/실기와 URDF 불일치 신호)
                self.status, self.detail = 'out_of_limits', out
                return None
            err = float(np.max(np.abs(self.q_cmd - self.q_meas)))
            if err > self.max_track_error:
                self.q_cmd = self.q_meas.copy()   # 재동기화: 다음 주기는 실측에서 다시 시작
                self.q_ik = None
                self.status, self.detail = 'tracking_error', f'{err:.3f} rad'
                return None
        if t - self.t_target > self.timeout:
            self.status = 'target_timeout'
            self.q_ik = None   # 목표 재개 시 현재 명령에서 다시 시작
            return None

        # IK 시드는 직전 IK 해: 목표는 주기마다 조금씩만 움직이므로 같은 해 가지를 따라간다.
        # (속도 제한으로 뒤처진 q_cmd에서 풀면 먼 목표로 점프하다 국소해에 빠짐)
        seed = self.q_ik if self.q_ik is not None else self.q_cmd
        q_sol, ok, err = self.kin.ik(self.target, seed)
        if not ok and err > self.accept:
            self.status, self.detail = 'ik_failed', f'err={err:.4f}'
            return None
        self.q_ik = q_sol
        # 관절별로 따로 자르면 관절 공간 방향이 바뀌어 IK 해 사이를 벗어난 자세가 되므로,
        # 가장 큰 관절이 max_step이 되도록 전체를 같은 비율로 줄인다.
        dq = q_sol - self.q_cmd
        peak = float(np.max(np.abs(dq)))
        if peak > self.max_step:
            dq *= self.max_step / peak
        self.q_cmd = self.q_cmd + dq
        if ok or err <= self.tol:
            self.status = 'ok'
        else:
            self.status, self.detail = 'approx', f'err={err:.4f}'
        return self.q_cmd
