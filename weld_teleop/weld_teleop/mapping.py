"""리더 EE 포즈 → 팔로워 TCP 목표 포즈 계산 (ROS 비의존, 단위 테스트용으로 분리)

클러치 ON 순간의 리더 포즈 L0, 팔로워 포즈 F0를 기준점으로 저장하고
  p_target = p_F0 + s * R_bl (p_L - p_L0)
  R_target = (R_bl dR_L R_bl^T) R_F0,   dR_L = R_L R_L0^T
로 계산한다. R_bl: 리더 베이스 → 팔로워 베이스 회전(축 정렬용).

이후 1주기당 최대 이동량(병진/회전)을 제한하며 목표 쪽으로 따라가고,
팔로워 베이스 기준 작업공간 박스로 위치를 자른다.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin


class EeMapper:
    def __init__(self, position_scale: float = 1.0,
                 leader_to_follower_rpy=(0.0, 0.0, 0.0),
                 orientation_mode: str = 'full',
                 max_linear_step: float = 0.002, max_angular_step: float = 0.02,
                 workspace_min=(-1.0, -1.0, 0.0), workspace_max=(1.0, 1.0, 1.2)):
        if orientation_mode not in ('full', 'position_only'):
            raise ValueError(f'orientation_mode는 full | position_only: {orientation_mode}')
        self.s = float(position_scale)
        self.R_bl = pin.rpy.rpyToMatrix(*leader_to_follower_rpy)
        self.mode = orientation_mode
        self.max_lin = float(max_linear_step)
        self.max_ang = float(max_angular_step)
        self.ws_min = np.array(workspace_min, dtype=float)
        self.ws_max = np.array(workspace_max, dtype=float)
        self.anchor = None          # (L0, F0)
        self.cmd: pin.SE3 | None = None

    def reset(self) -> None:
        """클러치 OFF: 다음 step()에서 재기준."""
        self.anchor = None

    def step(self, leader: pin.SE3, follower: pin.SE3) -> pin.SE3 | None:
        """한 주기 계산. 재기준 주기에는 None(명령 없음)을 반환한다."""
        if self.anchor is None:  # 클러치 ON 직후 재기준
            self.anchor = (pin.SE3(leader), pin.SE3(follower))
            self.cmd = pin.SE3(follower)
            return None

        L0, F0 = self.anchor
        L = leader
        p = F0.translation + self.s * self.R_bl @ (L.translation - L0.translation)
        p = np.clip(p, self.ws_min, self.ws_max)
        if self.mode == 'full':
            dR = self.R_bl @ (L.rotation @ L0.rotation.T) @ self.R_bl.T
            R = dR @ F0.rotation
        else:
            R = F0.rotation
        raw = pin.SE3(R, p)

        # 1주기 이동량 제한: cmd → raw 방향 twist를 잘라서 적용
        xi = pin.log6(self.cmd.actInv(raw)).vector
        lin, ang = xi[:3], xi[3:]
        nl, na = np.linalg.norm(lin), np.linalg.norm(ang)
        if nl > self.max_lin:
            lin = lin * self.max_lin / nl
        if na > self.max_ang:
            ang = ang * self.max_ang / na
        self.cmd = self.cmd * pin.exp6(np.concatenate([lin, ang]))
        return self.cmd
