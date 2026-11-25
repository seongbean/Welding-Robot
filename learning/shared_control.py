"""공유 제어: 고정한 기본 정책이 용접하는 동안 사람이 리더 암으로 보정을 더한다 (numpy만 사용).

docs/architecture.md "사람 개입"의 구현이다. 로봇 목표 보정 = 기본 정책 + 잔차 모델 + 사람 보정 (이음매 좌표계,
[좌우 mm, 높이 mm, 작업각 deg, 진행각 deg]). 진행 속도와 위빙은 모델이 맡는다.

- HumanCorrection: 클러치·유지 오프셋·초기화 버튼 규칙. 매 제어 주기 리더 포즈와 버튼 상태를 넣으면 지금 적용할
  사람 보정을 낸다.
  - 클러치 ON 순간의 리더 포즈를 기준으로, 그 뒤 리더 변화량(live)을 이음매 좌표로 바꿔 더한다.
  - 클러치를 놓으면 그때의 보정을 유지 오프셋(held)으로 남긴다. 다시 누르면 유지 오프셋 위에 더한다.
  - 초기화 버튼: 유지 오프셋을 reset_time 동안 일정 속도로 0까지 줄인다 (클러치 중이면 그때까지의 변화량도
    오프셋으로 넘기고 그 자세에서 다시 기준을 잡는다).
  - 적용 보정은 축별 한도(limit)로 자르고, 주기당 변화 속도(max_rate)를 제한해 리더 잡음·튐이 토치로 가지 않게 한다.
  - 리더 읽기가 끊기면 그때까지의 보정을 유지 오프셋으로 넘기고, 다시 읽히면 그 자세에서 기준을 잡는다.
- leader_delta: 리더 포즈 변화 → 이음매 좌표 보정 (축척·축별 비율 적용 전 원시값 6개도 돌려준다).
  리더 베이스 축을 이음매 좌표(x 진행, y 좌우, z 높이)에 R_ls로 바로 대응시킨다. 파이프처럼 이음매 방향이
  돌아도 리더를 위로 들면 모재에서 멀어진다.
- SharedDriver: learning.policy_driver.PolicyDriver에 사람 보정과 (선택) 잔차 모델을 더한 구동기.
  기록에 잔차 학습 정답과 개입 표시를 남긴다 (아래 RECORD_FIELDS).
- residual_frames: 공유 제어 기록 → 잔차 모델 학습 프레임 (정책 주기 10 Hz, 보정 직전 구간 제외 표시).
"""
from __future__ import annotations

import math

import numpy as np

from sim_errors.part_errors import DEG, MM, R_to_rotvec, rpy_to_R

from .features import make_obs, torch_offset
from .policy_driver import PolicyDriver

HUMAN_AXES = ('lateral', 'height', 'work', 'travel')

# 공유 제어 기록에 더하는 필드 (주기마다 한 행)
RECORD_FIELDS = {
    'policy_c': '로봇이 실제로 쓴 보정 합계 = base_c + residual_c + human_c [mm, mm, deg, deg] (평가의 명령 오차 기준)',
    'base_c': '고정한 기본 정책 보정 출력',
    'residual_c': '잔차 모델 출력 (없으면 0)',
    'human_c': '적용 중인 사람 보정 = 유지 오프셋 + 클러치 중 변화량. 잔차 모델 정답 (초기화 뒤 0)',
    'human_held': '유지 오프셋',
    'human_live': '이번 클러치 중 변화량 (축척·비율 적용 뒤, 한도 전)',
    'leader_delta': '클러치 기준 대비 리더 변화 원시값 [x, y, z mm, rx, ry, rz deg] (이음매 좌표, 축척 전)',
    'clutch': '클러치 ON',
    'reset_active': '오프셋 초기화 램프 중',
    'reset_event': '이 주기에 초기화 버튼이 눌림',
    'leader_ok': '리더 포즈를 읽음',
    'leader_aux': '리더 그리퍼 값 0(닫힘)~1(열림), 없으면 nan',
    'policy_tick': '이 주기에 기본 정책을 새로 불렀음 (10 Hz)',
    'obs': '정책 관측 61개 (OBS_NAMES 순서, policy_tick 주기만 값, 나머지 nan)',
}


def leader_delta(T: np.ndarray, T0: np.ndarray, R_ls: np.ndarray) -> np.ndarray:
    """리더 포즈 T(지금)와 T0(클러치 기준) → 이음매 좌표 변화 [x, y, z mm, rx, ry, rz deg].
    R_ls: 리더 베이스 → 이음매 좌표 회전. 회전은 베이스 기준 변화 dR = R R0^T를 이음매 좌표로 옮긴 회전 벡터."""
    dp = R_ls @ (np.asarray(T)[:3, 3] - np.asarray(T0)[:3, 3])
    dR = R_ls @ (np.asarray(T)[:3, :3] @ np.asarray(T0)[:3, :3].T) @ R_ls.T
    return np.r_[dp / MM, R_to_rotvec(dR) / DEG]


class HumanCorrection:
    """클러치·유지 오프셋·초기화 버튼 → 적용 사람 보정 [좌우 mm, 높이 mm, 작업각 deg, 진행각 deg].

    pos_scale, ang_scale: 리더 변화 → 토치 보정 축척 (0.25면 리더 4 mm = 토치 1 mm).
    gain: 축별 사람 권한 비율 0~1 (0이면 그 축은 사람이 못 바꿈).
    limit: 적용 보정 절댓값 한도. max_rate: 초당 최대 변화. reset_time: 초기화에 걸리는 시간 [s].
    leader_to_seam_rpy: 리더 베이스 → 이음매 좌표 회전 [rad]."""

    def __init__(self, pos_scale: float = 0.25, ang_scale: float = 0.5, gain=(1.0, 1.0, 1.0, 1.0),
                 limit=(5.0, 5.0, 10.0, 10.0), max_rate=(20.0, 20.0, 30.0, 30.0), reset_time: float = 1.0,
                 leader_to_seam_rpy=(0.0, 0.0, 0.0)):
        self.pos_scale, self.ang_scale = float(pos_scale), float(ang_scale)
        self.gain = np.clip(np.asarray(gain, dtype=float), 0.0, 1.0)
        self.limit = np.asarray(limit, dtype=float)
        self.max_rate = np.asarray(max_rate, dtype=float)
        self.reset_time = float(reset_time)
        self.rpy = [float(x) for x in leader_to_seam_rpy]
        self.R_ls = rpy_to_R(self.rpy)
        self.anchor: np.ndarray | None = None
        self.held = np.zeros(4)
        self.live = np.zeros(4)
        self.raw = np.zeros(6)
        self.applied = np.zeros(4)
        self.reset_rate: np.ndarray | None = None      # 초기화 램프 중이면 축별 초당 감소량
        self.clutch = False

    def config(self) -> dict:
        return {'pos_scale': self.pos_scale, 'ang_scale': self.ang_scale, 'gain': self.gain.tolist(),
                'limit': self.limit.tolist(), 'max_rate': self.max_rate.tolist(), 'reset_time': self.reset_time,
                'leader_to_seam_rpy': self.rpy}

    def _map(self, raw: np.ndarray) -> np.ndarray:
        """원시 변화 6개 → 보정 4축 (좌우 = y, 높이 = z, 작업각 = x축 회전, 진행각 = y축 회전). 진행 방향 x는 버림."""
        return self.gain * np.array([raw[1] * self.pos_scale, raw[2] * self.pos_scale,
                                     raw[3] * self.ang_scale, raw[4] * self.ang_scale])

    def _fold(self) -> None:
        """지금까지의 변화량을 유지 오프셋으로 넘기고 기준을 버린다."""
        self.held = np.clip(self.held + self.live, -self.limit, self.limit)
        self.live = np.zeros(4)
        self.raw = np.zeros(6)
        self.anchor = None

    def step(self, dt: float, T_leader: np.ndarray | None, clutch: bool, reset: bool = False) -> np.ndarray:
        """한 주기. T_leader: 리더 끝 포즈 4x4 (못 읽으면 None). 반환: 적용 사람 보정 (4,)."""
        if reset:
            self._fold()
            self.reset_rate = np.abs(self.held) / max(self.reset_time, 1e-6)
        if clutch and T_leader is not None:
            if self.anchor is None:                    # 클러치 ON 순간, 또는 리더가 다시 읽힌 순간
                self.anchor = np.array(T_leader, dtype=float)
            self.raw = leader_delta(T_leader, self.anchor, self.R_ls)
            self.live = self._map(self.raw)
        elif self.anchor is not None:                  # 클러치 OFF 또는 리더 끊김: 유지
            self._fold()
        self.clutch = bool(clutch)
        if self.reset_rate is not None:
            self.held = np.sign(self.held) * np.maximum(np.abs(self.held) - self.reset_rate * dt, 0.0)
            if not self.held.any():
                self.reset_rate = None
        want = np.clip(self.held + self.live, -self.limit, self.limit)
        lim = self.max_rate * dt
        self.applied = self.applied + np.clip(want - self.applied, -lim, lim)
        return self.applied.copy()

    @property
    def resetting(self) -> bool:
        return self.reset_rate is not None


class SharedDriver(PolicyDriver):
    """기본 정책 + 잔차 + 사람 보정으로 토치를 모는 구동기 (PolicyDriver와 같은 step 인터페이스).

    human_input(): 매 주기 불러 (T_leader 4x4 또는 None, clutch, reset_pressed, aux 또는 None)를 받는다.
    residual(obs, base_c) → 보정 4개 (없으면 0). 잔차 모델은 기본 정책과 같은 10 Hz로 부른다.
    stop_requested(): True면 에피소드를 멈춘다 (기록은 끝나지 않은 실행으로 남김)."""

    def __init__(self, policy, model, expert, human: HumanCorrection, human_input, residual=None,
                 stop_requested=None, **kw):
        super().__init__(policy, model, expert, **kw)
        self.human, self.human_input = human, human_input
        self.residual = residual
        self.stop_requested = stop_requested or (lambda: False)
        self.base_c = self.c.copy()
        self.res_c = np.zeros(4)
        self.stopped = False

    def step(self, dt: float, T_meas: np.ndarray) -> dict:
        s = self.s
        laser = self.m.laser(s, s, self.rng)
        self.buf.add(s + self.look, laser)
        weave = self.amp * math.sin(self.phase)
        tick = self.t + 1e-9 >= self.next_t
        obs = np.full(61, np.nan, dtype=np.float32)
        if tick:
            vals, valid = self.buf.query(s)
            obs = make_obs(vals, valid, torch_offset(self.m, s, T_meas), s, self.mode)
            a = np.asarray(self.policy(obs), dtype=float)
            self.action = a.copy()
            self.base_c = a[:4].copy()
            self.speed = float(np.clip(a[4], 0.3 * self.v_nom, 2.0 * self.v_nom))
            self.amp = float(np.clip(a[5], 0.0, 2 * self.cfg['weave_amp_max_mm']))
            self.hz = float(np.clip(a[6], 0.0, 3.0))
            if self.residual is not None:
                self.res_c = np.asarray(self.residual(obs, self.base_c), dtype=float)[:4]
            self.next_t += self.period
        T_leader, clutch, reset, aux = self.human_input()
        h = self.human.step(dt, T_leader, clutch, reset)
        self.c = self.base_c + self.res_c + h
        T = self.target(s, self.c, weave)
        lab = self.m.label(s, s)
        hc = self.human
        r = {'t': self.t, 'phase': 'weld', 'arc_on': True, 'done': False, 's': s, 's_arc': s,
             'T_target': T, 'T_expert': T, 'perturb': np.zeros(4),
             'e': lab['e'], 'rotvec': lab['rotvec'], 'gap': lab['gap'], 'tack': lab['tack'],
             'p_true': lab['p_true'], 'p_est': lab['p_est'],
             'weave_amp': self.amp * MM, 'weave_hz': self.hz, 'speed': self.speed * MM, 'laser': laser,
             'extra': {'policy_c': self.c.copy(), 'policy_weave': weave, 'policy_action': self.action.copy(),
                       'base_c': self.base_c.copy(), 'residual_c': self.res_c.copy(), 'human_c': h,
                       'human_held': hc.held.copy(), 'human_live': hc.live.copy(), 'leader_delta': hc.raw.copy(),
                       'clutch': bool(clutch), 'reset_active': hc.resetting, 'reset_event': bool(reset),
                       'leader_ok': T_leader is not None, 'leader_aux': float('nan') if aux is None else float(aux),
                       'policy_tick': bool(tick), 'obs': obs}}
        self.s = min(1.0, s + self.speed * MM * dt / self.m.L)
        self.phase += 2 * math.pi * self.hz * dt
        self.t += dt
        r['done'] = self.s >= 1.0
        self.stopped = bool(self.stop_requested())
        self.done = r['done'] or self.t >= self.t_max or self.stopped
        return r


def intervention_stats(rows: dict) -> dict:
    """공유 제어 기록 하나의 개입 요약: 클러치 횟수·시간 비율, 초기화 횟수, 사람 보정 최대·RMS, 보정이 남은 비율."""
    clutch = np.asarray(rows['clutch']).astype(bool)
    h = np.asarray(rows['human_c'], dtype=float)
    on = np.flatnonzero(clutch[1:] & ~clutch[:-1]) + 1
    n_on = len(on) + int(bool(len(clutch) and clutch[0]))
    active = np.abs(h).max(axis=1) > 1e-6 if len(h) else np.zeros(0, bool)
    out = {'n_clutch': int(n_on), 'clutch_frac': float(clutch.mean()) if len(clutch) else 0.0,
           'n_reset': int(np.asarray(rows['reset_event']).astype(bool).sum()),
           'human_frac': float(active.mean()) if len(active) else 0.0}
    for j, n in enumerate(HUMAN_AXES):
        out[f'human_{n}_max'] = float(np.abs(h[:, j]).max()) if len(h) else 0.0
        out[f'human_{n}_rms'] = float(np.sqrt(np.mean(h[:, j] ** 2))) if len(h) else 0.0
    return out


def residual_frames(rows: dict, pre_drop_s: float = 2.0) -> dict:
    """공유 제어 기록 → 잔차 학습 프레임 (기본 정책을 부른 10 Hz 주기만).

    반환: obs (n,61), base_c (n,4), label (n,4) = 그 주기의 적용 사람 보정(유지 오프셋 포함, 초기화 뒤 0),
    clutch (n,), human (n,) 보정이 0이 아님, keep (n,) = 클러치를 누르기 직전 pre_drop_s 초가 아님
    (사람이 어긋남을 보고도 아직 손대지 않은 구간이라 정답 0이 틀림), t (n,)."""
    tick = np.asarray(rows['policy_tick']).astype(bool)
    t = np.asarray(rows['t'], dtype=float)
    clutch = np.asarray(rows['clutch']).astype(bool)
    h = np.asarray(rows['human_c'], dtype=float)
    starts = t[np.flatnonzero(clutch[1:] & ~clutch[:-1]) + 1]
    if len(clutch) and clutch[0]:
        starts = np.r_[t[0], starts]
    idx = np.flatnonzero(tick)
    ti = t[idx]
    drop = np.zeros(len(idx), bool)
    for ts in starts:
        drop |= (ti >= ts - pre_drop_s) & (ti < ts)
    drop &= np.abs(h[idx]).max(axis=1) < 1e-9 if len(idx) else drop    # 이미 보정 중(유지 오프셋)이면 남김
    return {'obs': np.asarray(rows['obs'], dtype=np.float32)[idx], 'base_c': np.asarray(rows['base_c'])[idx],
            'label': h[idx], 'clutch': clutch[idx], 'human': np.abs(h[idx]).max(axis=1) > 1e-6,
            'keep': ~drop, 't': ti}
