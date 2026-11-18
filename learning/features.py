"""정책 관측·행동 정의 (numpy만 사용). 학습 데이터 변환과 평가 롤아웃이 같은 함수를 쓴다.

관측 (에피소드 생성 사양 "기록 형식", 단위 mm·deg)
- section: 단면 거리 버퍼. 토치 뒤 10 mm부터 앞 30 mm까지 5 mm 간격 9점 × (좌우, 높이, 틈새 폭) + 9점 유효 표시.
  단면 센서는 토치 앞 30 mm만 재므로, 잰 값을 잰 이음매 위치와 함께 쌓아 두고 지금 토치 위치 기준으로 꺼낸다.
  값은 추정 경로 좌표계 편차 (카메라가 토치에 붙어 있어 토치 기준으로 잰 값 + 그때의 토치 자세).
  아직 못 잰 점과 측정 실패는 0 + 유효 0.
- profile: 같은 버퍼 9점 × (작업각, 진행각). 단면 패치의 두 판 면 평면 맞춤으로 구한 이음매 좌표계 회전.
  이전 기록(단면 세 열)에는 없어서 0이다. sim_errors.resense로 다시 계산할 수 있다.
- torch: 추정 경로 기준 토치 자세 어긋남 (좌우, 높이, 작업각, 진행각). 위빙·섭동·시작 어긋남이 다 들어간다.
- progress: 진행도 s
- mode: 직선/위빙 원핫 (위빙, 직선)

행동 (이음매 좌표계 보정량, 정답 = 규칙 전문가)
- lateral, height(CTWD, 가접 올림 포함) [mm], work, travel [deg], speed [mm/s], weave_amp [mm], weave_hz [Hz]
"""
from __future__ import annotations

import json
import math

import numpy as np

from sim_errors.part_errors import DEG, MM, PartErrorModel, rotvec_to_R

BUF_OFFSETS_MM = np.arange(-10.0, 30.1, 5.0)          # 9점
BUF_BIN_MM = 1.0                                      # 버퍼 칸 크기
BUF_MATCH_MM = 2.5                                    # 질의점에서 이만큼 안의 가장 가까운 칸
BUF_CH = 5                                            # 좌우, 높이, 틈새 [mm], 작업각, 진행각 [deg]

SECTION_NAMES = ([f'sec_{c}_{int(o):+d}' for o in BUF_OFFSETS_MM for c in ('lat', 'h', 'gap')]
                 + [f'sec_valid_{int(o):+d}' for o in BUF_OFFSETS_MM])
PROFILE_NAMES = [f'sec_{c}_{int(o):+d}' for o in BUF_OFFSETS_MM for c in ('work', 'travel')]
GROUPS = {
    'section': SECTION_NAMES,
    'profile': PROFILE_NAMES,
    'torch': ['torch_lat', 'torch_h', 'torch_work', 'torch_travel'],
    'progress': ['s'],
    'mode': ['mode_weave', 'mode_straight'],
}
OBS_NAMES = [n for g in GROUPS.values() for n in g]
ACTION_NAMES = ['lateral', 'height', 'work', 'travel', 'speed', 'weave_amp', 'weave_hz']
FPS = 10


def obs_index(groups) -> np.ndarray:
    """관측 묶음 이름들 → OBS_NAMES 안의 열 번호."""
    names = [n for g in groups for n in GROUPS[g]]
    return np.array([OBS_NAMES.index(n) for n in names])


class SectionBuffer:
    """단면 측정을 이음매 거리(mm) 칸에 쌓는다. 같은 칸은 나중 값이 덮어씀."""

    def __init__(self, length_m: float):
        self.L_mm = length_m / MM
        self.bins: dict[int, np.ndarray] = {}

    def add(self, s_meas: float, meas) -> None:
        """meas: [좌우, 높이, (틈새), (작업각, 진행각)] m·rad. 없는 열은 0."""
        meas = np.asarray(meas, dtype=float)
        if not np.all(np.isfinite(meas[:2])):
            return
        v = np.zeros(BUF_CH)
        v[:len(meas)] = np.where(np.isfinite(meas), meas, 0.0)[:BUF_CH]
        v[:3] /= MM
        v[3:] /= DEG
        self.bins[int(round(s_meas * self.L_mm / BUF_BIN_MM))] = v

    def query(self, s_torch: float) -> tuple[np.ndarray, np.ndarray]:
        """(9,5) 값 [mm, deg], (9,) 유효."""
        vals = np.zeros((len(BUF_OFFSETS_MM), BUF_CH))
        valid = np.zeros(len(BUF_OFFSETS_MM))
        reach = int(BUF_MATCH_MM // BUF_BIN_MM)
        for k, off in enumerate(BUF_OFFSETS_MM):
            c = (s_torch * self.L_mm + off) / BUF_BIN_MM
            b0 = int(round(c))
            for d in sorted(range(-reach, reach + 1), key=abs):
                v = self.bins.get(b0 + d)
                if v is not None:
                    vals[k], valid[k] = v, 1.0
                    break
        return vals, valid


def torch_offset(model: PartErrorModel, s: float, T: np.ndarray) -> np.ndarray:
    """토치 포즈 T의 추정 경로 기준 어긋남 [좌우 mm, 높이 mm, 작업각 deg, 진행각 deg] (토치 롤은 무시)."""
    ex = model.scn['expert']
    p_e, R_e = model.seam_est(s)
    p_e, R_e = p_e[0], R_e[0]
    d = R_e.T @ (T[:3, 3] - p_e) - np.array([0.0, 0.0, ex['standoff_mm'] * MM])
    a = ex['travel_angle_deg'] * DEG
    zn = np.array([math.sin(a), 0.0, -math.cos(a)])          # 어긋남 0일 때 토치 방향 (이음매 좌표계)
    zm = R_e.T @ T[:3, 2]
    rv = np.zeros(2)                                         # zm = Rot([rx, ry, 0]) zn 을 푼다 (가우스-뉴턴)
    for _ in range(6):
        f = rotvec_to_R([rv[0], rv[1], 0.0]) @ zn - zm
        J = np.stack([(rotvec_to_R([rv[0] + h0, rv[1] + h1, 0.0]) @ zn - zm - f) / 1e-6
                      for h0, h1 in ((1e-6, 0.0), (0.0, 1e-6))], axis=1)
        step = np.linalg.lstsq(J, -f, rcond=None)[0]
        rv += step
        if np.abs(step).max() < 1e-10:
            break
    return np.array([d[1] / MM, d[2] / MM, rv[0] / DEG, rv[1] / DEG])


def make_obs(vals, valid, torch, s: float, mode: str) -> np.ndarray:
    """OBS_NAMES 순서: section(좌우·높이·틈새 27 + 유효 9), profile(작업각·진행각 18), torch 4, s, 모드 2."""
    vals = np.asarray(vals)
    return np.concatenate([vals[:, :3].ravel(), valid, vals[:, 3:5].ravel(), torch, [s],
                           [1.0, 0.0] if mode == 'weave' else [0.0, 1.0]]).astype(np.float32)


def expert_action(rows: dict, i: int, tack_follow: float) -> np.ndarray:
    e, rv = rows['e'][i], rows['rotvec'][i]
    return np.array([e[1] / MM, (e[2] + tack_follow * rows['tack'][i]) / MM, rv[0] / DEG, rv[1] / DEG,
                     rows['speed'][i] / MM, rows['weave_amp'][i] / MM, rows['weave_hz'][i]], dtype=np.float32)


def load_episode(path: str) -> tuple[dict, dict, dict]:
    d = np.load(path, allow_pickle=False)
    rows = {k: d[k] for k in d.files if k not in ('scenario_json', 'meta_json')}
    scn = json.loads(str(d['scenario_json']))
    meta = json.loads(str(d['meta_json'])) if 'meta_json' in d.files else {}
    return rows, scn, meta


def episode_frames(rows: dict, scn: dict, fps: int = FPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """100 Hz 기록 → fps 프레임 (관측 (n,61), 행동 (n,7), 시각 (n,)). 용접 구간(아크 ON)만."""
    model = PartErrorModel(scn)
    look = scn['sensors']['laser_lookahead'] / model.L
    mode = scn.get('episode', {}).get('mode', 'weave')
    tf = scn['expert']['tack_follow']
    weld = np.flatnonzero(rows['arc_on'].astype(bool))
    buf = SectionBuffer(model.L)
    t0 = float(rows['t'][weld[0]])
    next_t = t0
    obs, act, ts = [], [], []
    for i in weld:
        s = float(rows['s'][i])
        buf.add(s + look, rows['laser'][i])
        t = float(rows['t'][i])
        if t + 1e-9 >= next_t:
            vals, valid = buf.query(s)
            obs.append(make_obs(vals, valid, torch_offset(model, s, rows['T_meas'][i]), s, mode))
            act.append(expert_action(rows, i, tf))
            ts.append(t - t0)
            next_t += 1.0 / fps
    return np.array(obs), np.array(act), np.array(ts)
