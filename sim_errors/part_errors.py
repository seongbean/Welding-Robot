"""모재 오차 모델: 용접 전·중·후 오차를 분포에서 뽑고, 실제 경로·추정 경로·정답(보정값)을 계산한다.

numpy만 쓴다. Isaac Sim 파이썬(isaac_opti3_receiver.py)과 conda 환경(expert_to_sim.py) 양쪽에서 import한다.

경로 세 가지 (계획 문서 "경로 세 가지와 정답의 정의")
- 명목 경로 p_nom(s): CAD 모재 좌표계의 이음매 (s*L, 0, 0)
- 추정 경로 p_est(s) = T_est · p_nom(s), T_est = T_place · ICP 잔차. 용접 전에 한 번 정해지고 고정
- 실제 경로 p_true(s, s_arc) = T_place · [p_nom(s) + 형상 오차(s) + 열변형(s, s_arc) + 밀림(s_arc)]
정답 e = R_est(s)^T (p_true - p_est): y = 좌우, z = 높이. 각도 정답 = log(R_est^T R_true) (x = 작업각, y = 진행각)

모재 좌표계 (T이음 필릿): 원점 = 이음매 시작점, x = 이음매 방향, 바닥 판 윗면 z = 0,
웹 판 용접 쪽 면 y = 0 (웹은 y >= 0, 토치는 -y 쪽에서 접근). 이음매 좌표계 z = 두 판 법선의 이등분.
단위: 내부 계산은 m, rad. config(yaml)는 mm, deg.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np

BASE, WEB = 0, 1
SCHEMA_VERSION = 1


# ----------------------------------------------------------------------------- 회전 유틸
def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rpy_to_R(rpy) -> np.ndarray:
    """roll-pitch-yaw (URDF 규약: R = Rz(yaw) Ry(pitch) Rx(roll))."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rotvec_to_R(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def R_to_rotvec(R) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if a < 1e-9:
        return w / 2
    if math.pi - a < 1e-6:   # 180° 근처: 대각 성분으로 축 계산
        k = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        k *= np.sign(w + (w == 0))
        return k * a
    return w * a / (2 * math.sin(a))


def R_to_quat_xyzw(R) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = s / 4
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def _normalize(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


# 명목 이음매 좌표계 (모재 좌표계 기준): x 진행, z = 바닥 법선(0,0,1)과 웹 법선(0,-1,0)의 이등분, y = z × x
R_SEAM_NOM = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1 / math.sqrt(2), -1 / math.sqrt(2)],
                       [0.0, 1 / math.sqrt(2), 1 / math.sqrt(2)]])


# ----------------------------------------------------------------------------- 샘플링
def _uni(rng, spec) -> float:
    """[lo, hi] → 균등, 숫자 하나 → ±값 대칭 균등."""
    if isinstance(spec, (list, tuple)):
        return float(rng.uniform(spec[0], spec[1]))
    return float(rng.uniform(-spec, spec))


def _rand_dir(rng, dim=3) -> np.ndarray:
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def _plane_dir(rng) -> np.ndarray:
    a = rng.uniform(0, 2 * math.pi)
    return np.array([math.cos(a), math.sin(a), 0.0])


MM, DEG = 1e-3, math.pi / 180


def sample_scenario(cfg: dict, rng: np.random.Generator, seed: int | None = None) -> dict:
    """config(dict, yaml 그대로) → 시나리오(dict, JSON으로 저장 가능, 단위 m·rad)."""
    g, pre, inw = cfg['geometry'], cfg['pre_weld'], cfg['in_weld']
    if g.get('joint', 'fillet_t') != 'fillet_t':
        raise ValueError(f"지원하지 않는 이음 형태: {g['joint']} (현재 fillet_t만)")
    L = g['length_mm'] * MM
    nom = cfg['nominal_pose']

    tacks = []
    n_tack = int(rng.integers(pre['tack_count'][0], pre['tack_count'][1] + 1))
    if n_tack:
        # 가접은 양 끝 근처와 사이에 고르게, 위치는 조금씩 흔듦
        for c in np.linspace(0.05, 0.95, n_tack):
            tacks.append({'s': float(np.clip(c + rng.uniform(-0.03, 0.03), 0.02, 0.98)),
                          'length': _uni(rng, pre['tack_length_mm']) * MM,
                          'height': _uni(rng, pre['tack_height_mm']) * MM})

    gap0 = _uni(rng, pre['gap_mm']) * MM
    drift = (_plane_dir(rng) * _uni(rng, inw['drift_mm']) * MM
             if rng.uniform() < inw['drift_prob'] else np.zeros(3))
    jump = (_plane_dir(rng) * _uni(rng, inw['jump_mm']) * MM
            if rng.uniform() < inw['jump_prob'] else np.zeros(3))
    span = _uni(rng, inw['drift_span'])

    return {
        'version': SCHEMA_VERSION,
        'seed': seed,
        'pass_index': 0,
        'geometry': {
            'joint': 'fillet_t', 'length': L,
            'base_width': g['base_width_mm'] * MM, 'base_thickness': g['base_thickness_mm'] * MM,
            'web_height': g['web_height_mm'] * MM, 'web_thickness': g['web_thickness_mm'] * MM,
        },
        'T_nominal': make_T(rpy_to_R(np.array(nom['rpy_deg']) * DEG),
                            np.array(nom['xyz_mm']) * MM).tolist(),
        'place': {'xyz': [_uni(rng, pre['place_xyz_mm']) * MM for _ in range(3)],
                  'rpy': [_uni(rng, pre['place_rpy_deg']) * DEG for _ in range(3)]},
        'icp': {'xyz': (_rand_dir(rng) * _uni(rng, pre['icp_xyz_mm']) * MM).tolist(),
                'rotvec': (_rand_dir(rng) * _uni(rng, pre['icp_rot_deg']) * DEG).tolist()},
        'shape': {
            'web_bow': _uni(rng, pre['web_bow_mm_per_m']) * MM * L,
            'web_wave': _uni(rng, pre['web_wave_mm']) * MM,
            'web_wave_cycles': _uni(rng, pre['wave_cycles']),
            'web_wave_phase': float(rng.uniform(0, 2 * math.pi)),
            'base_bow': _uni(rng, pre['base_bow_mm_per_m']) * MM * L,
            'base_wave': _uni(rng, pre['base_wave_mm']) * MM,
            'base_wave_cycles': _uni(rng, pre['wave_cycles']),
            'base_wave_phase': float(rng.uniform(0, 2 * math.pi)),
            'gap0': gap0,
            'gap1': max(0.0, gap0 + _uni(rng, pre['gap_taper_mm']) * MM),
            'tacks': tacks,
        },
        'slip': {
            'drift': drift.tolist(), 'drift_s0': float(rng.uniform(0.0, 1.0 - span)), 'drift_span': span,
            'jump': jump.tolist(), 'jump_s': float(rng.uniform(0.1, 0.9)),
        },
        'heat': {
            'base_angle': _uni(rng, inw['base_angle_deg']) * DEG,
            'web_tilt': _uni(rng, inw['web_tilt_deg']) * DEG,
            'length': _uni(rng, inw['distort_length_mm']) * MM,
            'bow': _uni(rng, inw['bow_mm']) * MM,
            'gap_close': _uni(rng, inw['gap_close_mm']) * MM,
        },
        'prior': {'base_angle': 0.0, 'web_tilt': 0.0, 'bow': 0.0},   # 앞 패스의 잔류 변형
        'sensors': {
            'laser_lookahead': cfg['sensors']['laser_lookahead_mm'] * MM,
            'laser_noise': cfg['sensors']['laser_noise_mm'] * MM,
            'laser_dropout': float(cfg['sensors']['laser_dropout']),
        },
        'expert': copy.deepcopy(cfg.get('expert', {})),
    }


def next_pass(scn: dict, cfg: dict, rng: np.random.Generator) -> dict:
    """용접 후: 이 패스의 잔류 변형을 다음 패스(같은 이음매 다층 용접)의 용접 전 형상으로 넘긴다.

    모재는 지그에서 빼지 않으므로 배치 오차는 그대로, 재스캔이므로 ICP 잔차와 용접 중 오차는 새로 뽑는다.
    루트 틈새와 가접은 앞 패스 비드로 메워졌다고 보고 0으로 둔다.
    """
    new = sample_scenario(cfg, rng, seed=scn.get('seed'))
    for k in ('geometry', 'T_nominal', 'place', 'shape'):
        new[k] = copy.deepcopy(scn[k])
    new['shape'].update(gap0=0.0, gap1=0.0, tacks=[])
    h, p = scn['heat'], scn['prior']
    new['prior'] = {'base_angle': p['base_angle'] + h['base_angle'],
                    'web_tilt': p['web_tilt'] + h['web_tilt'],
                    'bow': p['bow'] + h['bow']}
    new['pass_index'] = scn.get('pass_index', 0) + 1
    return new


def save_scenario(scn: dict, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(scn, f, indent=1, ensure_ascii=False)


def load_scenario(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        scn = json.load(f)
    if scn.get('version') != SCHEMA_VERSION:
        raise ValueError(f'시나리오 버전 {scn.get("version")} (지원: {SCHEMA_VERSION})')
    return scn


# ----------------------------------------------------------------------------- 오차 모델
class PartErrorModel:
    """시나리오 하나의 실제·추정 경로, 정답, 모재 변형, 레이저 측정."""

    def __init__(self, scn: dict):
        self.scn = scn
        self.L = scn['geometry']['length']
        self.set_nominal(np.array(scn['T_nominal']))

    def set_nominal(self, T_nominal) -> None:
        """모재 명목 배치(로봇 베이스 기준)를 바꾼다. 자동 배치 때 사용."""
        self.scn['T_nominal'] = np.asarray(T_nominal, dtype=float).tolist()
        sc = self.scn
        self.T_place = np.asarray(T_nominal) @ make_T(rpy_to_R(sc['place']['rpy']), sc['place']['xyz'])
        self.T_est = self.T_place @ make_T(rotvec_to_R(sc['icp']['rotvec']), sc['icp']['xyz'])

    # --- 진행도의 함수들 (모재 좌표계, m / rad)
    def web_lateral(self, s):
        sh = self.scn['shape']
        return (sh['web_bow'] * 4 * s * (1 - s)
                + sh['web_wave'] * np.sin(2 * np.pi * sh['web_wave_cycles'] * s + sh['web_wave_phase']))

    def base_height(self, s, s_arc):
        sh, h, p = self.scn['shape'], self.scn['heat'], self.scn['prior']
        bow = sh['base_bow'] + p['bow'] + h['bow'] * s_arc   # 세로 휨은 진행도에 비례해 자람
        return (bow * 4 * s * (1 - s)
                + sh['base_wave'] * np.sin(2 * np.pi * sh['base_wave_cycles'] * s + sh['base_wave_phase']))

    def gap(self, s, s_arc):
        """틈새. 아크 앞쪽은 용접이 진행된 만큼(s_arc) 닫히고, 멀수록 더 닫힌다(토치 위치에서 절반, 끝에서 최대).
        아크가 지나간 곳은 지나갈 때의 값으로 굳는다."""
        sh, h = self.scn['shape'], self.scn['heat']
        s = np.asarray(s, dtype=float)
        g = sh['gap0'] + (sh['gap1'] - sh['gap0']) * s
        ahead = 0.5 + 0.5 * np.maximum(0.0, s - s_arc) / max(1e-6, 1.0 - s_arc)
        close = np.where(s >= s_arc, s_arc * ahead, 0.5 * s)
        return np.maximum(0.0, g - h['gap_close'] * close)

    def _heat_profile(self, s, s_arc):
        """아크가 지나간 뒤쪽에서 0 → 1로 다가가는 각변형 진행 비율."""
        behind = np.maximum(0.0, s_arc - s) * self.L
        return 1.0 - np.exp(-behind / self.scn['heat']['length'])

    def base_angle(self, s, s_arc):
        return self.scn['prior']['base_angle'] + self.scn['heat']['base_angle'] * self._heat_profile(s, s_arc)

    def web_tilt(self, s, s_arc):
        return self.scn['prior']['web_tilt'] + self.scn['heat']['web_tilt'] * self._heat_profile(s, s_arc)

    def slip(self, s_arc) -> np.ndarray:
        sl = self.scn['slip']
        r = np.clip((s_arc - sl['drift_s0']) / sl['drift_span'], 0.0, 1.0)
        return np.asarray(sl['drift']) * r + np.asarray(sl['jump']) * float(s_arc >= sl['jump_s'])

    def tack_height(self, s):
        s = np.asarray(s, dtype=float)
        h = np.zeros_like(s)
        for t in self.scn['shape']['tacks']:
            u = (s - t['s']) * self.L / t['length']
            h = np.maximum(h, t['height'] * np.sqrt(np.maximum(0.0, 1 - (2 * u) ** 2)))
        return h

    # --- 경로
    def _seam_part(self, s, s_arc):
        """실제 이음매 점 (모재 좌표계, 밀림 포함). 각변형은 이음매 선을 축으로 돌므로 점은 그대로."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        return np.stack([s * self.L, self.web_lateral(s), self.base_height(s, s_arc)], axis=-1) + self.slip(s_arc)

    def seam_true(self, s, s_arc):
        """실제 경로: (점 (n,3), 이음매 좌표계 (n,3,3)) 로봇 베이스 기준."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        h = 1e-3
        P = self._seam_part(s, s_arc)
        tx = _normalize(self._seam_part(s + h, s_arc) - self._seam_part(s - h, s_arc))
        th, ps = self.base_angle(s, s_arc), self.web_tilt(s, s_arc)
        n_base = np.stack([np.zeros_like(th), np.sin(th), np.cos(th)], axis=-1)
        n_web = np.stack([np.zeros_like(ps), -np.cos(ps), -np.sin(ps)], axis=-1)
        b = n_base + n_web
        z = _normalize(b - np.sum(b * tx, axis=-1, keepdims=True) * tx)
        y = np.cross(z, tx)
        R_part = np.stack([tx, y, z], axis=-1)
        Rw, tw = self.T_place[:3, :3], self.T_place[:3, 3]
        return P @ Rw.T + tw, Rw @ R_part

    def seam_est(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        P = np.stack([s * self.L, np.zeros_like(s), np.zeros_like(s)], axis=-1)
        Rw, tw = self.T_est[:3, :3], self.T_est[:3, 3]
        return P @ Rw.T + tw, np.broadcast_to(Rw @ R_SEAM_NOM, (len(s), 3, 3)).copy()

    def label(self, s: float, s_arc: float) -> dict:
        """진행도 s에서의 정답: 추정 경로 좌표계 기준 위치·각도 보정, 틈새, 가접 높이."""
        p_t, R_t = self.seam_true(s, s_arc)
        p_e, R_e = self.seam_est(s)
        e = R_e[0].T @ (p_t[0] - p_e[0])
        return {'e': e, 'rotvec': R_to_rotvec(R_e[0].T @ R_t[0]),
                'gap': float(self.gap(np.array([s]), s_arc)[0]), 'tack': float(self.tack_height([s])[0]),
                'p_true': p_t[0], 'R_true': R_t[0], 'p_est': p_e[0], 'R_est': R_e[0]}

    def laser(self, s_torch: float, s_arc: float, rng: np.random.Generator):
        """레이저 라인 센서: 토치 앞 lookahead 지점의 이음매 좌우·높이 편차 [m] (추정 경로 기준).

        측정 실패(스패터·연기, 이음매 끝을 넘음)는 NaN.
        """
        se = self.scn['sensors']
        s_look = s_torch + se['laser_lookahead'] / self.L
        if s_look > 1.0 or rng.uniform() < se['laser_dropout']:
            return np.array([np.nan, np.nan])
        lab = self.label(s_look, s_arc)
        meas = np.array([lab['e'][1], lab['e'][2] + lab['tack']])
        return meas + rng.normal(scale=se['laser_noise'], size=2)

    # --- 모재 메시 변형 (Isaac Sim 표시용)
    def deform_points(self, P, plate, s_arc) -> np.ndarray:
        """명목 모재 꼭짓점 (n,3, 모재 좌표계) → 변형된 꼭짓점 (로봇 베이스 기준)."""
        P = np.asarray(P, dtype=float)
        plate = np.asarray(plate)
        s = np.clip(P[:, 0] / self.L, 0.0, 1.0)
        out = P.copy()
        dy = self.web_lateral(s)
        dz = self.base_height(s, s_arc)
        web = plate == WEB
        out[web, 1] += dy[web]
        out[web, 2] += self.gap(s, s_arc)[web]        # 웹을 틈새만큼 띄움
        out[:, 2] += dz
        # 이음매 선 (y = dy, z = dz)을 축으로 회전: 바닥 판 용접 쪽 날개(y<0)는 들리고, 웹은 기울어짐
        th, ps = self.base_angle(s, s_arc), self.web_tilt(s, s_arc)
        for mask, ang in (((plate == BASE) & (P[:, 1] < 0), -th), (web, ps)):
            yr, zr, a = out[mask, 1] - dy[mask], out[mask, 2] - dz[mask], ang[mask]
            out[mask, 1] = dy[mask] + yr * np.cos(a) - zr * np.sin(a)
            out[mask, 2] = dz[mask] + yr * np.sin(a) + zr * np.cos(a)
        out += self.slip(s_arc)
        Rw, tw = self.T_place[:3, :3], self.T_place[:3, 3]
        return out @ Rw.T + tw

    def nominal_meshes(self, nx: int = 60):
        """바닥 판·웹 판의 명목 메시. 반환: (꼭짓점 (n,3), 판 번호 (n,), 면 꼭짓점 수, 면 인덱스)."""
        g = self.scn['geometry']
        W, tb, H, tw = g['base_width'], g['base_thickness'], g['web_height'], g['web_thickness']
        # 단면 (y, z) 다각형. 바닥 판은 y=0에 꼭짓점을 둬서 각변형 꺾임이 생기게 함
        sections = [(BASE, [(-W / 2, -tb), (0, -tb), (W / 2, -tb), (W / 2, 0), (0, 0), (-W / 2, 0)]),
                    (WEB, [(0, 0), (tw, 0), (tw, H), (0, H)])]
        pts, plate, counts, idx = [], [], [], []
        xs = np.linspace(0.0, self.L, nx + 1)
        for pid, sec in sections:
            m = len(sec)
            o = len(pts)
            for x in xs:
                for (y, z) in sec:
                    pts.append((x, y, z))
                    plate.append(pid)
            for i in range(nx):
                for j in range(m):
                    a, b = o + i * m + j, o + i * m + (j + 1) % m
                    counts.append(4)
                    idx += [a, b, b + m, a + m]
            counts += [m, m]
            idx += [o + j for j in reversed(range(m))]
            idx += [o + nx * m + j for j in range(m)]
        return np.array(pts), np.array(plate), np.array(counts), np.array(idx)


def zero_errors(scn: dict) -> dict:
    """모든 오차를 0으로 둔 사본 (형상·배치·센서·전문가 설정은 유지, 레이저 노이즈·실패도 0)."""
    z = copy.deepcopy(scn)
    z['place'] = {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}
    z['icp'] = {'xyz': [0.0, 0.0, 0.0], 'rotvec': [0.0, 0.0, 0.0]}
    for k in ('web_bow', 'web_wave', 'base_bow', 'base_wave', 'gap0', 'gap1'):
        z['shape'][k] = 0.0
    z['shape']['tacks'] = []
    z['slip'].update(drift=[0.0, 0.0, 0.0], jump=[0.0, 0.0, 0.0])
    z['heat'].update(base_angle=0.0, web_tilt=0.0, bow=0.0, gap_close=0.0)
    z['prior'] = {'base_angle': 0.0, 'web_tilt': 0.0, 'bow': 0.0}
    z['sensors'].update(laser_noise=0.0, laser_dropout=0.0)
    return z


# 오차별 단독 검증: 하나만 크게 과장해서 켠다 (계획 문서 "오차별 단독 검증" 표와 같은 순서)
SINGLE_ERRORS = {
    'place': ('지그 배치 오차', {'place': {'xyz': [10 * MM, -10 * MM, 5 * MM], 'rpy': [0.0, 0.0, 3 * DEG]}}),
    'icp': ('ICP 잔차', {'icp': {'xyz': [0.0, 3 * MM, 0.0], 'rotvec': [0.0, 0.0, 0.0]}}),
    'web_straightness': ('웹 판 직진도', {'shape': {'web_bow': 5 * MM}}),
    'base_flatness': ('바닥 판 평평도', {'shape': {'base_bow': 5 * MM}}),
    'gap': ('루트 틈새', {'shape': {'gap0': 2 * MM, 'gap1': 0.0}}),
    'tack': ('가접', {'shape': {'tacks': [{'s': s, 'length': 20 * MM, 'height': 3 * MM} for s in (0.25, 0.5, 0.75)]}}),
    'drift': ('서서히 밀림', {'slip': {'drift': [0.0, 10 * MM, 0.0], 'drift_s0': 0.3, 'drift_span': 0.3}}),
    'jump': ('갑자기 튐', {'slip': {'jump': [0.0, 5 * MM, 0.0], 'jump_s': 0.5}}),
    'base_angle': ('바닥 판 각변형', {'heat': {'base_angle': 5 * DEG, 'length': 30 * MM}}),
    'web_tilt': ('웹 기울어짐', {'heat': {'web_tilt': 3 * DEG, 'length': 30 * MM}}),
    'bow': ('세로 휨', {'heat': {'bow': 5 * MM}}),
    'gap_close': ('틈새 닫힘', {'shape': {'gap0': 2 * MM, 'gap1': 2 * MM}, 'heat': {'gap_close': 1.5 * MM}}),
    'residual': ('잔류 변형 (다음 패스)', {'prior': {'base_angle': 3 * DEG, 'web_tilt': 2 * DEG, 'bow': 3 * MM},
                                       'pass_index': 1}),
}


def single_error(scn: dict, name: str) -> dict:
    """오차 하나만 과장해서 켠 시나리오. 나머지 오차는 0."""
    if name not in SINGLE_ERRORS:
        raise ValueError(f'알 수 없는 오차: {name} (가능: {", ".join(SINGLE_ERRORS)})')
    z = zero_errors(scn)
    for key, val in SINGLE_ERRORS[name][1].items():
        if isinstance(val, dict):
            z[key].update(copy.deepcopy(val))
        else:
            z[key] = val
    z['single_error'] = name
    return z


def signature(scn: dict, points=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[dict]:
    """토치가 진행도 s에 있을 때(s_arc = s)의 정답 [mm, deg]. 단독 검증에서 그래프와 비교할 기대값."""
    m = PartErrorModel(copy.deepcopy(scn))
    out = []
    for s in points:
        lab = m.label(s, s)
        out.append({'s': s, 'e_y': lab['e'][1] / MM, 'e_z': lab['e'][2] / MM,
                    'work': lab['rotvec'][0] / DEG, 'travel': lab['rotvec'][1] / DEG,
                    'gap': lab['gap'] / MM, 'tack': lab['tack'] / MM})
    return out


def summarize(scn: dict, n: int = 101) -> dict:
    """토치가 진행도 s에 있을 때(s_arc = s)의 정답 크기 요약 [mm, deg]. 분포를 실측과 비교할 때 사용."""
    m = PartErrorModel(copy.deepcopy(scn))
    labs = [m.label(s, s) for s in np.linspace(0, 1, n)]
    e = np.array([lab['e'] for lab in labs]) / MM
    r = np.array([lab['rotvec'] for lab in labs]) / DEG
    g = np.array([lab['gap'] for lab in labs]) / MM
    return {'lateral_max_mm': float(np.max(np.abs(e[:, 1]))), 'height_max_mm': float(np.max(np.abs(e[:, 2]))),
            'lateral_change_mm': float(np.ptp(e[:, 1])), 'height_change_mm': float(np.ptp(e[:, 2])),
            'work_angle_max_deg': float(np.max(np.abs(r[:, 0]))), 'travel_angle_max_deg': float(np.max(np.abs(r[:, 1]))),
            'gap_min_mm': float(np.min(g)), 'gap_max_mm': float(np.max(g))}
