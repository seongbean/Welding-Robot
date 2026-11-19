"""모재 오차 모델: 용접 전·중·후 오차를 분포에서 뽑고, 실제 경로·추정 경로·정답(보정값)을 계산한다.

numpy만 쓴다. Isaac Sim 파이썬(isaac_opti3_receiver.py)과 conda 환경(expert_to_sim.py) 양쪽에서 import한다.

경로 세 가지 (계획 문서 "경로 세 가지와 정답의 정의")
- 명목 경로 p_nom(s): CAD 모재 좌표계의 이음매 (s*L, 0, 0)
- 추정 경로 p_est(s) = T_est · p_nom(s), T_est = T_place · ICP 잔차. 용접 전에 한 번 정해지고 고정
- 실제 경로 p_true(s, s_arc) = T_place · [p_nom(s) + 형상 오차(s) + 열변형(s, s_arc) + 밀림(s_arc)]
정답 e = R_est(s)^T (p_true - p_est): y = 좌우, z = 높이. 각도 정답 = log(R_est^T R_true) (x = 작업각, y = 진행각)

모재 좌표계 (필릿 두 군 공통): 원점 = 이음매 시작점, x = 이음매 방향, 바닥 판 윗면 z = 0,
웹 판 용접 쪽 면 y = 0 (웹은 y >= 0, 토치는 -y 쪽에서 접근). 이음매 좌표계 z = 두 판 법선의 이등분.
시편군 (geometry.joint)
- fillet_t: T이음 필릿. 바닥 판이 웹 양쪽으로 뻗는다 (폭 base_width는 양쪽 합).
- fillet_corner: ㄱ자 코너 필릿. 웹이 바닥 판 끝 위에 바깥 면을 맞춰 선다. 바닥 판은 토치 쪽으로만 뻗는다
  (base_width = 토치 쪽 날개 폭, 웹 밑은 웹 두께만큼). 안쪽 모서리 필릿이라 이음매·정답·단면 센서 식은 T와 같고,
  다른 것은 모재 모양과 오차 범위(config families.fillet_corner)다.
- butt_v: 맞대기 V홈 (아래보기 1G, 루트 패스). 두 판이 홈을 사이에 두고 같은 평면에 눕는다.
  모재 좌표계: 원점 = 루트 가운데(루트 면 윗높이), x = 이음매, y = 오른쪽 판 쪽, z = 위 (판 윗면 z = 두께 - 루트 면).
  BASE = 왼쪽 판(-y), WEB = 오른쪽 판(+y). 홈 벽(개선 면)은 연직에서 bevel만큼 벌어지고, 루트 틈새는 두 루트 끝 사이.
  이음매 좌표계 z = 두 개선 면 법선의 이등분(명목 = 위). 이름은 필릿 것을 그대로 쓰되 뜻을 바꿔 읽는다:
  web_lateral = 이음선 좌우 직진도, base_height = 판 높이 휨, heat.base_angle = 양쪽 판 각변형(같은 크기로 날개가 들림,
  이등분선은 그대로라 정답에 거의 영향 없음), heat.web_tilt = 좌우 판 각변형 차이(오른쪽 - 왼쪽, 작업각 = 절반),
  shape.hi_lo = 단차(오른쪽 판이 왼쪽보다 높은 양, 맞대기만. 단면 센서는 홈 양옆 판 윗면 높이 차로 재서 좌우를 고친다). 다층(채움 패스)의 목표 높이 변화는 다루지 않는다.
단위: 내부 계산은 m, rad. config(yaml)는 mm, deg.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np

BASE, WEB = 0, 1
SCHEMA_VERSION = 1
JOINTS = ('fillet_t', 'fillet_corner', 'butt_v')   # 지원하는 시편군 (geometry.joint)


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


def seam_nominal_R(joint: str) -> np.ndarray:
    """명목 이음매 좌표계 (모재 좌표계 기준). 맞대기 V홈은 z = 위, y = 오른쪽 판 쪽이라 단위 행렬."""
    return np.eye(3) if joint == 'butt_v' else R_SEAM_NOM


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
    joint = g.get('joint', 'fillet_t')
    if joint not in JOINTS:
        raise ValueError(f"지원하지 않는 이음 형태: {joint} (가능: {', '.join(JOINTS)})")
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

    scn = {
        'version': SCHEMA_VERSION,
        'seed': seed,
        'pass_index': 0,
        'geometry': _geometry(g, joint, L),
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
        'sensors': sensors_from_cfg(cfg['sensors']),
        'expert': copy.deepcopy(cfg.get('expert', {})),
    }
    # 웹 직각도 (조립 때 웹이 90°에서 기운 각, 용접 전부터 있음). 설정한 시편군만 뽑는다
    # (맨 끝에 뽑아서 설정 없는 T 필릿의 난수 순서는 그대로)
    if pre.get('web_square_deg', 0):
        scn['shape']['web_square'] = _uni(rng, pre['web_square_deg']) * DEG
    if pre.get('hi_lo_mm', 0):          # 단차 (맞대기만)
        scn['shape']['hi_lo'] = _uni(rng, pre['hi_lo_mm']) * MM
    return scn


def _geometry(g: dict, joint: str, L: float) -> dict:
    if joint == 'butt_v':
        return {'joint': joint, 'length': L, 'plate_width': g['plate_width_mm'] * MM,
                'thickness': g['thickness_mm'] * MM, 'bevel': g['bevel_deg'] * DEG, 'root_face': g['root_face_mm'] * MM}
    return {'joint': joint, 'length': L,
            'base_width': g['base_width_mm'] * MM, 'base_thickness': g['base_thickness_mm'] * MM,
            'web_height': g['web_height_mm'] * MM, 'web_thickness': g['web_thickness_mm'] * MM}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def family_cfg(cfg: dict, family: str | None = None) -> dict:
    """config에서 시편군 하나의 설정을 만든다. 기본 섹션(T 필릿)이 공통값이고, families.<군>이 덮어쓴다.
    family가 None이거나 기본 군(geometry.joint)이면 기본 섹션 그대로. 결과에는 families 섹션이 없다."""
    fams = cfg.get('families', {}) or {}
    out = {k: copy.deepcopy(v) for k, v in cfg.items() if k != 'families'}
    if family is None or family == out['geometry'].get('joint', 'fillet_t'):
        return out
    if family not in fams:
        raise ValueError(f'config에 시편군 {family}가 없음 (있는 것: {", ".join([out["geometry"]["joint"], *fams])})')
    geom = copy.deepcopy(fams[family].get('geometry', {}))
    out = _merge(out, fams[family])
    if 'plate_width_mm' in geom:            # 모양이 다른 이음(맞대기): 필릿 치수를 섞지 않는다
        out['geometry'] = geom
    out['geometry']['joint'] = family
    return out


def sensors_from_cfg(c: dict) -> dict:
    """config sensors 섹션 (mm, 비율) → 시나리오 sensors (m)."""
    return {'laser_lookahead': c['laser_lookahead_mm'] * MM, 'laser_noise': c['laser_noise_mm'] * MM,
            'laser_dropout': float(c['laser_dropout']),
            'laser_point_noise': c.get('laser_point_noise_mm', 0.2) * MM,
            'laser_point_dropout': float(c.get('laser_point_dropout', 0.1)),
            'laser_patch_along': c.get('laser_patch_along_mm', 10.0) * MM,
            'laser_patch_across': c.get('laser_patch_across_mm', 15.0) * MM,
            'laser_point_step': c.get('laser_point_step_mm', 1.0) * MM,
            'prescan': c.get('prescan_mm', 0.0) * MM, 'prescan_step': c.get('prescan_step_mm', 0.5) * MM}


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
        self.joint = scn['geometry'].get('joint', 'fillet_t')
        self.butt = self.joint == 'butt_v'
        self.R_nom = seam_nominal_R(self.joint)
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
        """웹 기울기 = 조립 직각도(shape.web_square, 없으면 0) + 앞 패스 잔류 + 이번 패스 열변형."""
        return (self.scn['shape'].get('web_square', 0.0) + self.scn['prior']['web_tilt']
                + self.scn['heat']['web_tilt'] * self._heat_profile(s, s_arc))

    def plate_angles(self, s, s_arc):
        """맞대기: (왼쪽 판, 오른쪽 판) 각변형 [rad], 날개가 들리는 쪽이 +. 양쪽 = base_angle, 차이 = web_tilt."""
        th, d = self.base_angle(s, s_arc), self.web_tilt(s, s_arc)
        return th - d / 2, th + d / 2

    def hi_lo(self) -> float:
        return float(self.scn['shape'].get('hi_lo', 0.0))

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
        z = self.base_height(s, s_arc) + (self.hi_lo() / 2 if self.butt else 0.0)   # 맞대기: 두 루트 끝의 가운데
        return np.stack([s * self.L, self.web_lateral(s), z], axis=-1) + self.slip(s_arc)

    def _face_normals(self, s, s_arc):
        """두 판 용접 면 법선 (모재 좌표계, 홈/이음매 쪽을 향함). 필릿: 바닥 판 윗면, 웹 용접 쪽 면.
        맞대기: 왼쪽·오른쪽 개선 면 (각변형으로 날개가 들리면 면이 연직 쪽으로 선다)."""
        if self.butt:
            b = self.scn['geometry']['bevel']
            aL, aR = self.plate_angles(s, s_arc)
            aL, aR = b - aL, b - aR
            z0 = np.zeros_like(aL)
            return (np.stack([z0, np.cos(aL), np.sin(aL)], axis=-1),
                    np.stack([z0, -np.cos(aR), np.sin(aR)], axis=-1))
        th, ps = self.base_angle(s, s_arc), self.web_tilt(s, s_arc)
        return (np.stack([np.zeros_like(th), np.sin(th), np.cos(th)], axis=-1),
                np.stack([np.zeros_like(ps), -np.cos(ps), -np.sin(ps)], axis=-1))

    def _seam_geom(self, s, s_arc):
        """실제 이음매 점, 이음매 좌표계, 바닥 판 면 법선, 웹 판 용접 쪽 면 법선 (모두 로봇 베이스 기준)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        h = 1e-3
        P = self._seam_part(s, s_arc)
        tx = _normalize(self._seam_part(s + h, s_arc) - self._seam_part(s - h, s_arc))
        n_base, n_web = self._face_normals(s, s_arc)
        b = n_base + n_web
        z = _normalize(b - np.sum(b * tx, axis=-1, keepdims=True) * tx)
        y = np.cross(z, tx)
        R_part = np.stack([tx, y, z], axis=-1)
        Rw, tw = self.T_place[:3, :3], self.T_place[:3, 3]
        return P @ Rw.T + tw, Rw @ R_part, n_base @ Rw.T, n_web @ Rw.T

    def seam_true(self, s, s_arc):
        """실제 경로: (점 (n,3), 이음매 좌표계 (n,3,3)) 로봇 베이스 기준."""
        P, R, _, _ = self._seam_geom(s, s_arc)
        return P, R

    def seam_est(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        P = np.stack([s * self.L, np.zeros_like(s), np.zeros_like(s)], axis=-1)
        Rw, tw = self.T_est[:3, :3], self.T_est[:3, 3]
        return P @ Rw.T + tw, np.broadcast_to(Rw @ self.R_nom, (len(s), 3, 3)).copy()

    def label(self, s: float, s_arc: float) -> dict:
        """진행도 s에서의 정답: 추정 경로 좌표계 기준 위치·각도 보정, 틈새, 가접 높이."""
        p_t, R_t = self.seam_true(s, s_arc)
        p_e, R_e = self.seam_est(s)
        e = R_e[0].T @ (p_t[0] - p_e[0])
        return {'e': e, 'rotvec': R_to_rotvec(R_e[0].T @ R_t[0]),
                'gap': float(self.gap(np.array([s]), s_arc)[0]), 'tack': float(self.tack_height([s])[0]),
                'p_true': p_t[0], 'R_true': R_t[0], 'p_est': p_e[0], 'R_est': R_e[0]}

    def laser(self, s_torch: float, s_arc: float, rng: np.random.Generator):
        """이음매 단면 센서 (손목 depth 카메라): 토치 앞 lookahead 지점의
        [좌우, 높이, 틈새 폭 [m], 작업각, 진행각 [rad]].

        카메라가 보는 작은 패치(이음매 방향 laser_patch_along, 판마다 laser_patch_across)의 두 판 면 위 점을 뽑아
        점마다 노이즈(laser_point_noise)와 측정 구멍(laser_point_dropout)을 넣고, 판마다 평면을 맞춘다.
        두 평면의 교선이 이음매 뿌리선이다.
        - 좌우·높이: 추정 경로 좌표계에서 본 뿌리선 위치 (가접 위에서는 높이에 가접 포함)
        - 작업각·진행각: 두 평면이 만드는 이음매 좌표계(x = 교선, z = 두 법선의 이등분)를 추정 좌표계에서 본 회전
          (정답 rotvec과 같은 정의, 측정 위치는 토치 앞)
        - 틈새 폭: 정답 + 노이즈(laser_noise)
        실기에서는 카메라가 토치에 붙어 있으므로 토치 기준으로 맞춘 값에 측정 순간의 토치 자세(FK)를 더해 같은 값을 만든다.
        전체 측정 실패(laser_dropout, 이음매 끝을 넘음)는 NaN 다섯 개.
        """
        se = self.scn['sensors']
        s_look = s_torch + se['laser_lookahead'] / self.L
        if s_look > 1.0 or rng.uniform() < se['laser_dropout']:
            return np.full(5, np.nan)
        return self.section_fit(s_look, s_arc, rng)

    def prescan(self, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """용접 전 시작 구간 스캔: 아크를 켜기 전(s_arc = 0)에 손목 카메라로 이음매 시작 prescan 길이를 훑는다.

        단면 센서는 토치 앞 lookahead에서만 재므로 스캔이 없으면 시작 구간(첫 가접이 있는 곳)은 한 번도 보이지 않는다.
        반환: (측정 위치 진행도 (n,), 측정 (n,5) laser와 같은 형식, 실패는 NaN). 설정이 없거나 0이면 빈 배열.
        """
        se = self.scn['sensors']
        length = float(se.get('prescan', 0.0))
        if length <= 0:
            return np.zeros(0), np.zeros((0, 5))
        step = float(se.get('prescan_step', 0.0005))
        s_pts = np.arange(0.0, min(length, self.L) + 1e-9, step) / self.L
        meas = np.full((len(s_pts), 5), np.nan)
        for k, sp in enumerate(s_pts):
            if rng is None or rng.uniform() >= se['laser_dropout']:
                meas[k] = self.section_fit(float(sp), 0.0, rng)
        return s_pts, meas

    def section_fit(self, s_look: float, s_arc: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """s_look 위치 단면 패치의 평면 맞춤 측정 (laser 설명 참고). rng=None이면 노이즈·구멍 없음."""
        se = self.scn['sensors']
        along = se.get('laser_patch_along', 0.01)
        across = se.get('laser_patch_across', 0.015)
        step = se.get('laser_point_step', 0.001)
        sig = se.get('laser_point_noise', 0.0) if rng is not None else 0.0
        drop = se.get('laser_point_dropout', 0.0) if rng is not None else 0.0
        a = np.arange(-along / 2, along / 2 + 1e-9, step)
        s_pts = np.clip(s_look + a / self.L, 0.0, 1.0)
        P, R, nb, nw = self._seam_geom(s_pts, s_arc)
        tx = R[:, :, 0]
        gap = float(self.gap(np.array([s_look]), s_arc)[0])
        if self.butt:
            base, web = self._butt_face_points(P, tx, nb, nw, gap, across, step)
        else:
            ub = _normalize(np.cross(tx, nb))        # 바닥 판 면 위, 토치 쪽으로
            uw = _normalize(np.cross(nw, tx))        # 웹 판 면 위, 위로
            r = np.arange(step, across + 1e-9, step)
            base = (P[:, None, :] + ub[:, None, :] * r[None, :, None]).reshape(-1, 3)
            web = (P[:, None, :] + uw[:, None, :] * (gap + r)[None, :, None]).reshape(-1, 3)   # 틈새 위로만 보임
        planes = []
        for pts in (base, web):
            if rng is not None:
                pts = pts + rng.normal(scale=sig, size=pts.shape) if sig > 0 else pts
                if drop > 0:
                    pts = pts[rng.uniform(size=len(pts)) >= drop]
            if len(pts) < 6:
                return np.full(5, np.nan)
            c = pts.mean(axis=0)
            n = np.linalg.svd(pts - c, full_matrices=False)[2][-1]
            planes.append((c, n))
        (cb, nb_f), (cw, nw_f) = planes
        # 법선 방향은 추정 모재 좌표계로 정함 (로봇이 아는 값): 필릿 바닥 판 +z, 웹 판 -y / 맞대기 왼쪽 +y, 오른쪽 -y
        Re_part = self.T_est[:3, :3]
        if nb_f @ (Re_part[:, 1] if self.butt else Re_part[:, 2]) < 0:
            nb_f = -nb_f
        if nw_f @ -Re_part[:, 1] < 0:
            nw_f = -nw_f
        p_e, R_e = self.seam_est(np.array([s_look]))
        p_e, R_e = p_e[0], R_e[0]
        d = np.cross(nb_f, nw_f)
        d /= np.linalg.norm(d)
        if d @ R_e[:, 0] < 0:
            d = -d
        # 두 평면의 교선 위에서 추정 경로 점에 가장 가까운 점
        A = np.stack([nb_f, nw_f, d])
        root = np.linalg.solve(A, np.array([nb_f @ cb, nw_f @ cw, d @ p_e]))
        bis = nb_f + nw_f
        z = bis - (bis @ d) * d
        z /= np.linalg.norm(z)
        R_m = np.stack([d, np.cross(z, d), z], axis=-1)
        g = gap + (rng.normal(scale=se['laser_noise']) if rng is not None else 0.0)
        if self.butt:
            # 두 개선 면의 교선(꼭짓점)은 틈새가 있으면 루트보다 아래에 있다: 잰 틈새와 잰 개선각으로 루트 높이로 올린다
            half = (math.pi - math.acos(float(np.clip(nb_f @ nw_f, -1.0, 1.0)))) / 2
            root = root + z * max(0.0, g) / 2 / max(1e-3, math.tan(half))
            # 단차: 두 개선 면 교선은 단차 h만큼 옆으로 밀린다(-h·tan(개선각)/2). 홈 양옆 판 윗면 높이 차로 h를 재서 되돌린다
            h_meas = self._butt_top_step(s_pts, s_arc, z, across, step, sig, drop, rng)
            if h_meas is None:
                return np.full(5, np.nan)
            root = root + np.cross(z, d) * h_meas * math.tan(half) / 2
        e = R_e.T @ (root - p_e)
        rv = R_to_rotvec(R_e.T @ R_m)
        tack = float(self.tack_height([s_look])[0])
        return np.array([e[1], e[2] + tack, max(0.0, g), rv[0], rv[1]])

    def _butt_top_step(self, s_pts, s_arc, z, across, step, sig, drop, rng):
        """맞대기 단차 측정: 홈 양옆 판 윗면(홈 위 모서리에서 바깥으로 across)의 점 평균 높이 차 (오른쪽 - 왼쪽, z 방향).
        각변형은 아크 뒤에서만 생기고 대칭 각변형은 양쪽이 같이 올라가므로, 토치 앞 측정에서는 거의 단차만 남는다."""
        g = self.scn['geometry']
        h = g['thickness'] - g['root_face']
        a = h * math.tan(g['bevel'])
        r = np.arange(step, across + 1e-9, step)
        x = s_pts * self.L
        tops = []
        for pid, side in ((BASE, -1.0), (WEB, 1.0)):
            X, R = np.meshgrid(x, r, indexing='ij')
            pts = np.stack([X.ravel(), side * (a + R.ravel()), np.full(X.size, h)], axis=-1)
            pts = self.deform_points(pts, np.full(len(pts), pid), s_arc)
            if rng is not None:
                pts = pts + rng.normal(scale=sig, size=pts.shape) if sig > 0 else pts
                if drop > 0:
                    pts = pts[rng.uniform(size=len(pts)) >= drop]
            if len(pts) < 3:
                return None
            tops.append(pts.mean(axis=0))
        return float((tops[1] - tops[0]) @ z)

    def _butt_face_points(self, P, tx, nL, nR, gap, across, step):
        """맞대기 두 개선 면 위의 점 (로봇 베이스 기준). 루트 끝에서 위로 면을 따라 across까지 (면 길이를 넘지 않음)."""
        g = self.scn['geometry']
        Rw = self.T_place[:3, :3]
        y, z = Rw[:, 1], Rw[:, 2]
        face = (g['thickness'] - g['root_face']) / math.cos(g['bevel'])
        r = np.arange(step, min(across, face) + 1e-9, step)
        out = []
        for n, side in ((nL, -1.0), (nR, 1.0)):            # 법선은 이미 로봇 베이스 기준
            u = _normalize(np.cross(tx, n))
            u = np.where(np.sum(u * z, axis=-1, keepdims=True) < 0, -u, u)       # 면을 따라 위로
            edge = P + y * side * gap / 2 + z * side * self.hi_lo() / 2              # 그 판의 루트 끝
            out.append((edge[:, None, :] + u[:, None, :] * r[None, :, None]).reshape(-1, 3))
        return out

    # --- 모재 메시 변형 (Isaac Sim 표시용)
    def deform_points(self, P, plate, s_arc) -> np.ndarray:
        """명목 모재 꼭짓점 (n,3, 모재 좌표계) → 변형된 꼭짓점 (로봇 베이스 기준)."""
        P = np.asarray(P, dtype=float)
        plate = np.asarray(plate)
        if self.butt:
            return self._deform_butt(P, plate, s_arc)
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

    def _deform_butt(self, P, plate, s_arc) -> np.ndarray:
        """맞대기: 판마다 틈새 절반만큼 벌리고(오른쪽은 단차만큼 올림) 자기 루트 끝을 축으로 각변형, 그다음 좌우·높이·밀림."""
        s = np.clip(P[:, 0] / self.L, 0.0, 1.0)
        out = P.copy()
        gap = self.gap(s, s_arc)
        aL, aR = self.plate_angles(s, s_arc)
        for pid, side, ang in ((BASE, -1.0, -aL), (WEB, 1.0, aR)):
            m = plate == pid
            ey, ez = side * gap[m] / 2, (self.hi_lo() if side > 0 else 0.0)
            yr, zr, a = out[m, 1], out[m, 2], np.broadcast_to(ang, s.shape)[m]
            out[m, 1] = ey + yr * np.cos(a) - zr * np.sin(a)
            out[m, 2] = ez + yr * np.sin(a) + zr * np.cos(a)
        out[:, 1] += self.web_lateral(s)
        out[:, 2] += self.base_height(s, s_arc)
        out += self.slip(s_arc)
        Rw, tw = self.T_place[:3, :3], self.T_place[:3, 3]
        return out @ Rw.T + tw

    def nominal_meshes(self, nx: int = 60):
        """바닥 판·웹 판의 명목 메시. 반환: (꼭짓점 (n,3), 판 번호 (n,), 면 꼭짓점 수, 면 인덱스)."""
        g = self.scn['geometry']
        if self.butt:   # 맞대기 V홈: 왼쪽·오른쪽 판, 루트 끝 (0, 0)에 꼭짓점 (틈새는 deform_points에서 벌림)
            W, f, h = g['plate_width'], g['root_face'], g['thickness'] - g['root_face']
            a = h * math.tan(g['bevel'])
            sections = [(BASE, [(-W, -f), (0, -f), (0, 0), (-a, h), (-W, h)]),
                        (WEB, [(0, -f), (W, -f), (W, h), (a, h), (0, 0)])]
        else:
            W, tb, H, tw = g['base_width'], g['base_thickness'], g['web_height'], g['web_thickness']
            # 단면 (y, z) 다각형. 바닥 판은 y=0에 꼭짓점을 둬서 각변형 꺾임이 생기게 함
            if g.get('joint', 'fillet_t') == 'fillet_corner':   # ㄱ자: 바닥 판은 토치 쪽 날개 + 웹 밑까지
                base = [(-W, -tb), (0, -tb), (tw, -tb), (tw, 0), (0, 0), (-W, 0)]
            else:
                base = [(-W / 2, -tb), (0, -tb), (W / 2, -tb), (W / 2, 0), (0, 0), (-W / 2, 0)]
            sections = [(BASE, base), (WEB, [(0, 0), (tw, 0), (tw, H), (0, H)])]
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
    for k in ('web_square', 'hi_lo'):
        if k in z['shape']:
            z['shape'][k] = 0.0
    z['slip'].update(drift=[0.0, 0.0, 0.0], jump=[0.0, 0.0, 0.0])
    z['heat'].update(base_angle=0.0, web_tilt=0.0, bow=0.0, gap_close=0.0)
    z['prior'] = {'base_angle': 0.0, 'web_tilt': 0.0, 'bow': 0.0}
    z['sensors'].update(laser_noise=0.0, laser_dropout=0.0, laser_point_noise=0.0, laser_point_dropout=0.0)
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
    'web_square': ('웹 직각도 (코너 필릿)', {'shape': {'web_square': 3 * DEG}}),
    'hi_lo': ('단차 (맞대기)', {'shape': {'hi_lo': 2 * MM}}),
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


# ----------------------------------------------------------------------------- 학습용 묶음
# 에피소드 생성 사양 "배치와 오차": 깨끗함 / 단독 / 혼합 / 가혹(평가 전용) 네 강도
KINDS = ('clean', 'single', 'mixed', 'hard')

# 단독 묶음에서 오차마다 샘플 시나리오에서 남길 항목 (나머지는 0)
_SINGLE_KEEP = {
    'place': [('place',)], 'icp': [('icp',)],
    'web_straightness': [('shape', 'web_bow'), ('shape', 'web_wave'), ('shape', 'web_wave_cycles'),
                         ('shape', 'web_wave_phase')],
    'base_flatness': [('shape', 'base_bow'), ('shape', 'base_wave'), ('shape', 'base_wave_cycles'),
                      ('shape', 'base_wave_phase')],
    'gap': [('shape', 'gap0'), ('shape', 'gap1')],
    'tack': [('shape', 'tacks')],
    'drift': [('slip', 'drift'), ('slip', 'drift_s0'), ('slip', 'drift_span')],
    'jump': [('slip', 'jump'), ('slip', 'jump_s')],
    'base_angle': [('heat', 'base_angle'), ('heat', 'length')],
    'web_tilt': [('heat', 'web_tilt'), ('heat', 'length')],
    'bow': [('heat', 'bow')],
    'gap_close': [('shape', 'gap0'), ('shape', 'gap1'), ('heat', 'gap_close')],
    'web_square': [('shape', 'web_square')],
    'hi_lo': [('shape', 'hi_lo')],
}

# 설정에 범위가 있어야 쓰는 단독 오차 (그 시편군에만 있는 오차)
_SINGLE_NEEDS = {'web_square': ('pre_weld', 'web_square_deg'), 'hi_lo': ('pre_weld', 'hi_lo_mm')}


def single_names(cfg: dict) -> list[str]:
    """이 config(시편군)에서 쓸 단독 오차 이름 (SINGLE_ERRORS 순서). T 필릿은 13종, 코너 필릿은 웹 직각도를 더해 14종,
    맞대기 V홈은 단차를 더해 14종."""
    return [n for n in SINGLE_ERRORS
            if n not in _SINGLE_NEEDS or cfg.get(_SINGLE_NEEDS[n][0], {}).get(_SINGLE_NEEDS[n][1], 0)]

# 가혹 묶음에서 크기를 키울 범위 (개수·확률·주기·구간 길이는 그대로)
_SCALE_KEYS = {
    'pre_weld': ('place_xyz_mm', 'place_rpy_deg', 'icp_xyz_mm', 'icp_rot_deg', 'web_bow_mm_per_m', 'web_wave_mm',
                 'base_bow_mm_per_m', 'base_wave_mm', 'gap_mm', 'gap_taper_mm', 'tack_height_mm', 'web_square_deg',
                 'hi_lo_mm'),
    'in_weld': ('drift_mm', 'jump_mm', 'base_angle_deg', 'web_tilt_deg', 'bow_mm', 'gap_close_mm'),
}


def scale_cfg(cfg: dict, k: float) -> dict:
    """오차 크기 범위를 k배 한 config 사본 (가혹 묶음: 범위 상한의 1.5배)."""
    c = copy.deepcopy(cfg)
    for sec, keys in _SCALE_KEYS.items():
        for key in keys:
            if key not in c[sec]:
                continue
            v = c[sec][key]
            c[sec][key] = [x * k for x in v] if isinstance(v, (list, tuple)) else v * k
    return c


def single_in_range(cfg: dict, name: str, rng: np.random.Generator, seed=None) -> dict:
    """오차 하나만 범위 안에서 뽑은 시나리오 (단독 묶음). 센서 노이즈·실패는 설정값 그대로."""
    if name not in single_names(cfg) and name in SINGLE_ERRORS and name != 'residual':
        raise ValueError(f'이 시편군 config에는 {name} 범위가 없음 ({".".join(_SINGLE_NEEDS[name])})')
    c = copy.deepcopy(cfg)
    c['in_weld']['drift_prob'] = c['in_weld']['jump_prob'] = 1.0   # 밀림·튐 단독은 반드시 생기게
    src = sample_scenario(c, rng, seed=seed)
    z = zero_errors(src)
    z['sensors'] = copy.deepcopy(src['sensors'])
    if name == 'residual':
        z['prior'] = {'base_angle': src['heat']['base_angle'], 'web_tilt': src['heat']['web_tilt'],
                      'bow': src['heat']['bow']}
        z['pass_index'] = 1
    elif name in _SINGLE_KEEP:
        for path in _SINGLE_KEEP[name]:
            if len(path) == 1:
                z[path[0]] = copy.deepcopy(src[path[0]])
            else:
                z[path[0]][path[1]] = copy.deepcopy(src[path[0]][path[1]])
    else:
        raise ValueError(f'알 수 없는 오차: {name}')
    z['single_error'] = name
    return z


def episode_meta(kind: str, mode: str, rng: np.random.Generator, ep: dict, perturb: bool, split: str) -> dict:
    """시나리오에 붙일 에피소드 설정: 강도, 직선/위빙 모드, 시작 어긋남, 실행 섭동, 나눠 쓰기."""
    so = ep['start_offset']
    return {'kind': kind, 'mode': mode, 'split': split, 'weld_only': True,
            # 시작 토치 어긋남 (이음매 좌표계: 좌우 y, 높이 z [m], 작업각 x, 진행각 y [rad])
            'start_offset': [_uni(rng, so['lateral_mm']) * MM, _uni(rng, so['height_mm']) * MM,
                             _uni(rng, so['work_deg']) * DEG, _uni(rng, so['travel_deg']) * DEG],
            'start_settle': float(ep.get('start_settle_s', 0.3)),
            'perturb': bool(perturb), 'perturb_seed': int(rng.integers(2 ** 31)),
            'perturb_cfg': {'lateral': ep['perturb']['lateral_mm'] * MM, 'height': ep['perturb']['height_mm'] * MM,
                            'angle': ep['perturb']['angle_deg'] * DEG,
                            'duration': list(ep['perturb']['duration_s']),
                            'interval': list(ep['perturb']['interval_s'])}}
