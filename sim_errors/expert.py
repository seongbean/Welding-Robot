"""규칙 전문가 (privileged): 실제 경로를 알고 토치 목표 포즈를 만들며, 매 주기 정답을 함께 낸다.

numpy만 사용. IK와 Isaac Sim 연결은 sim_bridge/expert_to_sim.py가 맡는다.

단계: approach(시작점 위로) → descend(시작점으로) → weld(s 0→1, 아크 ON) → retract(위로) → done
- 용접 구간만 (시나리오 episode.weld_only): weld → done. 로봇은 start_pose()(시작점 + 어긋남)에 바로 놓는다.
  시작 어긋남은 실행 명령에서 start_settle 시간상수로 줄어들고, 정답(e, rotvec)에는 없다.
- 직선/위빙 모드 (episode.mode, 에피소드 중 고정): 직선은 위빙 진폭 0, 틈새에 따른 속도 규칙만 쓴다.
- 실행 섭동 (episode.perturb, DART 방식): 좌우·높이·작업각·진행각에 반 코사인 펄스를 실행 명령에만 더한다.
  기록의 T_expert = 섭동 없는 전문가 목표, T_target = 실제로 보낸 목표, perturb = 더한 양.
- 용접 중 목표: 실제 이음매 점 + 이등분 방향 standoff(+가접 높이) + 좌우(이음매 y) 위빙
- 토치 방향: 이음매 쪽을 향함(-z_seam), 진행 방향으로 travel_angle만큼 기울임(미는 각)
- 토치 롤: 시작 자세의 툴 x축에 가장 가깝게 고정 (손목이 불필요하게 돌지 않게)
- 틈새 → 위빙 진폭·진행 속도 규칙 (config expert 섹션, WPS 값으로 바꿀 것)
"""
from __future__ import annotations

import math

import numpy as np

from .part_errors import MM, DEG, PartErrorModel, R_to_rotvec, rotvec_to_R


def _interp_pose(T0, T1, a: float) -> np.ndarray:
    a = min(1.0, max(0.0, a))
    T = np.eye(4)
    T[:3, 3] = (1 - a) * T0[:3, 3] + a * T1[:3, 3]
    T[:3, :3] = T0[:3, :3] @ rotvec_to_R(a * R_to_rotvec(T0[:3, :3].T @ T1[:3, :3]))
    return T


def make_pulses(cfg: dict, rng: np.random.Generator, t_max: float) -> list[tuple[float, float, np.ndarray]]:
    """실행 섭동 펄스 [(시작 시각, 지속, [좌우, 높이, 작업각, 진행각])]."""
    out, t = [], float(rng.uniform(*cfg['interval']))
    sd = np.array([cfg['lateral'], cfg['height'], cfg['angle'], cfg['angle']])
    while t < t_max:
        dur = float(rng.uniform(*cfg['duration']))
        out.append((t, dur, rng.normal(size=4) * sd))
        t += dur + float(rng.uniform(*cfg['interval']))
    return out


def apply_offset(T: np.ndarray, R_seam: np.ndarray, o) -> np.ndarray:
    """토치 포즈에 이음매 좌표계 어긋남 o = [좌우 y, 높이 z, 작업각(x축), 진행각(y축)]을 더한다."""
    T = T.copy()
    T[:3, 3] += R_seam[:, 1] * o[0] + R_seam[:, 2] * o[1]
    T[:3, :3] = R_seam @ rotvec_to_R([o[2], o[3], 0.0]) @ R_seam.T @ T[:3, :3]
    return T


class ScriptedExpert:
    def __init__(self, model: PartErrorModel, T_start: np.ndarray, rng: np.random.Generator | None = None,
                 weld_only: bool | None = None, mode: str | None = None):
        """T_start: 시작 자세의 토치 포즈 (툴 롤 기준 방향). 용접 구간만 모드에서는 홈 자세 FK를 주고,
        로봇은 start_pose()에 놓는다."""
        self.m = model
        self.ex = model.scn['expert']
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.T_start = np.asarray(T_start, dtype=float)
        self.roll_ref = self.T_start[:3, 0].copy()   # 툴 x축 기준 방향 (월드)
        epi = model.scn.get('episode', {})
        self.weld_only = bool(epi.get('weld_only', False)) if weld_only is None else weld_only
        self.mode = mode or epi.get('mode', 'weave')
        if self.mode not in ('weave', 'straight'):
            raise ValueError(f'알 수 없는 모드: {self.mode}')
        self.start_offset = np.array(epi.get('start_offset', [0.0] * 4), dtype=float) if self.weld_only else np.zeros(4)
        self.settle = float(epi.get('start_settle', 0.3))
        self.pulses = []
        if epi.get('perturb'):
            t_max = 2 * self.m.L / (self.ex['travel_speed_mm_s'] * MM * self.ex['min_speed_ratio'])
            self.pulses = make_pulses(epi['perturb_cfg'], np.random.default_rng(epi.get('perturb_seed', 0)), t_max)
        self.phase = 'approach'
        self.t = 0.0
        self.t_phase = 0.0
        self.s = 0.0           # 토치 진행도
        self.s_arc = 0.0       # 아크가 지나간 진행도 (열변형·밀림의 기준)
        self.weave_phase = 0.0
        self.T_from = self.T_start
        self.T_last = self.T_start
        if self.weld_only:
            self.phase = 'weld'
            self.T_from = self.T_last = self.seam_pose(0.0, 0.0, self.ex['standoff_mm'] * MM)

    def start_pose(self) -> np.ndarray:
        """용접 구간만 모드에서 로봇을 놓을 토치 포즈: 이음매 시작점 + 시작 어긋남."""
        lab = self.m.label(0.0, 0.0)
        return apply_offset(self.seam_pose(0.0, 0.0, self.ex['standoff_mm'] * MM), lab['R_true'], self.start_offset)

    def offset(self, t: float) -> np.ndarray:
        """t 시각의 실행 어긋남 = 시작 어긋남(지수 감소) + 섭동 펄스."""
        o = self.start_offset * math.exp(-t / max(1e-6, self.settle))
        for t0, dur, a in self.pulses:
            if t0 <= t < t0 + dur:
                o = o + a * 0.5 * (1 - math.cos(2 * math.pi * (t - t0) / dur))
        return o

    # --- 포즈 계산
    def tool_R(self, R_seam) -> np.ndarray:
        x_s, z_s = R_seam[:, 0], R_seam[:, 2]
        a = self.ex['travel_angle_deg'] * DEG
        z = -z_s * math.cos(a) + x_s * math.sin(a)
        z /= np.linalg.norm(z)
        ref = self.roll_ref - np.dot(self.roll_ref, z) * z
        if np.linalg.norm(ref) < 1e-3:              # 기준 방향이 토치 방향과 겹치면 진행 방향 사용
            ref = x_s - np.dot(x_s, z) * z
        x = ref / np.linalg.norm(ref)
        return np.stack([x, np.cross(z, x), z], axis=-1)

    def seam_pose(self, s: float, s_arc: float, lift: float, lateral: float = 0.0) -> np.ndarray:
        lab = self.m.label(s, s_arc)
        R = lab['R_true']
        T = np.eye(4)
        T[:3, :3] = self.tool_R(R)
        T[:3, 3] = lab['p_true'] + R[:, 2] * lift + R[:, 1] * lateral
        return T

    def process(self, gap: float) -> tuple[float, float]:
        """틈새 [m] → (위빙 진폭 [m], 진행 속도 [m/s])."""
        ex = self.ex
        g_mm = gap / MM
        amp = min(ex['weave_amp_max_mm'], ex['weave_amp_mm'] + ex['weave_amp_per_gap'] * g_mm) * MM
        if self.mode == 'straight':
            amp = 0.0
        v = ex['travel_speed_mm_s'] * MM * max(ex['min_speed_ratio'], 1 - ex['speed_drop_per_gap_mm'] * g_mm)
        return amp, v

    def _goto(self, T_to, speed_mm_s: float) -> tuple[np.ndarray, bool]:
        dist = float(np.linalg.norm(T_to[:3, 3] - self.T_from[:3, 3]))
        ang = float(np.linalg.norm(R_to_rotvec(self.T_from[:3, :3].T @ T_to[:3, :3])))
        dur = max(1.0, dist / (speed_mm_s * MM), ang / (30 * DEG))   # 회전은 30°/s 이하
        a = self.t_phase / dur
        return _interp_pose(self.T_from, T_to, a), a >= 1.0

    def _next(self, phase: str, T) -> None:
        self.phase, self.t_phase, self.T_from = phase, 0.0, T

    # --- 1주기
    def step(self, dt: float) -> dict:
        ex = self.ex
        phase = self.phase      # 이 주기의 목표를 만든 단계 (주기 끝에 다음 단계로 넘어가도 기록은 이 단계)
        self.t += dt
        self.t_phase += dt
        standoff = ex['standoff_mm'] * MM
        amp = v = 0.0
        if self.phase == 'approach':
            T, arrived = self._goto(self.seam_pose(0.0, 0.0, ex['approach_height_mm'] * MM), ex['approach_speed_mm_s'])
            if arrived:
                self._next('descend', T)
        elif self.phase == 'descend':
            T, arrived = self._goto(self.seam_pose(0.0, 0.0, standoff), ex['descend_speed_mm_s'])
            if arrived:
                self._next('weld', T)
        elif self.phase == 'weld':
            amp, v = self.process(float(self.m.gap(np.array([self.s]), self.s_arc)[0]))
            self.s = min(1.0, self.s + v * dt / self.m.L)
            self.s_arc = self.s
            self.weave_phase += 2 * math.pi * self.weave_hz * dt
            lift = standoff + ex['tack_follow'] * float(self.m.tack_height([self.s])[0])
            T = self.seam_pose(self.s, self.s_arc, lift, amp * math.sin(self.weave_phase))
            if self.s >= 1.0:
                self._next('done' if self.weld_only else 'retract', T)
        elif self.phase == 'retract':
            T_up = self.T_from.copy()
            T_up[:3, 3] += self.T_from[:3, 2] * -ex['retract_height_mm'] * MM   # 토치 반대 방향으로 뺌
            T, arrived = self._goto(T_up, ex['approach_speed_mm_s'])
            if arrived:
                self._next('done', T)
        else:
            T = self.T_last
        self.T_last = T

        lab = self.m.label(self.s, self.s_arc)
        o = self.offset(self.t) if phase == 'weld' else np.zeros(4)
        T_cmd = apply_offset(T, lab['R_true'], o) if np.any(o) else T
        return {'t': self.t, 'phase': phase, 'arc_on': phase == 'weld', 'done': self.done,
                's': self.s, 's_arc': self.s_arc, 'T_target': T_cmd, 'T_expert': T, 'perturb': o,
                'e': lab['e'], 'rotvec': lab['rotvec'], 'gap': lab['gap'], 'tack': lab['tack'],
                'p_true': lab['p_true'], 'p_est': lab['p_est'],
                'weave_amp': amp, 'weave_hz': self.weave_hz if phase == 'weld' else 0.0, 'speed': v,
                'laser': (self.m.laser(self.s, self.s_arc, self.rng) if phase == 'weld'
                          else np.full(5, np.nan))}

    @property
    def weave_hz(self) -> float:
        return 0.0 if self.mode == 'straight' else float(self.ex['weave_hz'])

    @property
    def done(self) -> bool:
        return self.phase == 'done'


def auto_nominal(T_tip: np.ndarray, length: float, drop: float = 0.02) -> np.ndarray:
    """시작 자세의 토치 팁 근처에 모재를 둔다: 이음매 중간 = 팁 아래 drop, 이음매 방향 = 베이스 -Y,
    웹 판은 로봇 반대쪽(+X)에 서서 토치가 로봇 쪽에서 접근한다. 모재 좌표계 x=-Y, y=+X, z=+Z."""
    R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mid = np.asarray(T_tip[:3, 3], dtype=float) - np.array([0.0, 0.0, drop])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mid - R[:, 0] * length / 2
    return T
