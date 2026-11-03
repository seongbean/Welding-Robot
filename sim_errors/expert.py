"""규칙 전문가 (privileged): 실제 경로를 알고 토치 목표 포즈를 만들며, 매 주기 정답을 함께 낸다.

numpy만 사용. IK와 Isaac Sim 연결은 sim_bridge/expert_to_sim.py가 맡는다.

단계: approach(시작점 위로) → descend(시작점으로) → weld(s 0→1, 아크 ON) → retract(위로) → done
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


class ScriptedExpert:
    def __init__(self, model: PartErrorModel, T_start: np.ndarray, rng: np.random.Generator | None = None):
        self.m = model
        self.ex = model.scn['expert']
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.T_start = np.asarray(T_start, dtype=float)
        self.roll_ref = self.T_start[:3, 0].copy()   # 툴 x축 기준 방향 (월드)
        self.phase = 'approach'
        self.t = 0.0
        self.t_phase = 0.0
        self.s = 0.0           # 토치 진행도
        self.s_arc = 0.0       # 아크가 지나간 진행도 (열변형·밀림의 기준)
        self.weave_phase = 0.0
        self.T_from = self.T_start
        self.T_last = self.T_start

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
            self.weave_phase += 2 * math.pi * ex['weave_hz'] * dt
            lift = standoff + ex['tack_follow'] * float(self.m.tack_height([self.s])[0])
            T = self.seam_pose(self.s, self.s_arc, lift, amp * math.sin(self.weave_phase))
            if self.s >= 1.0:
                self._next('retract', T)
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
        return {'t': self.t, 'phase': phase, 'arc_on': phase == 'weld', 'done': self.done,
                's': self.s, 's_arc': self.s_arc, 'T_target': T,
                'e': lab['e'], 'rotvec': lab['rotvec'], 'gap': lab['gap'], 'tack': lab['tack'],
                'p_true': lab['p_true'], 'p_est': lab['p_est'],
                'weave_amp': amp, 'weave_hz': ex['weave_hz'] if phase == 'weld' else 0.0, 'speed': v,
                'laser': (self.m.laser(self.s, self.s_arc, self.rng) if phase == 'weld'
                          else np.array([np.nan, np.nan]))}

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
