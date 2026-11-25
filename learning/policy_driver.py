"""학습 정책이 토치를 모는 구동기와 그 기록의 닫힌 루프 오차 (numpy만 사용, 정책 객체는 밖에서 받음).

Isaac 정책 실행기(sim_bridge/policy_to_sim.py)가 규칙 전문가 자리에 넣는다. 한 주기 흐름은 운동학 닫힌 루프
(learning.evaluate.closed_loop)와 같다: 단면 측정을 버퍼에 쌓고, 10 Hz마다 관측을 만들어 정책 행동(보정 4 + 속도·
위빙)을 받고, 추정 경로 + 보정 + 위빙으로 토치 목표를 만든 뒤 정책 속도로 진행도를 올린다. 다른 점은 관측의
토치 자세가 명령이 아니라 실측(Isaac 관절 실측의 FK)이라는 것 하나다.

기록(Recorder)에는 전문가 기록 필드에 더해 policy_c(보정 명령 [mm, mm, deg, deg]), policy_weave(위빙 좌우 [mm]),
policy_action(행동 7개)을 남긴다. e, rotvec, gap, tack은 그 진행도의 정답이라, 이 기록은 나중에 개입·DAgger 데이터로도 쓸 수 있다.

오차(episode_metrics)는 두 가지로 낸다.
- 측정: 실측 토치(FK) − 실제 이음매. 좌우는 명령한 위빙을 뺀다. Isaac 관절 추종 오차가 들어간 실제 닫힌 루프 오차.
- 명령: 정책 보정 명령 − 실제 이음매. 운동학 평가와 같은 정의라 둘을 바로 비교할 수 있다.
"""
from __future__ import annotations

import math

import numpy as np

from sim_errors.expert import ScriptedExpert
from sim_errors.part_errors import DEG, MM, PartErrorModel, rotvec_to_R

from .evaluate import AXES, truth
from .features import FPS, SectionBuffer, fill_prescan, make_obs, torch_offset


class PolicyDriver:
    """정책으로 토치 목표를 낸다. ScriptedExpert.step과 같은 필드의 dict를 돌려줘 expert_to_sim.run_episode에 그대로 들어간다.

    expert: 같은 모재 모델의 규칙 전문가 (용접 구간만). 토치 롤 기준(홈 자세)과 시작 자세, 속도·위빙 규칙(비교용)만 빌린다."""

    def __init__(self, policy, model: PartErrorModel, expert: ScriptedExpert, sensor_seed: int = 0,
                 fps: int = FPS, warn=print):
        if not expert.weld_only:
            raise ValueError('정책 실행은 용접 구간만 담은 시나리오(episode.weld_only, make_scenarios --batch)만 지원합니다')
        self.policy, self.m, self.ex = policy, model, expert
        self.cfg = model.scn['expert']
        self.mode = expert.mode
        self.rng = np.random.default_rng(sensor_seed)
        self.look = model.scn['sensors']['laser_lookahead'] / model.L
        self.v_nom = self.cfg['travel_speed_mm_s']
        self.buf = SectionBuffer(model.L)
        if getattr(policy, 'cfg', {}).get('prescan', False) and not fill_prescan(self.buf, model, self.rng):
            warn('경고: 정책은 시작 구간 스캔으로 학습했는데 시나리오에 스캔 설정(sensors.prescan)이 없음')
        if hasattr(policy, 'bind'):
            policy.bind(model.scn)
        policy.reset()
        so = expert.start_offset
        self.c = truth(model, 0.0)[0] + np.array([so[0] / MM, so[1] / MM, so[2] / DEG, so[3] / DEG])   # 놓인 자세
        self.speed, self.amp, self.hz = self.v_nom, 0.0, 0.0
        self.action = np.full(7, np.nan)
        self.s = self.t = self.phase = 0.0
        self.period = 1.0 / fps
        self.next_t = 0.0
        self.t_max = 3 * model.L / (self.v_nom * MM)
        self.done = False

    def target(self, s: float, c, weave_mm: float) -> np.ndarray:
        """추정 경로 + 보정 c + 위빙 → 토치 포즈. 위치·방향은 운동학 평가의 torch_pose와 같고, 롤은 전문가처럼
        홈 자세 툴 x축에 가깝게 고정한다 (손목이 돌지 않게)."""
        p_e, R_e = self.m.seam_est(s)
        p_e, R_e = p_e[0], R_e[0]
        T = np.eye(4)
        T[:3, :3] = self.ex.tool_R(R_e @ rotvec_to_R([c[2] * DEG, c[3] * DEG, 0.0]))
        T[:3, 3] = p_e + R_e @ np.array([0.0, (c[0] + weave_mm) * MM, (self.cfg['standoff_mm'] + c[1]) * MM])
        return T

    def step(self, dt: float, T_meas: np.ndarray) -> dict:
        s = self.s
        laser = self.m.laser(s, s, self.rng)
        self.buf.add(s + self.look, laser)
        weave = self.amp * math.sin(self.phase)
        if self.t + 1e-9 >= self.next_t:
            vals, valid = self.buf.query(s)
            a = np.asarray(self.policy(make_obs(vals, valid, torch_offset(self.m, s, T_meas), s, self.mode)), dtype=float)
            self.action = a.copy()
            self.c = a[:4].copy()
            self.speed = float(np.clip(a[4], 0.3 * self.v_nom, 2.0 * self.v_nom))
            self.amp = float(np.clip(a[5], 0.0, 2 * self.cfg['weave_amp_max_mm']))
            self.hz = float(np.clip(a[6], 0.0, 3.0))
            self.next_t += self.period
        T = self.target(s, self.c, weave)
        lab = self.m.label(s, s)
        r = {'t': self.t, 'phase': 'weld', 'arc_on': True, 'done': False, 's': s, 's_arc': s,
             'T_target': T, 'T_expert': T, 'perturb': np.zeros(4),
             'e': lab['e'], 'rotvec': lab['rotvec'], 'gap': lab['gap'], 'tack': lab['tack'],
             'p_true': lab['p_true'], 'p_est': lab['p_est'],
             'weave_amp': self.amp * MM, 'weave_hz': self.hz, 'speed': self.speed * MM, 'laser': laser,
             'extra': {'policy_c': self.c.copy(), 'policy_weave': weave, 'policy_action': self.action.copy()}}
        self.s = min(1.0, s + self.speed * MM * dt / self.m.L)
        self.phase += 2 * math.pi * self.hz * dt
        self.t += dt
        r['done'] = self.s >= 1.0                       # 이음매 끝까지 감 (기록의 done)
        self.done = r['done'] or self.t >= self.t_max   # 시간 초과도 멈춤 (운동학 평가와 같은 한도)
        return r


def episode_metrics(rows: dict, scn: dict, start_zone_mm: float = 30.0) -> dict:
    """정책 실행 기록 → 닫힌 루프 오차. 키는 learning.evaluate.closed_loop과 같고(측정 기준), 명령 기준은 cmd_ 접두사.
    track_p95_mm: 실측 팁과 보낸 목표 사이 거리 95%, lag_mm: 실측 팁이 이음매 방향으로 명령보다 뒤진 평균."""
    model = PartErrorModel(scn)
    rule = ScriptedExpert(model, np.eye(4), mode=scn.get('episode', {}).get('mode', 'weave'), weld_only=True)
    weld = np.flatnonzero(np.asarray(rows['arc_on']).astype(bool))
    meas, cmd, base, proc, lag, s_log = [], [], [], [], [], []
    for i in weld:
        s = float(rows['s'][i])
        tr, gap = truth(model, s)
        T = np.asarray(rows['T_meas'][i])
        off = torch_offset(model, s, T)
        off[0] -= float(rows['policy_weave'][i])
        meas.append(off - tr)
        cmd.append(np.asarray(rows['policy_c'][i]) - tr)
        base.append(-tr)
        p_e, R_e = model.seam_est(s)
        lag.append(-(R_e[0].T @ (T[:3, 3] - p_e[0]))[0] / MM)
        amp_r, v_r = rule.process(gap)
        proc.append([rows['speed'][i] / MM - v_r / MM, rows['weave_amp'][i] / MM - amp_r / MM,
                     rows['weave_hz'][i] - rule.weave_hz])
        s_log.append(s)
    meas, cmd, base, proc = np.array(meas), np.array(cmd), np.array(base), np.array(proc)
    pos = np.linalg.norm(np.asarray(rows['T_meas'])[weld, :3, 3] - np.asarray(rows['T_target'])[weld, :3, 3], axis=1) / MM
    out = {'done': bool(len(weld) and rows['done'][weld[-1]]),
           'time': float(rows['t'][weld[-1]] - rows['t'][weld[0]]) if len(weld) else 0.0,
           'track_p95_mm': float(np.percentile(pos, 95)), 'lag_mm': float(np.mean(lag))}
    for j, n in enumerate(AXES):
        out[n] = float(np.sqrt(np.mean(meas[:, j] ** 2)))
        out[n + '_base'] = float(np.sqrt(np.mean(base[:, j] ** 2)))
        out[n + '_p95'] = float(np.percentile(np.abs(meas[:, j]), 95))
        out['cmd_' + n] = float(np.sqrt(np.mean(cmd[:, j] ** 2)))
        out['cmd_' + n + '_p95'] = float(np.percentile(np.abs(cmd[:, j]), 95))
    for j, n in enumerate(('speed', 'weave_amp', 'weave_hz')):
        out[n + '_err'] = float(np.sqrt(np.mean(proc[:, j] ** 2)))
    start = np.array(s_log) * model.L < start_zone_mm * MM
    out['height_sq_start'] = float(np.sum(meas[start, 1] ** 2))
    out['height_sq_rest'] = float(np.sum(meas[~start, 1] ** 2))
    out['n_start'], out['n_rest'] = int(start.sum()), int((~start).sum())
    return out


def as_cmd(row: dict) -> dict:
    """명령 기준 오차를 측정 키 자리에 옮긴 행 (learning.evaluate.summarize로 같은 요약을 내려고)."""
    r = dict(row)
    for n in AXES:
        r[n], r[n + '_p95'] = row['cmd_' + n], row['cmd_' + n + '_p95']
    r.pop('n_start', None)
    return r
