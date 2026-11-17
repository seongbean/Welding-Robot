"""에피소드 기록(.npz) 공통 부분과 Isaac 없이 도는 운동학 에피소드 (numpy만 사용).

- Recorder: 주기마다 한 행을 모아 .npz로 저장. 필드 설명은 sim_errors/README.md.
- run_kinematic: 완벽 추종(실측 = 보낸 목표)을 가정하고 규칙 전문가 에피소드를 만든다. 관절 필드는 없다.
  Isaac Sim 없이 묶음 생성 → 변환 → 학습 → 평가 흐름 전체를 시험할 때 쓴다 (sim_bridge/run_batch.py --kinematic).
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from .expert import ScriptedExpert
from .part_errors import PartErrorModel

# 기록 필드 중 정책 학습·점검에 쓰는 것 (관절 필드 q_cmd, q_meas는 Isaac 실행에만 있음)
EXPERT_FIELDS = ('t', 'phase', 'arc_on', 'done', 's', 's_arc', 'e', 'rotvec', 'gap', 'tack', 'weave_amp', 'weave_hz',
                 'speed', 'laser', 'p_true', 'p_est', 'T_target', 'T_expert', 'perturb')


def git_commit() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=here, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return ''


class Recorder:
    def __init__(self):
        self.rows: dict[str, list] = {}

    def add(self, **kw):
        for k, v in kw.items():
            self.rows.setdefault(k, []).append(v)

    def add_expert(self, r: dict, **kw):
        self.add(**{k: r[k] for k in EXPERT_FIELDS}, laser_valid=bool(np.isfinite(r['laser'][0])), **kw)

    def __len__(self):
        return len(self.rows.get('t', []))

    def save(self, path: str, scn: dict, **meta):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        out = {}
        for k, v in self.rows.items():
            out[k] = np.array(v, dtype='U16') if k in ('phase', 'status') else np.array(v)
        epi = scn.get('episode', {})
        info = {'family': scn['geometry'].get('joint', 'fillet_t'), 'mode': epi.get('mode', 'weave'),
                'kind': epi.get('kind', ''), 'split': epi.get('split', ''), 'git': git_commit(), **meta}
        np.savez_compressed(path, scenario_json=json.dumps(scn), meta_json=json.dumps(info, ensure_ascii=False), **out)


def default_home_pose() -> np.ndarray:
    """운동학 에피소드용 툴 롤 기준 (툴 x축 = 베이스 +X). Isaac 실행은 홈 관절 자세의 FK를 쓴다."""
    return np.eye(4)


def run_kinematic(scn: dict, sensor_seed: int = 0, dt: float = 0.01, T_home=None, max_steps: int = 200000) -> Recorder:
    """완벽 추종 운동학 에피소드. scn['T_nominal']을 그대로 쓴다."""
    model = PartErrorModel(scn)
    ex = ScriptedExpert(model, default_home_pose() if T_home is None else T_home, np.random.default_rng(sensor_seed))
    rec = Recorder()
    for _ in range(max_steps):
        r = ex.step(dt)
        rec.add_expert(r, T_meas=r['T_target'], status='ok')
        if ex.done:
            break
    return rec
