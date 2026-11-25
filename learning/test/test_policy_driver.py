"""정책 구동기(Isaac 정책 실행기의 토치 목표·오차 계산)를 Isaac 없이: 완벽 추종이면 운동학 닫힌 루프와 같아야 한다."""
import copy

import numpy as np
import pytest
import yaml

from learning.evaluate import AXES, RulePolicy, closed_loop
from learning.features import torch_offset
from learning.policy_driver import PolicyDriver, as_cmd, episode_metrics
from sim_errors.episode import Recorder
from sim_errors.expert import ScriptedExpert
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch
from sim_errors.part_errors import PartErrorModel, family_cfg


def scenarios(families=('fillet_t', 'pipe_girth'), length=0.06):
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    out = []
    for fam in families:
        c = family_cfg(cfg, fam)
        scn = copy.deepcopy(make_batch(c, {'clean': 0, 'single': 0, 'mixed': 1, 'hard': 0}, seed=3)[0][1])
        if fam != 'pipe_girth':
            scn['geometry']['length'] = length
        out.append(scn)
    return out


def drive(policy, scn, seed=5, dt=0.01, lag=1, max_steps=40000):
    """완벽 추종(lag=0: 실측 = 이번 목표) 또는 lag 주기 늦은 추종으로 정책 구동기를 돌린 기록."""
    model = PartErrorModel(scn)
    expert = ScriptedExpert(model, np.eye(4), weld_only=True)
    d = PolicyDriver(policy, model, expert, sensor_seed=seed)
    rec = Recorder()
    hist = [expert.start_pose()]
    for _ in range(max_steps):
        T_meas = hist[max(0, len(hist) - 1 - lag)] if lag else None
        r = d.step(dt, T_meas if T_meas is not None else hist[-1])
        hist.append(r['T_target'])
        rec.add_expert(r, T_meas=r['T_target'] if lag == 0 else T_meas, status='ok', **r['extra'])
        if d.done:
            break
    rows = {k: np.array(v) for k, v in rec.rows.items()}
    rows['status'] = rows['status'].astype('U16')
    return rows


@pytest.mark.parametrize('scn', scenarios(), ids=['fillet_t', 'pipe_girth'])
def test_perfect_tracking_matches_kinematic_closed_loop(scn):
    rows = drive(RulePolicy('rule'), scn, lag=0)
    m = episode_metrics(rows, scn)
    k = closed_loop(RulePolicy('rule'), scn, sensor_seed=5)
    assert m['done'] and k['done']
    for n in AXES:
        assert m['cmd_' + n] == pytest.approx(k[n], rel=1e-6, abs=1e-9), n          # 명령 = 운동학 정의
        assert m[n] == pytest.approx(m['cmd_' + n], rel=1e-4, abs=1e-6), n          # 완벽 추종이면 측정 = 명령
        assert m[n + '_base'] == pytest.approx(k[n + '_base'], rel=1e-6)
    assert m['track_p95_mm'] < 1e-9 and abs(m['lag_mm']) < 1e-6
    assert as_cmd(m)['lateral'] == m['cmd_lateral']


def test_target_matches_offset_and_tracking_lag_shows_up():
    scn = scenarios(('fillet_corner',))[0]
    scn['episode']['mode'] = 'weave'
    rows = drive(RulePolicy('rule'), scn, lag=5)
    model = PartErrorModel(scn)
    for i in range(0, len(rows['s']), 50):          # 목표 = 추정 경로 + 보정 + 위빙
        off = torch_offset(model, float(rows['s'][i]), rows['T_target'][i])
        off[0] -= rows['policy_weave'][i]
        assert np.allclose(off, rows['policy_c'][i], atol=1e-6)
    m = episode_metrics(rows, scn)
    assert m['track_p95_mm'] > 0 and m['lag_mm'] > 0      # 늦게 따라가면 이음매 방향으로 뒤진다
    assert m['cmd_height'] <= m['height'] + 0.05
    assert m['lateral_center'] < m['lateral']             # 늦게 따라간 위빙 흔들림을 걷어 낸 중심선 오차


def test_driver_rejects_full_episode_scenario():
    scn = scenarios(('fillet_t',))[0]
    scn['episode']['weld_only'] = False
    model = PartErrorModel(scn)
    with pytest.raises(ValueError):
        PolicyDriver(RulePolicy('rule'), model, ScriptedExpert(model, np.eye(4)))
