"""check_episode: 규칙 전문가 기록 점검 (완벽 추종을 가정한 합성 기록으로)."""
import json

import numpy as np
import pytest
import yaml

from sim_errors.check_episode import FAIL, OK, check, load, plot
from sim_errors.make_scenarios import DEFAULT_CFG
from sim_errors.part_errors import sample_scenario

from .test_part_errors import run_expert


@pytest.fixture(scope='module')
def episode():
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scn = sample_scenario(cfg, np.random.default_rng(4))
    m, ex, out = run_expert(scn, dt=0.05)
    rows = {k: np.array([r[k] for r in out]) for k in out[0] if k not in ('p_true', 'p_est')}
    rows['T_meas'] = rows['T_target'].copy()          # 완벽 추종
    rows['status'] = np.array(['ok'] * len(out))
    return rows, m.scn


def test_perfect_episode_passes(episode):
    rows, scn = episode
    res = check(rows, scn)
    assert all(lv == OK for _, lv, _ in res), res


def test_wrong_target_height_fails(episode):
    rows, scn = episode
    bad = {k: v.copy() for k, v in rows.items()}
    i = np.flatnonzero(bad['arc_on'])[10]
    bad['T_target'][i][:3, 3] += bad['T_target'][i][:3, 2] * 0.001     # 토치 방향으로 1 mm
    res = dict((n, lv) for n, lv, _ in check(bad, scn))
    assert res['목표 기하'] == FAIL


def test_interrupted_episode_fails(episode):
    rows, scn = episode
    n = int(np.flatnonzero(rows['arc_on'])[50])
    cut = {k: v[:n] for k, v in rows.items()}
    res = dict((name, lv) for name, lv, _ in check(cut, scn))
    assert res['단계 순서'] == FAIL and res['진행도'] == FAIL


def test_npz_roundtrip_and_plot(episode, tmp_path):
    rows, scn = episode
    p = tmp_path / 'ep.npz'
    np.savez_compressed(p, scenario_json=json.dumps(scn), **rows)
    rows2, scn2 = load(str(p))
    assert all(lv == OK for _, lv, _ in check(rows2, scn2))
    pytest.importorskip('matplotlib')
    png = plot(rows2, scn2, str(p))
    assert png and (tmp_path / 'ep.png').exists()
