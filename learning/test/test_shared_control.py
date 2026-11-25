"""공유 제어 규칙(클러치·유지 오프셋·초기화·한도), 리더 매핑, 잔차 학습 프레임, 기본 정책 해시 확인."""
import copy
import json

import numpy as np
import pytest
import yaml

from learning.base_policy import identify
from learning.evaluate import RulePolicy
from learning.shared_control import HumanCorrection, SharedDriver, intervention_stats, leader_delta, residual_frames
from sim_errors.expert import ScriptedExpert
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch
from sim_errors.part_errors import DEG, MM, PartErrorModel, rotvec_to_R

DT = 0.01


def pose(dx=0.0, dy=0.0, dz=0.0, rx=0.0, ry=0.0):
    T = np.eye(4)
    T[:3, :3] = rotvec_to_R([rx * DEG, ry * DEG, 0.0])
    T[:3, 3] = [0.1 + dx * MM, 0.2 + dy * MM, 0.3 + dz * MM]
    return T


def run(h, T, clutch, n, reset_at=None):
    out = None
    for k in range(n):
        out = h.step(DT, T, clutch, reset=(k == reset_at))
    return out


def test_leader_delta_axes():
    d = leader_delta(pose(dx=3, dy=1, dz=2, rx=4, ry=-2), pose(), np.eye(3))
    assert np.allclose(d[:3], [3, 1, 2], atol=1e-9)
    assert np.allclose(d[3:], [4, -2, 0], atol=1e-6)
    # 리더 베이스를 z축으로 90° 돌려 놓으면 리더 x 이동 = 이음매 좌우
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    assert np.allclose(leader_delta(pose(dx=2), pose(), R)[:3], [0, 2, 0], atol=1e-9)


def test_clutch_adds_then_holds_after_release():
    h = HumanCorrection(pos_scale=0.25, ang_scale=0.5)
    assert np.allclose(run(h, pose(dy=100), False, 50), 0)            # 클러치 OFF면 리더가 움직여도 0
    run(h, pose(), True, 1)                                           # 클러치 ON: 이 자세가 기준
    c = run(h, pose(dy=8, dz=-4, dx=50, rx=6), True, 100)
    assert np.allclose(c, [2, -1, 3, 0], atol=1e-6)                   # 축척, 진행 방향 x는 버림
    c = run(h, pose(dy=40), False, 100)                               # 놓은 뒤: 리더를 움직여도 유지
    assert np.allclose(c, [2, -1, 3, 0], atol=1e-6) and np.allclose(h.held, c)
    run(h, pose(dy=40), True, 1)                                      # 다시 잡으면 유지 오프셋 위에 더함
    c = run(h, pose(dy=44), True, 100)
    assert np.allclose(c, [3, -1, 3, 0], atol=1e-6)


def test_reset_ramps_to_zero_without_jump():
    h = HumanCorrection(reset_time=1.0)
    run(h, pose(), True, 1)
    run(h, pose(dz=16), True, 100)
    run(h, pose(dz=16), False, 5)
    assert h.held[1] == pytest.approx(4.0)
    vals = []
    for k in range(150):
        vals.append(h.step(DT, None, False, reset=(k == 0))[1])
    vals = np.array(vals)
    assert vals[49] == pytest.approx(2.0, abs=0.05)                   # 1초에 걸쳐 일정하게 줄어듦
    assert np.all(np.diff(vals) <= 1e-12) and np.abs(np.diff(vals)).max() < 0.05
    assert vals[-1] == 0 and not h.resetting


def test_limit_rate_gain_and_leader_loss():
    h = HumanCorrection(gain=(1, 0, 1, 1), limit=(5, 5, 10, 10), max_rate=(20, 20, 30, 30))
    run(h, pose(), True, 1)
    c1 = h.step(DT, pose(dy=400, dz=40), True)
    assert c1[0] == pytest.approx(0.2) and c1[1] == 0                 # 주기당 20 mm/s × 0.01 s, 높이 권한 0
    c = run(h, pose(dy=400, dz=40), True, 100)
    assert c[0] == pytest.approx(5.0)                                 # 한도
    c = run(h, None, True, 10)                                        # 리더 끊김: 유지 오프셋으로
    assert c[0] == pytest.approx(5.0) and h.anchor is None
    h.step(DT, pose(dy=-1000), True)                                  # 다시 읽히면 그 자세에서 기준
    assert h.step(DT, pose(dy=-1000), True)[0] == pytest.approx(5.0)


def make_scn(length=0.04, seed=3):
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    _, scn = make_batch(cfg, {'clean': 0, 'single': 0, 'mixed': 1, 'hard': 0}, seed=seed)[0]
    scn = copy.deepcopy(scn)
    scn['geometry']['length'] = length
    return scn


def test_shared_driver_records_labels_and_residual_frames():
    """사람이 1초부터 2초까지 높이를 올리고 놓은 뒤 3초에 초기화: 정답 = 적용 사람 보정, 목표 = 정책 + 사람."""
    scn = make_scn()
    model = PartErrorModel(scn)
    expert = ScriptedExpert(model, np.eye(4), np.random.default_rng(0))
    h = HumanCorrection(pos_scale=0.25)
    clock = {'t': 0.0}

    def human_input():
        t = clock['t']
        clutch = 1.0 <= t < 2.0
        T = pose(dz=12 * min(max(t - 1.0, 0.0), 0.5) / 0.5)            # 0.5초 동안 리더 12 mm 올림 → 3 mm
        return T, clutch, abs(t - 3.0) < DT / 2, 0.1 if clutch else 0.5

    drv = SharedDriver(RulePolicy('rule'), model, expert, h, human_input, sensor_seed=1)
    rows = {k: [] for k in ('t', 'policy_tick', 'clutch', 'human_c', 'base_c', 'obs', 'reset_event', 'policy_c',
                            'T_target')}
    for _ in range(600):
        r = drv.step(DT, np.eye(4))
        clock['t'] += DT
        ex = r['extra']
        for k in rows:
            rows[k].append(r[k] if k in ('t', 'T_target') else ex[k])
        assert np.allclose(ex['policy_c'], ex['base_c'] + ex['residual_c'] + ex['human_c'])
        if drv.done:
            break
    t = np.array(rows['t'])
    hc = np.array(rows['human_c'])
    assert hc[t < 1.0].max() == 0
    assert hc[(t > 1.7) & (t < 2.95), 1] == pytest.approx(3.0, abs=1e-6)  # 놓은 뒤에도 유지 (정답에 포함)
    assert np.abs(hc[t > 4.1]).max() == 0                                # 초기화 뒤 0
    st = intervention_stats(rows)
    assert st['n_clutch'] == 1 and st['n_reset'] == 1 and st['human_height_max'] == pytest.approx(3.0)
    fr = residual_frames(rows)
    assert len(fr['obs']) == int(np.sum(rows['policy_tick'])) and np.isfinite(fr['obs']).all()
    assert not fr["keep"][fr["t"] < 0.99].any()   # 클러치 직전 2초 제외
    assert fr['keep'][fr['t'] >= 1.0].all()
    assert np.allclose(fr['label'], hc[np.array(rows['policy_tick'])])


def test_identify_policy_hash(tmp_path):
    p = tmp_path / 'policy.pt'
    p.write_bytes(b'abc')
    reg = {'base-v1': {'sha256': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'}}
    assert identify(str(p), registry=reg)[0] == 'base-v1'
    assert identify(str(p), 'base-v1', registry=reg)[0] == 'base-v1'
    p.write_bytes(b'abd')
    assert identify(str(p), registry=reg)[0] is None
    with pytest.raises(ValueError):
        identify(str(p), 'base-v1', registry=reg)
    assert identify('rule') == (None, '')
    with open('learning/base_policies.json', encoding='utf-8') as f:
        assert len(json.load(f)['base-v1']['sha256']) == 64
