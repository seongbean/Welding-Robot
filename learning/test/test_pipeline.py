"""생성 → 변환 → 학습 → 평가 흐름 (짧은 이음매의 운동학 에피소드로, Isaac 없이)."""
import copy
import os

import numpy as np
import pytest
import yaml

from learning.features import (ACTION_NAMES, BUF_OFFSETS_MM, OBS_NAMES, SectionBuffer, episode_frames,
                               load_episode, obs_index, torch_offset)
from learning.to_lerobot import convert
from sim_errors.episode import run_kinematic
from sim_errors.expert import apply_offset
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch
from sim_errors.part_errors import DEG, MM, PartErrorModel


def test_section_buffer_distance_indexing():
    buf = SectionBuffer(0.3)                      # 300 mm 이음매
    for x_mm in np.arange(0, 60, 0.5):            # 0~60 mm에서 잰 값: 좌우 = 위치(mm) × 0.001 m
        buf.add(x_mm / 300, [x_mm * MM * 0.01, 0.0, 0.001, 0.01, -0.01])
    vals, valid = buf.query(40 / 300)             # 토치 40 mm: 질의점 30, 35, ..., 70 mm
    exp = (40 + BUF_OFFSETS_MM) * 0.01
    ok = 40 + BUF_OFFSETS_MM < 62              # 마지막 측정 59.5 mm에서 2.5 mm 안까지
    assert np.array_equal(valid.astype(bool), ok)
    assert np.allclose(vals[ok, 0], exp[ok], atol=0.006)
    assert np.allclose(vals[ok, 2], 1.0)
    assert np.allclose(vals[ok, 3], np.degrees(0.01)) and np.allclose(vals[ok, 4], -np.degrees(0.01))
    buf.add(45 / 300, [np.nan] * 3)               # 측정 실패는 무시
    assert buf.query(40 / 300)[1].sum() == ok.sum()


def test_torch_offset_inverts_apply_offset():
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scn = make_batch(cfg, {'clean': 1, 'single': 0, 'mixed': 0, 'hard': 0})[0][1]
    m = PartErrorModel(scn)
    from sim_errors.expert import ScriptedExpert
    ex = ScriptedExpert(m, np.eye(4))
    T = ex.seam_pose(0.4, 0.4, ex.ex['standoff_mm'] * MM)
    o = [0.8 * MM, -0.5 * MM, 1.5 * DEG, -1.0 * DEG]
    got = torch_offset(m, 0.4, apply_offset(T, m.label(0.4, 0.4)['R_true'], o))
    assert np.allclose(got, [0.8, -0.5, 1.5, -1.0], atol=1e-3)


@pytest.fixture(scope='module')
def small_batch(tmp_path_factory):
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    root = tmp_path_factory.mktemp('loop')
    ep_dir = root / 'episodes'
    ep_dir.mkdir()
    batch = make_batch(cfg, {'clean': 2, 'single': 2, 'mixed': 10, 'hard': 2}, seed=7)
    for k, (name, scn) in enumerate(batch):
        scn = copy.deepcopy(scn)
        scn['geometry']['length'] = 0.06                 # 12초 용접
        run_kinematic(scn, sensor_seed=k).save(str(ep_dir / name.replace('.json', '.npz')), scn)
    return root, ep_dir


def test_frames_shapes(small_batch):
    _, ep_dir = small_batch
    rows, scn, meta = load_episode(str(ep_dir / 'mixed_0001.npz'))
    obs, act, ts = episode_frames(rows, scn)
    assert obs.shape[1] == len(OBS_NAMES) == 61 and act.shape[1] == len(ACTION_NAMES) == 7
    assert len(obs) == pytest.approx(rows['t'][-1] * 10, abs=2)
    assert np.allclose(np.diff(ts), 0.1, atol=0.011)
    assert meta['mode'] == scn['episode']['mode']
    # 행동 정답 = 기록된 전문가 정답 (mm, deg)
    assert np.allclose(act[:, 4], rows['speed'][0] / MM, rtol=0.5)
    # 단면 버퍼 유효 표시는 0/1, 토치 앞 30 mm 점은 대부분 유효
    valid = obs[:, obs_index(['section'])][:, 27:]
    prof = obs[:, obs_index(['profile'])]
    assert prof.shape[1] == 18 and np.abs(prof).max() > 0      # 평면 맞춤 각도가 들어 있음
    assert set(np.unique(valid)) <= {0.0, 1.0} and valid[:, -1].mean() > 0.4   # 60 mm 이음매라 끝 30 mm는 앞이 없음


def test_convert_train_evaluate(small_batch):
    torch = pytest.importorskip('torch')  # noqa: F841
    from learning.evaluate import evaluate
    from learning.train_bc import Policy, train
    root, ep_dir = small_batch
    data = str(root / 'data')
    r = convert(str(ep_dir), data, lerobot=False)
    assert r['episodes'] == 16 and r['splits']['train'] and r['splits']['test']
    for model in ('mlp', 'gru'):
        out = str(root / model)
        res = train(data, out, model=model, epochs=3, seq=8, verbose=False)
        assert np.isfinite(res['best_val'])
        pol = Policy(os.path.join(out, 'policy.pt'))
        a = pol(np.zeros(61, dtype=np.float32))
        assert a.shape == (7,) and np.all(np.isfinite(a))
        ev = evaluate(os.path.join(out, 'policy.pt'), data, str(ep_dir), verbose=False)
        assert ev['closed_loop']['all']['done'] == 1.0
        assert set(ev['open_loop']) == set(ACTION_NAMES)
    # 입력 묶음을 줄여도 학습·실행
    out = str(root / 'notorch')
    train(data, out, model='mlp', groups=('section', 'progress', 'mode'), epochs=2, verbose=False)
    assert len(Policy(os.path.join(out, 'policy.pt')).idx) == 39
    assert Policy(os.path.join(out, 'policy.pt'))(np.zeros(61, dtype=np.float32)).shape == (7,)


def test_failed_episodes_are_skipped(small_batch, tmp_path):
    import json
    import shutil
    _, ep_dir = small_batch
    d = tmp_path / 'eps'
    shutil.copytree(ep_dir, d)
    with open(d / 'batch_report.json', 'w', encoding='utf-8') as f:
        json.dump({'episodes': [{'file': 'mixed_0000.npz', 'ok': False, 'reason': '시험'}]}, f)
    r = convert(str(d), str(tmp_path / 'data'), lerobot=False)
    assert r['episodes'] == 15 and r['skipped'] == [('mixed_0000.npz', '시험')]


def test_train_subsets_and_parallel_eval(small_batch, tmp_path):
    pytest.importorskip('torch')
    import json
    from learning.evaluate import evaluate
    from learning.train_bc import load_frames, pick_train, train
    root, ep_dir = small_batch
    data = str(tmp_path / 'data')
    convert(str(ep_dir), data, lerobot=False)
    fr = load_frames(data)
    tr = [i for i, s in enumerate(fr['ep_split']) if s == 'train']
    assert fr['ep_perturb'].any() and not fr['ep_perturb'].all()
    # 같은 시드면 작은 N이 큰 N의 부분집합, 강도가 고루 들어감
    small, big = pick_train(fr, tr, 3, seed=1), pick_train(fr, tr, 6, seed=1)
    assert len(small) == 3 and len(big) == 6 and set(small) <= set(big)
    assert len({str(fr['ep_kind'][e]) for e in big}) >= 2
    calm = pick_train(fr, tr, no_perturb=True)
    assert calm and not any(fr['ep_perturb'][e] for e in calm)
    out = str(tmp_path / 'sub')
    res = train(data, out, epochs=1, max_train=4, verbose=False)
    log = json.load(open(os.path.join(out, 'train_log.json'), encoding='utf-8'))
    assert res['train_episodes'] == 4 and len(log['train_files']) == 4
    one = evaluate(os.path.join(out, 'policy.pt'), data, str(ep_dir), verbose=False)
    two = evaluate(os.path.join(out, 'policy.pt'), data, str(ep_dir), verbose=False, jobs=2)
    assert one['closed_loop']['all'] == two['closed_loop']['all']


def test_prescan_fills_start_zone(small_batch):
    """용접 전 시작 구간 스캔: 단면이 토치 앞 30 mm만 재므로 스캔이 없으면 시작 구간 버퍼가 빈다."""
    _, ep_dir = small_batch
    rows, scn, _ = load_episode(str(ep_dir / 'mixed_0001.npz'))
    assert scn['sensors']['prescan'] == pytest.approx(0.03)
    s_pts, meas = PartErrorModel(scn).prescan(None)
    assert s_pts[0] == 0 and s_pts[-1] * scn['geometry']['length'] == pytest.approx(0.03)
    assert np.isfinite(meas).all()
    k0 = 27 + int(np.flatnonzero(BUF_OFFSETS_MM == 0)[0])        # 토치 위치 점의 유효 표시 열
    with_scan = episode_frames(rows, scn)[0]
    without = episode_frames(rows, scn, prescan=False)[0]
    assert with_scan[:5, k0].all() and not without[:5, k0].any()
    assert np.array_equal(with_scan, episode_frames(rows, scn)[0])   # 스캔 노이즈는 시나리오마다 고정
    off = copy.deepcopy(scn)
    off['sensors']['prescan'] = 0.0
    assert len(PartErrorModel(off).prescan(None)[0]) == 0


def test_rule_baseline_and_start_zone(small_batch, tmp_path):
    """규칙 기준선(torch 없이): 시작 구간 스캔이 있으면 시작 30 mm 높이 오차가 줄어든다."""
    from learning.evaluate import evaluate
    _, ep_dir = small_batch
    data = str(tmp_path / 'data')
    convert(str(ep_dir), data, lerobot=False)
    scan = evaluate('rule', data, str(ep_dir), splits=('train', 'val', 'test'), verbose=False,
                    rule_dir=str(tmp_path))
    blind = evaluate('rule-noprescan', data, str(ep_dir), splits=('train', 'val', 'test'), verbose=False,
                     rule_dir=str(tmp_path))
    a, b = scan['closed_loop']['all'], blind['closed_loop']['all']
    assert a['done'] == 1.0 and scan['prescan'] and not blind['prescan']
    assert a['height_start'] < 0.5 * b['height_start']
    assert a['height_gain'] > b['height_gain']
    assert os.path.exists(tmp_path / 'rule' / 'eval.json')
    assert set(scan['open_loop']) == set(ACTION_NAMES)


@pytest.fixture(scope='module')
def corner_batch(tmp_path_factory):
    from sim_errors.part_errors import family_cfg
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = family_cfg(yaml.safe_load(f), 'fillet_corner')
    ep_dir = tmp_path_factory.mktemp('corner') / 'episodes'
    ep_dir.mkdir()
    for k, (name, scn) in enumerate(make_batch(cfg, {'clean': 1, 'single': 2, 'mixed': 10, 'hard': 1}, seed=7)):
        scn = copy.deepcopy(scn)
        scn['geometry']['length'] = 0.06
        run_kinematic(scn, sensor_seed=k).save(str(ep_dir / name.replace('.json', '.npz')), scn)
    return ep_dir


def test_two_families_convert_and_evaluate(small_batch, corner_batch, tmp_path):
    """T + 코너 에피소드 폴더를 한 데이터로 합치고, 시편군별로 나눠 평가 (규칙 기준선, torch 없이)."""
    import json
    from learning.evaluate import evaluate
    from learning.report_evals import table
    _, t_dir = small_batch
    data = str(tmp_path / 'data')
    r = convert([str(t_dir), str(corner_batch)], data, lerobot=False)
    assert r['families'] == ['fillet_corner', 'fillet_t'] and r['episodes'] == 16 + 14
    fr = np.load(os.path.join(data, 'frames.npz'))
    assert set(fr['ep_family']) == {'fillet_t', 'fillet_corner'}
    assert all(str(f).startswith('corner_') == (str(g) == 'fillet_corner')
               for f, g in zip(fr['ep_file'], fr['ep_family']))
    both = evaluate('rule', data, [str(t_dir), str(corner_batch)], splits=('train', 'val', 'test'), verbose=False,
                    rule_dir=str(tmp_path))
    assert {'fam_fillet_t', 'fam_fillet_corner'} <= set(both['closed_loop'])
    n_corner = sum(str(g) == 'fillet_corner' for g in fr['ep_family'])
    assert both['closed_loop']['fam_fillet_corner']['n'] == n_corner
    one = evaluate('rule', data, [str(t_dir), str(corner_batch)], splits=('train', 'val', 'test'), verbose=False,
                   rule_dir=str(tmp_path), families=['fillet_corner'], name='corner')
    assert one['eval_families'] == ['fillet_corner'] and 'fam_fillet_t' not in one['closed_loop']
    assert one['closed_loop']['all'] == both['closed_loop']['fam_fillet_corner']
    assert os.path.exists(tmp_path / 'rule' / 'eval_corner.json')
    md = table([str(tmp_path / 'rule' / 'eval.json'), str(tmp_path / 'rule' / 'eval_corner.json')])
    assert 'rule/corner' in md and '→ 코너' in md
    with open(tmp_path / 'rule' / 'eval_corner.json', encoding='utf-8') as f:
        assert json.load(f)['closed_loop']['all']['done'] == 1.0
    with pytest.raises(SystemExit):
        convert([str(t_dir), str(t_dir)], str(tmp_path / 'dup'), lerobot=False)   # 이름이 겹치면 거부


def test_train_on_one_family(small_batch, corner_batch, tmp_path):
    pytest.importorskip('torch')
    from learning.evaluate import evaluate
    from learning.train_bc import train
    _, t_dir = small_batch
    data = str(tmp_path / 'data')
    convert([str(t_dir), str(corner_batch)], data, lerobot=False)
    out = str(tmp_path / 't_only')
    res = train(data, out, epochs=1, verbose=False, families=['fillet_t'])
    assert res['cfg']['families'] == ['fillet_t']
    assert all(not f.startswith('corner_') for f in res['train_files'])
    ev = evaluate(os.path.join(out, 'policy.pt'), data, [str(t_dir), str(corner_batch)], verbose=False,
                  families=['fillet_corner'], name='corner')
    assert ev['train_families'] == ['fillet_t'] and ev['eval_families'] == ['fillet_corner']
