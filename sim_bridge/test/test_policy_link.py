"""policy_to_sim 통신 시험: 가짜 receiver(UDP)와 시험용 6축 팔로 규칙 기준선 정책이 에피소드를 몰고 요약을 낸다."""
import copy
import json
import os

import numpy as np
import pytest
import yaml

pytest.importorskip('pinocchio')
from test_batch_link import FakeReceiver  # noqa: E402

import policy_to_sim  # noqa: E402
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch  # noqa: E402
from sim_errors.part_errors import save_scenario  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = '-0.02,-1.557,2.039,-2.053,-1.571,-1.591'


def test_policy_runner_with_fake_receiver(tmp_path):
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scn_dir = tmp_path / 'batch'
    scn_dir.mkdir()
    batch = make_batch(cfg, {'clean': 1, 'single': 0, 'mixed': 2, 'hard': 0}, seed=2)
    for name, scn in batch:
        scn = copy.deepcopy(scn)
        scn['geometry']['length'] = 0.05
        save_scenario(scn, str(scn_dir / name))
    (scn_dir / 'batch_report.json').write_text('{"episodes": []}')     # 시나리오가 아닌 파일은 건너뜀
    rx = FakeReceiver(('127.0.0.1', 19880), ('127.0.0.1', 19881))
    rx.start()
    out = tmp_path / 'out'
    argv = ['--policy', 'rule', '--scenarios', str(scn_dir), '--split', 'all', '--out', str(out),
            '--follower-urdf', os.path.join(HERE, 'test_arm.urdf'),
            '--follower-config', os.path.join(HERE, 'follower_test_arm.yaml'), f'--home-q={HOME}',
            '--send', '127.0.0.1:19880', '--listen', '127.0.0.1:19881', '--max-time', '60']
    try:
        assert policy_to_sim.main(argv) == 0
    finally:
        rx.stop = True
        rx.join(2)
        rx.sock.close()
    names = [os.path.splitext(n)[0] for n, _ in batch]
    assert sorted(p.stem for p in out.glob('*.npz')) == sorted(names)        # 모두 끝까지 감 (.failed 없음)
    assert [os.path.basename(s) for s in rx.scenarios] == [n + '.json' for n in sorted(names)]
    with open(out / 'eval_isaac.json', encoding='utf-8') as f:
        res = json.load(f)
    assert res['n'] == len(names) and not res['failed_runs']
    g = res['groups']['all']
    assert g['isaac']['done'] == 1.0
    assert g['track_p95_mm'] < 2.0
    for n in ('lateral', 'height'):     # 시험 팔도 정책을 따라가 보정 0보다 낫다
        assert g['isaac'][n] < g['isaac'][n + '_base']
        assert abs(g['isaac'][n] - g['isaac_cmd'][n]) < 0.5
    d = np.load(out / (names[0] + '.npz'))
    assert {'policy_c', 'policy_action', 'q_meas', 'T_meas'} <= set(d.files)
    # 다시 돌리면 건너뛰고, 요약만 다시 낼 수 있다
    assert policy_to_sim.main(argv[:4] + ['--out', str(out), '--report-only']) == 0


def test_select_reads_episode_npz_and_filters(tmp_path):
    """운동학 실험 에피소드 폴더(.npz + batch_report.json)에서 나눔·시편군 수로 고른다."""
    from sim_errors.episode import run_kinematic
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    batch = make_batch(cfg, {'clean': 2, 'single': 0, 'mixed': 3, 'hard': 1}, seed=4)
    for name, scn in batch:
        scn = copy.deepcopy(scn)
        scn['geometry']['length'] = 0.03
        run_kinematic(scn).save(str(tmp_path / name.replace('.json', '.npz')), scn)
    (tmp_path / 'batch_report.json').write_text('{"episodes": []}')
    splits = {os.path.splitext(n)[0]: s['episode']['split'] for n, s in batch}
    got = policy_to_sim.select(str(tmp_path), 'test', None, None, None)
    assert [n for n, _ in got] == sorted(n for n, sp in splits.items() if sp == 'test')
    assert len(policy_to_sim.select(str(tmp_path), 'all', None, 2, None)) == 2          # 시편군 하나, 앞에서 2개
    assert [n for n, _ in policy_to_sim.select(str(tmp_path), 'all', 'hard', None, None)] == ['hard_0000']
