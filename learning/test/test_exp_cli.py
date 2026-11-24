"""--exp로 변환 → 학습 → 평가를 경로 없이 돌리면 outputs/experiments/<ID>/ 배치에 쓰고 manifest에 단계가 쌓이는지."""
import copy
import json
import os

import pytest
import yaml

from common import paths
from sim_errors.episode import run_kinematic
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch

EXP = '20251201-cli'


def test_exp_convert_train_evaluate(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from learning import evaluate, to_lerobot, train_bc
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / 'out'))
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    ep_dir = paths.exp_dir(EXP, 'episodes', create=True)
    for k, (name, scn) in enumerate(make_batch(cfg, {'clean': 1, 'single': 1, 'mixed': 6, 'hard': 1}, seed=3)):
        scn = copy.deepcopy(scn)
        scn['geometry']['length'] = 0.05
        run_kinematic(scn, sensor_seed=k).save(str(ep_dir / name.replace('.json', '.npz')), scn)

    to_lerobot.main(['--exp', EXP, '--no-lerobot'])
    assert (paths.exp_dir(EXP, 'data') / 'frames.npz').is_file()
    train_bc.main(['--exp', EXP, '--run', 'mlp', '--epochs', '1'])
    assert (paths.exp_dir(EXP, 'runs') / 'mlp' / 'policy.pt').is_file()
    evaluate.main(['mlp', 'rule', '--exp', EXP])
    for name in ('mlp', 'rule'):
        assert (paths.exp_dir(EXP, 'runs') / name / 'eval.json').is_file()

    with open(paths.exp_dir(EXP) / 'manifest.json', encoding='utf-8') as f:
        stages = json.load(f)['stages']
    assert [s['stage'] for s in stages] == ['to_lerobot', 'train_bc', 'evaluate']
    assert stages[1]['extra']['out'] == str(paths.exp_dir(EXP, 'runs') / 'mlp')
    assert stages[2]['extra']['policies'][0] == str(paths.exp_dir(EXP, 'runs') / 'mlp' / 'policy.pt')


def test_train_needs_out_or_run(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from learning import train_bc
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / 'out'))
    with pytest.raises(SystemExit):
        train_bc.main(['--exp', EXP])
    assert not os.path.exists(paths.exp_dir(EXP))   # 실패한 명령은 manifest를 남기지 않음
