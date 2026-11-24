"""--exp 옵션: 실험 폴더 배치로 경로를 채우고, 주지 않으면 지금과 같게 동작하는지."""
import argparse
import json
import os
import sys

import pytest

from common import paths

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'sim_bridge'))
import run_batch  # noqa: E402

EXP = '20251201-test'


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / 'out'))
    return tmp_path / 'out'


def _manifest(exp_id):
    with open(paths.exp_dir(exp_id) / 'manifest.json', encoding='utf-8') as f:
        return json.load(f)


def test_make_scenarios_exp_writes_batch_and_manifest(root):
    from sim_errors import make_scenarios
    make_scenarios.main(['--batch', '--counts', '1,1,1,1', '--exp', EXP])
    files = sorted(os.listdir(paths.exp_dir(EXP, 'batch')))
    assert files == ['clean_0000.json', 'hard_0000.json', 'mixed_0000.json', 'single_0000.json']
    st = _manifest(EXP)['stages'][0]
    assert st['stage'] == 'make_scenarios' and st['extra']['batch'] is True
    assert len(st['extra']['config_sha256']) == 64


def test_make_scenarios_default_out_unchanged(tmp_path, monkeypatch, root):
    from sim_errors import make_scenarios
    monkeypatch.chdir(tmp_path)
    make_scenarios.main(['--n', '1'])
    assert os.listdir(tmp_path / 'scenarios') == ['part0000_pass0.json']
    assert not (root / 'experiments').exists()   # --exp 없으면 manifest도 안 씀


def test_bad_exp_id_rejected():
    from sim_errors import make_scenarios
    with pytest.raises(SystemExit):
        make_scenarios.main(['--batch', '--exp', 'weave'])


def _rb(**kw):
    d = dict(scenarios=None, out=None, exp=None, reach_only=False, kinematic=False, group='all')
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.mark.parametrize('kw, part', [
    ({'kinematic': True}, 'episodes'),
    ({'reach_only': True}, 'reach'),
])
def test_run_batch_exp_defaults(root, kw, part):
    a = _rb(exp=EXP, **kw)
    run_batch.resolve_args(argparse.ArgumentParser(), a)
    assert a.scenarios == str(paths.exp_dir(EXP, 'batch'))
    assert a.out == str(paths.exp_dir(EXP, part))


def test_run_batch_isaac_group(root):
    a = _rb(exp=EXP, group='butt')
    run_batch.resolve_args(argparse.ArgumentParser(), a)
    assert a.out == str(paths.exp_dir(EXP, 'episodes_isaac') / 'butt')


def test_run_batch_explicit_out_wins(root, tmp_path):
    a = _rb(exp=EXP, kinematic=True, scenarios='s', out=str(tmp_path / 'x'))
    run_batch.resolve_args(argparse.ArgumentParser(), a)
    assert (a.scenarios, a.out) == ('s', str(tmp_path / 'x'))


def test_run_batch_needs_paths_without_exp():
    with pytest.raises(SystemExit):
        run_batch.resolve_args(argparse.ArgumentParser(), _rb(scenarios='s'))
