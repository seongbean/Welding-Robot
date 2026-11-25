import datetime as dt
import json

import pytest

from common import paths


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / 'out'))
    return tmp_path / 'out'


def test_default_root_is_repo_outputs(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    assert paths.outputs_root() == paths.REPO_ROOT / 'outputs'


def test_env_overrides_root(root):
    assert paths.outputs_root() == root.resolve()


def test_exp_dir_layout(root):
    assert paths.exp_dir('20261201-weave') == root.resolve() / 'experiments' / '20261201-weave'
    assert paths.exp_dir('20261201-weave', 'episodes_isaac').name == 'episodes_isaac'
    p = paths.exp_dir('20261201-weave', 'runs', create=True)
    assert p.is_dir()


@pytest.mark.parametrize('bad', ['weave', '2026-12-01-weave', '20261301-x', '20261201-Weave', '20261201-'])
def test_bad_exp_id(bad):
    with pytest.raises(ValueError):
        paths.check_exp_id(bad)


def test_unknown_part(root):
    with pytest.raises(KeyError):
        paths.exp_dir('20261201-weave', 'models')


def test_new_exp_id():
    assert paths.new_exp_id('kin-4fam', dt.date(2026, 11, 23)) == '20261123-kin-4fam'


def test_ci_and_scratch(root):
    assert paths.ci_dir(44492062655) == root.resolve() / 'ci' / '44492062655'
    with pytest.raises(ValueError):
        paths.ci_dir('../x')
    assert paths.scratch_dir('a', 'b') == root.resolve() / 'scratch' / 'a' / 'b'


def test_resolve_out_prefers_explicit_out(root, tmp_path):
    assert paths.resolve_out(tmp_path / 'x', '20261201-weave', 'batch') == tmp_path / 'x'
    assert paths.resolve_out(None, '20261201-weave', 'batch') == paths.exp_dir('20261201-weave', 'batch')
    with pytest.raises(ValueError):
        paths.resolve_out(None, None, 'batch')


def test_write_manifest_appends(tmp_path):
    paths.write_manifest(tmp_path, 'make_scenarios', {'seed': 0}, argv=['a'])
    p = paths.write_manifest(tmp_path, 'run_batch', argv=['b'])
    data = json.loads(p.read_text(encoding='utf-8'))
    assert [s['stage'] for s in data['stages']] == ['make_scenarios', 'run_batch']
    assert data['stages'][0]['extra'] == {'seed': 0}
    assert data['stages'][1]['argv'] == ['b']
    assert 'commit' in data['stages'][0]['git']
