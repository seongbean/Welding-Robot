"""shared_to_sim 통신 시험: 가짜 receiver와 시험용 팔, mock 리더, 미리 정한 키 입력으로 공유 제어 에피소드를 기록한다."""
import copy
import json
import os

import numpy as np
import pytest
import yaml

pytest.importorskip('pinocchio')
from test_batch_link import FakeReceiver  # noqa: E402

import shared_to_sim  # noqa: E402
from learning.features import load_episode  # noqa: E402
from learning.shared_control import residual_frames  # noqa: E402
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch  # noqa: E402
from sim_errors.part_errors import save_scenario  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = '-0.02,-1.557,2.039,-2.053,-1.571,-1.591'


class ScriptedKeys:
    """시작(s) 뒤 클러치(c)를 켜 두고, 몇 번 물어본 뒤 클러치를 놓고 초기화(r)."""

    def __init__(self):
        self.pending = ['s', 'c']
        self.calls = 0

    def take(self, ch):
        self.calls += 1
        if self.calls == 600:
            self.pending += ['c', 'r']
        if ch in self.pending:
            self.pending.remove(ch)
            return True
        return False

    def clear(self):
        pass

    def push(self, ch):
        self.pending.append(ch)


def test_shared_runner_records_intervention(tmp_path, monkeypatch):
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scn_dir = tmp_path / 'batch'
    scn_dir.mkdir()
    name, scn = make_batch(cfg, {'clean': 1, 'single': 0, 'mixed': 0, 'hard': 0}, seed=2)[0]
    scn = copy.deepcopy(scn)
    scn['geometry']['length'] = 0.05
    save_scenario(scn, str(scn_dir / name))
    monkeypatch.setattr(shared_to_sim, 'Keys', ScriptedKeys)
    rx = FakeReceiver(('127.0.0.1', 19890), ('127.0.0.1', 19891))
    rx.start()
    out = tmp_path / 'out'
    argv = ['--policy', 'rule', '--scenarios', str(scn_dir), '--split', 'all', '--out', str(out), '--mock',
            '--mock-amp', '0.1', '--follower-urdf', os.path.join(HERE, 'test_arm.urdf'),
            '--follower-config', os.path.join(HERE, 'follower_test_arm.yaml'), f'--home-q={HOME}',
            '--send', '127.0.0.1:19890', '--listen', '127.0.0.1:19891', '--max-time', '60']
    try:
        assert shared_to_sim.main(argv) == 0
    finally:
        rx.stop = True
        rx.join(2)
        rx.sock.close()
    files = sorted(out.glob('*.npz'))
    assert len(files) == 1 and files[0].name.startswith(os.path.splitext(name)[0] + '__')
    assert not files[0].name.endswith('.failed.npz')
    rec, _, meta = load_episode(str(files[0]))
    assert meta['runner'] == 'isaac_shared' and meta['policy'] == 'rule' and meta['clutch_source'] == 'key'
    assert meta['human']['pos_scale'] == 0.25
    clutch = rec['clutch'].astype(bool)
    assert clutch[0] and not clutch[-1] and rec['reset_event'].sum() == 1
    assert np.abs(rec['human_c']).max() > 0.01                   # mock 리더가 움직여 사람 보정이 생김
    assert np.allclose(rec['policy_c'], rec['base_c'] + rec['residual_c'] + rec['human_c'])
    fr = residual_frames(rec)
    assert len(fr['obs']) > 10 and np.isfinite(fr['obs']).all()
    with open(out / 'interventions.json', encoding='utf-8') as f:
        assert json.load(f)['episodes'][0]['n_reset'] == 1
    assert (out / 'eval_isaac.json').is_file()
