"""run_batch --reach-only: 시험용 6축 팔로 네 시편군 시나리오의 IK 도달 점검이 돌고 보고서를 남기는지."""
import json
import os
import subprocess
import sys

import pytest
import yaml

pytest.importorskip('pinocchio')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from sim_errors.make_scenarios import DEFAULT_CFG, make_batch  # noqa: E402
from sim_errors.part_errors import family_cfg, save_scenario  # noqa: E402

HOME = '-0.02,-1.557,2.039,-2.053,-1.571,-1.591'


def test_reach_only_all_families(tmp_path):
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    batch = tmp_path / 'batch'
    batch.mkdir()
    for fam, pre in (('fillet_t', ''), ('fillet_corner', 'corner_'), ('butt_v', 'butt_'), ('pipe_girth', 'pipe_')):
        c = family_cfg(cfg, fam) if fam != 'fillet_t' else cfg
        for name, scn in make_batch(c, {'clean': 1, 'single': 0, 'mixed': 1, 'hard': 0}, seed=3):
            save_scenario(scn, str(batch / (pre + name)))
    out = tmp_path / 'out'
    out.mkdir()
    r = subprocess.run([sys.executable, os.path.join(REPO, 'sim_bridge', 'run_batch.py'), str(batch), '--out', str(out),
                        '--reach-only', '--follower-urdf', os.path.join(HERE, 'test_arm.urdf'),
                        '--follower-config', os.path.join(HERE, 'follower_test_arm.yaml'), f'--home-q={HOME}'],
                       capture_output=True, text=True, encoding='utf-8', cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.load(open(out / 'reach_report.json', encoding='utf-8'))
    assert rep['n'] == rep['ok'] == 8
    assert {s['family'] for s in rep['scenarios']} == {'fillet_t', 'fillet_corner', 'butt_v', 'pipe_girth'}
    assert not list(out.glob('*.npz'))     # 에피소드는 만들지 않음


def test_isaac_shift_only_for_fillets():
    from sim_errors.expert import ISAAC_SHIFT, auto_nominal, isaac_nominal
    from sim_errors.part_errors import PartErrorModel
    import numpy as np
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    T_tip = np.eye(4)
    T_tip[:3, 3] = [0.45, 0.0, 0.2]
    for fam in ('fillet_t', 'fillet_corner', 'butt_v', 'pipe_girth'):
        c = family_cfg(cfg, fam) if fam != 'fillet_t' else cfg
        _, scn = make_batch(c, {'clean': 1, 'single': 0, 'mixed': 0, 'hard': 0}, seed=1)[0]
        m = PartErrorModel(scn)
        d = isaac_nominal(m, T_tip)[:3, 3] - auto_nominal(T_tip, m.L, mid_part=m.nominal(0.5)[0][0])[:3, 3]
        assert np.allclose(d, ISAAC_SHIFT.get(fam, (0, 0, 0)))
