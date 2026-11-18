"""resense: 기록의 단면 열만 다시 계산 (정답·목표는 그대로)."""
import copy
import json

import numpy as np
import yaml

from sim_errors.episode import run_kinematic
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch
from sim_errors.resense import main as resense_main


def test_resense_rewrites_only_laser(tmp_path):
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    name, scn = make_batch(cfg, {'clean': 0, 'single': 0, 'mixed': 1, 'hard': 0}, seed=4)[0]
    scn = copy.deepcopy(scn)
    scn['geometry']['length'] = 0.05
    src = tmp_path / 'a'
    src.mkdir()
    rec = run_kinematic(scn)
    rec.rows['laser'] = [r[:3] for r in rec.rows['laser']]          # 예전 세 열 기록 흉내
    rec.save(str(src / 'mixed_0000.npz'), scn)
    resense_main([str(src), '--out', str(tmp_path / 'b')])
    old = np.load(src / 'mixed_0000.npz')
    new = np.load(tmp_path / 'b' / 'mixed_0000.npz')
    assert old['laser'].shape[1] == 3 and new['laser'].shape[1] == 5
    for k in ('e', 'rotvec', 'T_target', 'T_meas', 's', 's_arc'):
        assert np.array_equal(old[k], new[k])
    assert np.isfinite(new['laser'][:, 3]).mean() > 0.3      # 50 mm 이음매라 끝 30 mm는 측정 범위 밖
    assert np.array_equal(new['laser_valid'], np.isfinite(new['laser'][:, 0]))
    assert 'laser_point_noise' in json.loads(str(new['scenario_json']))['sensors']
    # 같은 명령은 같은 결과
    resense_main([str(src), '--out', str(tmp_path / 'c')])
    assert np.array_equal(np.load(tmp_path / 'c' / 'mixed_0000.npz')['laser'], new['laser'], equal_nan=True)
