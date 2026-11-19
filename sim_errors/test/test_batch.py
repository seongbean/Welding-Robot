"""학습용 묶음: 시나리오 묶음, 용접 구간만 에피소드, 직선 모드, 시작 어긋남, 실행 섭동 (Isaac 없이)."""
import copy

import numpy as np
import pytest
import yaml

from sim_errors.check_episode import FAIL, OK, check
from sim_errors.episode import run_kinematic
from sim_errors.expert import ScriptedExpert, apply_offset
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch
from sim_errors.part_errors import MM, DEG, PartErrorModel, R_to_rotvec, scale_cfg, single_in_range, single_names


@pytest.fixture(scope='module')
def cfg():
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        return yaml.safe_load(f)


def short(scn, length=0.06):
    """빠른 시험용: 이음매를 짧게."""
    s = copy.deepcopy(scn)
    s['geometry']['length'] = length
    return s


def rows_of(rec):
    return {k: np.array(v) for k, v in rec.rows.items()}


def test_batch_counts_modes_splits(cfg):
    counts = {'clean': 4, 'single': 13, 'mixed': 20, 'hard': 4}
    batch = make_batch(cfg, counts, seed=3)
    assert len(batch) == sum(counts.values())
    eps = [s['episode'] for _, s in batch]
    for kind, n in counts.items():
        g = [e for e in eps if e['kind'] == kind]
        assert len(g) == n
        assert abs(sum(e['mode'] == 'weave' for e in g) - n / 2) <= 0.5     # 직선:위빙 반반
    assert all(e['split'] == 'test' for e in eps if e['kind'] == 'hard')
    mixed = [e for e in eps if e['kind'] == 'mixed']
    assert sum(e['perturb'] for e in mixed) == 10 and not any(e['perturb'] for e in eps if e['kind'] != 'mixed')
    assert {e['split'] for e in mixed} == {'train', 'val', 'test'}
    for e in eps:
        so = np.abs(e['start_offset'])
        assert e['weld_only'] and np.all(so[:2] <= 2 * MM + 1e-12) and np.all(so[2:] <= 2 * DEG + 1e-12)
    # 같은 시드면 같은 묶음
    again = make_batch(cfg, counts, seed=3)
    assert [s for _, s in again] == [s for _, s in batch]


def test_single_in_range_keeps_one_error(cfg):
    rng = np.random.default_rng(0)
    for name in single_names(cfg):
        scn = single_in_range(cfg, name, rng)
        assert scn['single_error'] == name
        assert scn['sensors']['laser_noise'] > 0        # 센서 노이즈는 그대로
        m = PartErrorModel(scn)
        labs = [m.label(s, 1.0) for s in np.linspace(0, 1, 11)]   # 각변형은 아크 뒤에만 생기므로 s_arc = 1
        mag = max(np.abs(lab['e']).max() + np.abs(lab['rotvec']).max() + lab['gap'] + lab['tack'] for lab in labs)
        if name == 'place':     # 배치 오차는 ICP가 알아내므로 정답 0, 경로만 움직임
            assert mag < 1e-9 and np.abs(m.T_place - np.asarray(scn['T_nominal'])).max() > 0
        else:
            assert mag > 0, name
    other = single_in_range(cfg, 'gap', rng)
    assert other['shape']['tacks'] == [] and other['slip']['drift'] == [0.0, 0.0, 0.0]


def test_scale_cfg_scales_magnitudes_only(cfg):
    h = scale_cfg(cfg, 1.5)
    assert h['pre_weld']['gap_mm'] == [x * 1.5 for x in cfg['pre_weld']['gap_mm']]
    assert h['in_weld']['web_tilt_deg'] == cfg['in_weld']['web_tilt_deg'] * 1.5
    assert h['pre_weld']['tack_count'] == cfg['pre_weld']['tack_count']
    assert h['in_weld']['drift_prob'] == cfg['in_weld']['drift_prob']


@pytest.fixture(scope='module')
def batch_eps(cfg):
    batch = dict(make_batch(cfg, {'clean': 2, 'single': 0, 'mixed': 4, 'hard': 0}, seed=1))
    out = {}
    for name in ('clean_0000', 'mixed_0000', 'mixed_0001', 'mixed_0002'):
        scn = short(batch[name + '.json'])
        out[name] = (scn, rows_of(run_kinematic(scn, sensor_seed=2)))
    return out


def test_weld_only_episodes_pass_checks(batch_eps):
    for name, (scn, rows) in batch_eps.items():
        res = check(rows, scn)
        assert all(lv == OK for _, lv, _ in res), (name, res)
        assert rows['phase'][0] == 'weld' and rows['done'][-1]
        assert rows['laser'].shape[1] == 5


def test_straight_mode_has_no_weave(batch_eps):
    modes = {n: s['episode']['mode'] for n, (s, _) in batch_eps.items()}
    assert set(modes.values()) == {'weave', 'straight'}
    for name, (scn, rows) in batch_eps.items():
        if modes[name] == 'straight':
            assert np.all(rows['weave_amp'] == 0) and np.all(rows['weave_hz'] == 0)
        else:
            assert np.all(rows['weave_amp'] > 0)


def test_start_offset_and_perturbation_only_in_command(batch_eps):
    scn, rows = batch_eps['mixed_0002']
    assert scn['episode']['perturb']
    m = PartErrorModel(scn)
    # 시작 어긋남: 첫 주기 명령은 이음매 시작점에서 어긋남 만큼 벗어나 있고 점점 줄어듦
    off = np.abs(rows['perturb'][:, :2]).max(axis=1)
    so = np.abs(scn['episode']['start_offset'][:2]).max()
    assert off[0] == pytest.approx(so * np.exp(-0.01 / scn['episode']['start_settle']), rel=1e-6)
    # 섭동이 들어간 구간이 있음 (시작 어긋남이 사라진 뒤)
    late = rows['t'] > 5 * scn['episode']['start_settle']
    assert np.abs(rows['perturb'][late]).max() > 0
    # 정답은 섭동과 무관: 같은 시나리오에서 섭동을 끄고 돌린 정답과 같음
    calm = copy.deepcopy(scn)
    calm['episode'].update(perturb=False, start_offset=[0.0] * 4)
    rows2 = rows_of(run_kinematic(calm, sensor_seed=2))
    n = min(len(rows['e']), len(rows2['e']))
    assert np.allclose(rows['e'][:n], rows2['e'][:n]) and np.allclose(rows['T_expert'][:n], rows2['T_expert'][:n])
    # 보낸 목표 = 전문가 목표 + 어긋남
    i = np.flatnonzero(late)[0]
    lab = m.label(rows['s'][i], rows['s_arc'][i])
    assert np.allclose(apply_offset(rows['T_expert'][i], lab['R_true'], rows['perturb'][i]), rows['T_target'][i])


def test_start_pose_matches_offset(cfg):
    scn = make_batch(cfg, {'clean': 1, 'single': 0, 'mixed': 0, 'hard': 0}, seed=5)[0][1]
    ex = ScriptedExpert(PartErrorModel(scn), np.eye(4))
    lab = ex.m.label(0.0, 0.0)
    T0 = ex.seam_pose(0.0, 0.0, ex.ex['standoff_mm'] * MM)
    Ts = ex.start_pose()
    d = lab['R_true'].T @ (Ts[:3, 3] - T0[:3, 3])
    so = scn['episode']['start_offset']
    assert np.allclose(d, [0.0, so[0], so[1]], atol=1e-12)
    rv = lab['R_true'].T @ R_to_rotvec(Ts[:3, :3] @ T0[:3, :3].T)
    assert np.allclose(rv[:2], so[2:], atol=1e-9)


def test_unfinished_weld_only_fails(batch_eps):
    scn, rows = batch_eps['clean_0000']
    cut = {k: v[: len(v) // 2] for k, v in rows.items()}
    res = dict((n, lv) for n, lv, _ in check(cut, scn))
    assert res['단계 순서'] == FAIL
