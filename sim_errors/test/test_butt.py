"""맞대기 V홈 시편군 (butt_v): config 덮어쓰기, 묶음 이름·난수, 개선 면 평면 맞춤 센서, 단차 오차, 모재 메시."""
import numpy as np
import pytest
import yaml

from sim_errors.check_episode import OK, check
from sim_errors.episode import run_kinematic
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch
from sim_errors.part_errors import (BASE, DEG, MM, WEB, PartErrorModel, family_cfg, sample_scenario, scale_cfg,
                                    signature, single_error, single_in_range, single_names, zero_errors)


@pytest.fixture(scope='module')
def raw():
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='module')
def butt(raw):
    return family_cfg(raw, 'butt_v')


def _ref(lab):
    return np.array([lab['e'][1], lab['e'][2] + lab['tack'], lab['gap'], lab['rotvec'][0], lab['rotvec'][1]])


def test_family_cfg_replaces_geometry(raw, butt):
    g = butt['geometry']
    assert g['joint'] == 'butt_v' and 'web_height_mm' not in g and g['bevel_deg'] == 30
    assert butt['pre_weld']['hi_lo_mm'] == 1.0 and 'hi_lo_mm' not in family_cfg(raw)['pre_weld']
    assert butt['expert'] == family_cfg(raw)['expert'] and butt['sensors'] == family_cfg(raw)['sensors']
    scn = sample_scenario(butt, np.random.default_rng(0))
    assert set(scn['geometry']) == {'joint', 'length', 'plate_width', 'thickness', 'bevel', 'root_face'}
    assert 'hi_lo' in scn['shape']


def test_batch_names_and_single_names(raw, butt):
    counts = {'clean': 1, 'single': 3, 'mixed': 2, 'hard': 1}
    bb = make_batch(butt, counts, seed=0)
    cb = make_batch(family_cfg(raw, 'fillet_corner'), counts, seed=0)
    assert all(n.startswith('butt_') for n, _ in bb)
    assert [s for n, s in bb if 'mixed' in n][0]['icp'] != [s for n, s in cb if 'mixed' in n][0]['icp']
    names = single_names(butt)
    assert len(names) == 14 and 'hi_lo' in names and 'web_square' not in names
    assert 'hi_lo' not in single_names(family_cfg(raw))
    scn = single_in_range(butt, 'hi_lo', np.random.default_rng(0))
    assert scn['shape']['hi_lo'] != 0 and scn['shape']['gap0'] == 0
    assert scale_cfg(butt, 1.5)['pre_weld']['hi_lo_mm'] == pytest.approx(1.5)


def test_nominal_seam_frame_points_up(butt):
    m = PartErrorModel(zero_errors(sample_scenario(butt, np.random.default_rng(1))))
    _, R_true = m.seam_true(0.5, 0.5)
    _, R_est = m.seam_est(0.5)
    Rp = m.T_place[:3, :3]
    assert np.allclose(Rp.T @ R_true[0], np.eye(3), atol=1e-9)
    assert np.allclose(R_true, R_est, atol=1e-9)


@pytest.mark.parametrize('name', ['gap', 'tack', 'base_angle', 'web_tilt', 'residual', 'bow', 'drift'])
def test_section_fit_matches_label_without_noise(butt, name):
    """노이즈 없으면 개선 면 평면 맞춤 + 틈새 보정이 정답(좌우, 높이+가접, 틈새, 작업각, 진행각)과 같다."""
    scn = single_error(sample_scenario(butt, np.random.default_rng(2)), name)
    if name in ('base_angle', 'web_tilt'):
        scn['shape'].update(gap0=2 * MM, gap1=1 * MM)          # 틈새와 각변형이 함께 있어도
    m = PartErrorModel(scn)
    for s in (0.2, 0.5, 0.8):
        for s_arc in (0.0, s - 0.1):
            d = m.section_fit(s, s_arc) - _ref(m.label(s, s_arc))
            assert np.all(np.abs(d[:3]) < 0.02 * MM) and np.all(np.abs(d[3:]) < 0.02 * DEG), (s, s_arc, d)


def test_angle_errors_mean_what_the_config_says(butt):
    """양쪽 각변형(base_angle)은 이등분선을 안 바꾸고, 좌우 차이(web_tilt)는 작업각 = 절반."""
    base = sample_scenario(butt, np.random.default_rng(4))
    sym = single_error(base, 'base_angle')
    lab = PartErrorModel(sym).label(0.2, 0.8)
    assert abs(lab['rotvec'][0]) < 1e-9 and np.linalg.norm(lab['e']) < 1e-12
    asym = single_error(base, 'web_tilt')
    m = PartErrorModel(asym)
    assert m.label(0.2, 1.0)['rotvec'][0] == pytest.approx(asym['heat']['web_tilt'] / 2 * m._heat_profile(0.2, 1.0),
                                                           rel=1e-6)


def test_hi_lo_raises_root_and_top_step_removes_lateral_bias(butt):
    """단차 h: 정답 높이 = h/2. 개선 면 교선만 쓰면 좌우가 -h·tan(개선각)/2 밀리지만, 판 윗면 높이 차로 h를 재서 되돌린다."""
    scn = single_error(sample_scenario(butt, np.random.default_rng(5)), 'hi_lo')
    h = scn['shape']['hi_lo']
    for r in signature(scn):
        assert r['e_z'] == pytest.approx(h / MM / 2) and abs(r['e_y']) < 1e-9 and abs(r['work']) < 1e-9
    m = PartErrorModel(scn)
    meas = m.section_fit(0.5, 0.0)
    assert meas[1] == pytest.approx(h / 2, abs=1e-6)
    assert abs(meas[0]) < 1e-6
    z = m.seam_true(0.5, 0.0)[1][0][:, 2]
    assert m._butt_top_step(np.array([0.5]), 0.0, z, 0.015, 0.001, 0.0, 0.0, None) == pytest.approx(h, abs=1e-9)
    # 노이즈가 있어도 치우침 없이 (평균 0.02 mm 안)
    rng = np.random.default_rng(0)
    lat = [m.section_fit(0.5, 0.0, rng)[0] for _ in range(200)]
    assert abs(np.nanmean(lat)) < 0.02 * MM


def test_butt_mesh_root_edges_straddle_seam(butt):
    scn = sample_scenario(butt, np.random.default_rng(6))
    m = PartErrorModel(scn)
    pts, plate, counts, idx = m.nominal_meshes(nx=20)
    g = scn['geometry']
    assert pts[plate == BASE][:, 1].min() == pytest.approx(-g['plate_width'])
    assert pts[plate == WEB][:, 1].max() == pytest.approx(g['plate_width'])
    assert pts[:, 2].max() == pytest.approx(g['thickness'] - g['root_face'])
    eL = (plate == BASE) & (pts[:, 1] == 0) & (pts[:, 2] == 0)
    eR = (plate == WEB) & (pts[:, 1] == 0) & (pts[:, 2] == 0)
    for s_arc in (0.0, 0.6, 1.0):
        PL = m.deform_points(pts[eL], plate[eL], s_arc)
        PR = m.deform_points(pts[eR], plate[eR], s_arc)
        Pt, _ = m.seam_true(pts[eL, 0] / m.L, s_arc)
        assert np.allclose((PL + PR) / 2, Pt, atol=1e-12)
        gap = m.gap(pts[eL, 0] / m.L, s_arc)
        assert np.allclose(np.linalg.norm(PR - PL, axis=1), np.hypot(gap, m.hi_lo()), atol=1e-9)


def test_butt_kinematic_episode_passes_checks(butt):
    _, scn = make_batch(butt, {'clean': 0, 'single': 0, 'mixed': 1, 'hard': 0}, seed=3)[0]
    scn['geometry']['length'] = 0.06
    rec = run_kinematic(scn, sensor_seed=1)
    rows = {k: np.array(v) for k, v in rec.rows.items()}
    res = check(rows, scn)
    assert all(lv == OK for _, lv, _ in res), res
    assert rows['done'][-1] and np.isfinite(rows['laser'][:, 3]).any()
