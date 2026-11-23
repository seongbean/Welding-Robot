"""ㄱ자 코너 필릿 시편군 (fillet_corner): config 덮어쓰기, 묶음 이름·난수, 모재 모양, 웹 직각도 오차."""
import numpy as np
import pytest
import yaml

from sim_errors.check_episode import OK, check
from sim_errors.episode import run_kinematic
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch
from sim_errors.part_errors import (BASE, DEG, MM, WEB, PartErrorModel, family_cfg, sample_scenario, scale_cfg,
                                    signature, single_error, single_in_range, single_names)


@pytest.fixture(scope='module')
def raw():
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='module')
def corner(raw):
    return family_cfg(raw, 'fillet_corner')


def test_family_cfg_overrides_only_listed_keys(raw, corner):
    t = family_cfg(raw)
    assert 'families' not in t and 'families' not in corner
    assert t['geometry']['joint'] == 'fillet_t' and corner['geometry']['joint'] == 'fillet_corner'
    assert corner['in_weld']['web_tilt_deg'] != t['in_weld']['web_tilt_deg']
    assert corner['pre_weld']['icp_xyz_mm'] == t['pre_weld']['icp_xyz_mm']       # 공통값은 그대로
    assert corner['expert'] == t['expert'] and corner['sensors'] == t['sensors']
    with pytest.raises(ValueError):
        family_cfg(raw, 'no_such_family')


def test_batch_names_and_draws_differ_from_t(raw, corner):
    counts = {'clean': 1, 'single': 3, 'mixed': 4, 'hard': 1}
    tb, cb = make_batch(family_cfg(raw), counts, seed=0), make_batch(corner, counts, seed=0)
    assert all(not n.startswith('corner_') for n, _ in tb)
    assert all(n.startswith('corner_') for n, _ in cb)
    assert all(s['geometry']['joint'] == 'fillet_corner' for _, s in cb)
    # 같은 시드여도 T와 다른 오차 (밀림·ICP)
    tm = [s for n, s in tb if 'mixed' in n][0]
    cm = [s for n, s in cb if 'mixed' in n][0]
    assert tm['icp'] != cm['icp']
    assert 'web_square' in cm['shape'] and 'web_square' not in tm['shape']


def test_single_names_add_web_square_only_for_corner(raw, corner):
    t = single_names(family_cfg(raw))
    c = single_names(corner)
    assert len(t) == 13 and 'web_square' not in t
    assert set(c) == set(t) | {'web_square'} and len(c) == 14
    with pytest.raises(ValueError):
        single_in_range(family_cfg(raw), 'web_square', np.random.default_rng(0))
    scn = single_in_range(corner, 'web_square', np.random.default_rng(0))
    assert scn['shape']['web_square'] != 0 and scn['shape']['tacks'] == []


def test_web_square_is_static_work_angle(corner):
    """조립 직각도는 용접 전부터 있고 아크와 무관: 작업각 = 기울기의 절반, 위치 정답은 0."""
    base = sample_scenario(corner, np.random.default_rng(1))
    scn = single_error(base, 'web_square')
    for r in signature(scn):
        assert r['work'] == pytest.approx(1.5, abs=1e-6)
        assert abs(r['e_y']) < 1e-9 and abs(r['e_z']) < 1e-9 and abs(r['travel']) < 1e-9
    m = PartErrorModel(scn)
    meas = m.section_fit(0.5, 0.0)          # 단면 센서가 아크 전에도 본다
    assert meas[3] == pytest.approx(1.5 * DEG, abs=1e-6)


def test_corner_mesh_one_sided_and_matches_seam(corner):
    scn = sample_scenario(corner, np.random.default_rng(3))
    scn['shape'].update(gap0=0.0, gap1=0.0)
    m = PartErrorModel(scn)
    pts, plate, counts, idx = m.nominal_meshes(nx=20)
    g = scn['geometry']
    b = pts[plate == BASE]
    assert b[:, 1].min() == pytest.approx(-g['base_width']) and b[:, 1].max() == pytest.approx(g['web_thickness'])
    edge = (plate == WEB) & (pts[:, 1] == 0) & (pts[:, 2] == 0)
    for s_arc in (0.0, 0.6, 1.0):
        P = m.deform_points(pts[edge], plate[edge], s_arc)
        Pt, _ = m.seam_true(pts[edge, 0] / m.L, s_arc)
        assert np.allclose(P, Pt, atol=1e-12)


def test_hard_scale_includes_web_square(corner):
    assert scale_cfg(corner, 1.5)['pre_weld']['web_square_deg'] == pytest.approx(3.0)


def test_corner_kinematic_episode_passes_checks(corner):
    _, scn = make_batch(corner, {'clean': 0, 'single': 0, 'mixed': 1, 'hard': 0}, seed=3)[0]
    scn['geometry']['length'] = 0.06
    rec = run_kinematic(scn, sensor_seed=1)
    rows = {k: np.array(v) for k, v in rec.rows.items()}
    res = check(rows, scn)
    assert all(lv == OK for _, lv, _ in res), res
    assert rows['done'][-1] and np.isfinite(rows['laser'][:, 3]).any()
    assert np.max(np.abs(rows['e'][:, 1])) < 20 * MM
