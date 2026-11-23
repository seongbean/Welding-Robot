"""파이프 원주 시편군 (pipe_girth): config, 관 둘레 명목 이음매, 무잡음 단면 측정 = 정답, 진원도·단차, 메시, 운동학 에피소드."""
import math

import numpy as np
import pytest
import yaml

from sim_errors.check_episode import OK, check
from sim_errors.episode import run_kinematic
from sim_errors.expert import auto_nominal
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch
from sim_errors.part_errors import (BASE, DEG, MM, WEB, PartErrorModel, family_cfg, sample_scenario, scale_cfg,
                                    signature, single_error, single_in_range, single_names, zero_errors)


@pytest.fixture(scope='module')
def raw():
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='module')
def pipe(raw):
    return family_cfg(raw, 'pipe_girth')


def _ref(lab):
    return np.array([lab['e'][1], lab['e'][2] + lab['tack'], lab['gap'], lab['rotvec'][0], lab['rotvec'][1]])


def test_family_cfg_and_geometry(raw, pipe):
    g = pipe['geometry']
    assert g['joint'] == 'pipe_girth' and 'web_height_mm' not in g and 'length_mm' not in g
    assert pipe['expert'] == family_cfg(raw)['expert'] and pipe['sensors'] == family_cfg(raw)['sensors']
    scn = sample_scenario(pipe, np.random.default_rng(0))
    geo = scn['geometry']
    assert geo['radius'] == pytest.approx((323.9 / 2 - 12 + 1.5) * MM)
    assert geo['length'] == pytest.approx(geo['radius'] * 100 * DEG)
    assert {'hi_lo', 'oval_l', 'oval_r'} <= set(scn['shape'])
    # 다른 군 config에는 진원도가 없고, T 필릿 시나리오 난수는 그대로
    assert 'oval_l' not in sample_scenario(family_cfg(raw), np.random.default_rng(0))['shape']


def test_batch_names_and_single_names(raw, pipe):
    counts = {'clean': 1, 'single': 3, 'mixed': 2, 'hard': 1}
    pb = make_batch(pipe, counts, seed=0)
    bb = make_batch(family_cfg(raw, 'butt_v'), counts, seed=0)
    assert all(n.startswith('pipe_') for n, _ in pb)
    assert [s for n, s in pb if 'mixed' in n][0]['icp'] != [s for n, s in bb if 'mixed' in n][0]['icp']
    names = single_names(pipe)
    assert len(names) == 15 and {'hi_lo', 'oval'} <= set(names) and 'web_square' not in names
    assert 'oval' not in single_names(family_cfg(raw, 'butt_v'))
    scn = single_in_range(pipe, 'oval', np.random.default_rng(0))
    assert scn['shape']['oval_l'] != 0 and scn['shape']['hi_lo'] == 0 and scn['shape']['gap0'] == 0
    assert scale_cfg(pipe, 1.5)['pre_weld']['oval_mm'] == pytest.approx(1.5)


def test_nominal_seam_is_arc_over_top(pipe):
    m = PartErrorModel(zero_errors(sample_scenario(pipe, np.random.default_rng(1))))
    r, arc = m.scn['geometry']['radius'], m.scn['geometry']['arc']
    s = np.linspace(0, 1, 11)
    P, Rn = m.nominal(s)
    assert np.allclose(np.linalg.norm(P[:, [0, 2]], axis=1), r) and np.allclose(P[:, 1], 0)
    assert np.allclose(P[5], [0, 0, r]) and np.allclose(Rn[5], np.eye(3))
    assert np.allclose(Rn[:, :, 2], P / r)                               # z = 바깥 반지름 방향
    assert math.atan2(P[-1, 0], P[-1, 2]) == pytest.approx(arc / 2)
    # 호 길이 = L, 실제 = 추정 (오차 0)
    assert np.sum(np.linalg.norm(np.diff(m.nominal(np.linspace(0, 1, 2001))[0], axis=0), axis=1)) == pytest.approx(m.L)
    Pt, Rt = m.seam_true(s, 0.5)
    Pe, Re = m.seam_est(s)
    assert np.allclose(Pt, Pe, atol=1e-12) and np.allclose(Rt, Re, atol=1e-9)
    for r_ in signature(m.scn):
        assert max(abs(r_[k]) for k in ('e_y', 'e_z', 'work', 'travel')) < 1e-9


@pytest.mark.parametrize('name', ['gap', 'tack', 'base_angle', 'web_tilt', 'bow', 'drift', 'hi_lo', 'oval',
                                  'web_straightness', 'icp'])
def test_section_fit_matches_label_without_noise(pipe, name):
    """노이즈 없으면 관 둘레 개선 면 평면 맞춤(곡률 보정 포함)이 정답과 같다 (센서 치우침이 없어야 공용 MLP가 다른 군을 안 망친다)."""
    scn = single_error(sample_scenario(pipe, np.random.default_rng(2)), name)
    if name in ('base_angle', 'web_tilt', 'oval'):
        scn['shape'].update(gap0=3 * MM, gap1=2 * MM)
    m = PartErrorModel(scn)
    for s in (0.0, 0.2, 0.5, 0.8, 1.0):
        for s_arc in (0.0, max(0.0, s - 0.1)):
            d = m.section_fit(s, s_arc) - _ref(m.label(s, s_arc))
            assert np.all(np.abs(d[:3]) < 0.03 * MM) and np.all(np.abs(d[3:]) < 0.03 * DEG), (s, s_arc, d)


def test_mixed_section_bias_is_small(pipe):
    rng = np.random.default_rng(3)
    res = []
    for _ in range(10):
        m = PartErrorModel(sample_scenario(pipe, rng))
        for s in np.linspace(0, 0.88, 8):
            look = s + m.scn['sensors']['laser_lookahead'] / m.L
            res.append(m.section_fit(look, s) - _ref(m.label(look, s)))
        res.append(m.section_fit(0.0, 0.0) - _ref(m.label(0.0, 0.0)))      # 시작 구간 스캔 (아크 전)
    r = np.array(res)
    assert np.all(np.abs(r.mean(axis=0)[:3]) < 0.005 * MM) and np.all(np.abs(r[:, :3]).max(axis=0) < 0.06 * MM)
    assert np.all(np.abs(r[:, 3:]).max(axis=0) < 0.06 * DEG)


def test_oval_gives_varying_hi_lo_and_height(pipe):
    scn = single_error(sample_scenario(pipe, np.random.default_rng(4)), 'oval')   # 왼쪽 +1.5, 오른쪽 -1.5 (위상 0)
    m = PartErrorModel(scn)
    s = np.array([0.0, 0.5, 1.0])
    psi = m.psi(s)
    assert np.allclose(m.hi_lo(s), -3 * MM * np.cos(2 * psi))
    sig = {round(r['s'], 2): r for r in signature(scn)}
    assert sig[0.5]['e_z'] == pytest.approx(0.0, abs=1e-6)                  # 평균 반지름 편차 0
    assert abs(sig[0.5]['e_y']) < 1e-9
    meas = m.section_fit(0.5, 0.0)
    assert abs(meas[0]) < 0.01 * MM                                          # 단차가 3 mm여도 좌우 치우침 없음


def test_pipe_mesh_ring_and_root_edges(pipe):
    scn = sample_scenario(pipe, np.random.default_rng(6))
    m = PartErrorModel(scn)
    pts, plate, counts, idx = m.nominal_meshes(nx=20)
    g = scn['geometry']
    assert pts[:, 0].max() - pts[:, 0].min() == pytest.approx(2 * math.pi * g['radius'])   # 관 둘레 전체
    eL = (plate == BASE) & (pts[:, 1] == 0) & (pts[:, 2] == 0) & (pts[:, 0] >= 0) & (pts[:, 0] <= m.L)
    eR = (plate == WEB) & (pts[:, 1] == 0) & (pts[:, 2] == 0) & (pts[:, 0] >= 0) & (pts[:, 0] <= m.L)
    assert eL.sum() >= 3
    for s_arc in (0.0, 0.6, 1.0):
        PL = m.deform_points(pts[eL], plate[eL], s_arc)
        PR = m.deform_points(pts[eR], plate[eR], s_arc)
        Pt, _ = m.seam_true(pts[eL, 0] / m.L, s_arc)
        assert np.allclose((PL + PR) / 2, Pt, atol=1e-9)
    # 명목(오차 0) 바깥 면은 반지름 = 바깥지름/2
    z = PartErrorModel(zero_errors(scn))
    top = np.isclose(pts[:, 2], g['thickness'] - g['root_face'])
    P = (z.deform_points(pts[top], plate[top], 0.0) - z.T_place[:3, 3]) @ z.T_place[:3, :3]
    assert np.allclose(np.hypot(P[:, 0], P[:, 2]), 323.9 / 2 * MM, atol=1e-9)


def test_auto_nominal_puts_seam_middle_below_tip(pipe):
    m = PartErrorModel(zero_errors(sample_scenario(pipe, np.random.default_rng(7))))
    T_tip = np.eye(4)
    T_tip[:3, 3] = [0.5, 0.0, 0.3]
    m.set_nominal(auto_nominal(T_tip, m.L, mid_part=m.nominal(0.5)[0][0]))
    P, R = m.seam_true(0.5, 0.0)
    assert np.allclose(P[0], [0.5, 0.0, 0.28]) and np.allclose(R[0][:, 2], [0, 0, 1])


def test_pipe_kinematic_episode_passes_checks(pipe):
    _, scn = make_batch(pipe, {'clean': 0, 'single': 0, 'mixed': 1, 'hard': 0}, seed=3)[0]
    g = scn['geometry']
    g['arc'] = 0.06 / g['radius']           # 짧은 호로 (L = 반지름 × 호 각)
    g['length'] = 0.06
    rec = run_kinematic(scn, sensor_seed=1)
    rows = {k: np.array(v) for k, v in rec.rows.items()}
    res = check(rows, scn)
    assert all(lv == OK for _, lv, _ in res), res
    assert rows['done'][-1] and np.isfinite(rows['laser'][:, 3]).any()
