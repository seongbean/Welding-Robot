"""sim_errors: 모재 오차 모델과 규칙 전문가 (Isaac Sim 없이)."""
import copy
import math

import numpy as np
import pytest
import yaml

from sim_errors.expert import ScriptedExpert, auto_nominal
from sim_errors.make_scenarios import DEFAULT_CFG
from sim_errors.part_errors import (MM, WEB, DEG, SINGLE_ERRORS, PartErrorModel, R_SEAM_NOM, load_scenario,
                                    next_pass, R_to_rotvec, rotvec_to_R, sample_scenario, save_scenario, signature,
                                    single_error, summarize)


@pytest.fixture(scope='module')
def cfg():
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        return yaml.safe_load(f)


def zero_scenario(cfg):
    """모든 오차 0인 시나리오 (항목을 하나씩 켜서 시험)."""
    scn = sample_scenario(cfg, np.random.default_rng(1))
    scn['place'] = {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]}
    scn['icp'] = {'xyz': [0, 0, 0], 'rotvec': [0, 0, 0]}
    for k in ('web_bow', 'web_wave', 'base_bow', 'base_wave', 'gap0', 'gap1'):
        scn['shape'][k] = 0.0
    scn['shape']['tacks'] = []
    scn['slip'].update(drift=[0, 0, 0], jump=[0, 0, 0])
    scn['heat'].update(base_angle=0.0, web_tilt=0.0, bow=0.0, gap_close=0.0)
    scn['sensors'].update(laser_noise=0.0, laser_dropout=0.0)
    return scn


S = np.linspace(0, 1, 11)


def test_rotation_utils_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(size=3) * 0.5
        assert np.allclose(R_to_rotvec(rotvec_to_R(v)), v, atol=1e-9)


def test_zero_error_true_equals_estimate(cfg):
    m = PartErrorModel(zero_scenario(cfg))
    for s_arc in (0.0, 0.5, 1.0):
        Pt, Rt = m.seam_true(S, s_arc)
        Pe, Re = m.seam_est(S)
        assert np.allclose(Pt, Pe, atol=1e-12)
        assert np.allclose(Rt, Re, atol=1e-6)
    lab = m.label(0.3, 0.3)
    assert np.allclose(lab['e'], 0, atol=1e-12) and np.allclose(lab['rotvec'], 0, atol=1e-6)
    # 명목 이음매 좌표계: z는 바닥 법선과 웹 법선의 이등분, 오른손 좌표계
    assert np.allclose(R_SEAM_NOM[:, 2], [0, -1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert np.isclose(np.linalg.det(R_SEAM_NOM), 1.0)


def test_placement_error_is_known_to_estimate(cfg):
    """배치 오차는 ICP가 잡으므로 정답에 남지 않고, ICP 잔차만 남는다."""
    scn = zero_scenario(cfg)
    scn['place'] = {'xyz': [0.003, -0.002, 0.001], 'rpy': [0.01, -0.02, 0.015]}
    m = PartErrorModel(scn)
    assert np.allclose(m.label(0.5, 0.5)['e'], 0, atol=1e-12)
    scn['icp'] = {'xyz': [0.001, 0, 0], 'rotvec': [0, 0, 0]}
    m = PartErrorModel(scn)
    assert np.isclose(np.linalg.norm(m.label(0.5, 0.5)['e']), 0.001, atol=1e-9)


def test_web_bow_moves_seam_laterally(cfg):
    scn = zero_scenario(cfg)
    scn['shape']['web_bow'] = 0.002
    m = PartErrorModel(scn)
    e_mid = m.label(0.5, 0.0)['e']
    # 모재 y(웹 쪽) 0.002 → 이음매 좌표계 y,z에 각각 0.002/√2 (y축 = (0,1,1)/√2, z축 = (0,-1,1)/√2)
    assert np.allclose(e_mid, [0, 0.002 / math.sqrt(2), -0.002 / math.sqrt(2)], atol=1e-9)
    assert np.allclose(m.label(0.0, 0.0)['e'], 0, atol=1e-12)


def test_slip_drift_ramp_and_jump(cfg):
    scn = zero_scenario(cfg)
    scn['slip'] = {'drift': [0.002, 0, 0], 'drift_s0': 0.2, 'drift_span': 0.2, 'jump': [0, 0.001, 0], 'jump_s': 0.7}
    m = PartErrorModel(scn)
    assert np.allclose(m.slip(0.1), 0)
    assert np.allclose(m.slip(0.3), [0.001, 0, 0])
    assert np.allclose(m.slip(0.5), [0.002, 0, 0])
    assert np.allclose(m.slip(0.8), [0.002, 0.001, 0])


def test_heat_distortion_behind_and_ahead_of_arc(cfg):
    scn = zero_scenario(cfg)
    scn['heat'].update(base_angle=1.5 * DEG, length=0.02, bow=0.001, gap_close=0.0008)
    scn['shape'].update(gap0=0.001, gap1=0.001)
    m = PartErrorModel(scn)
    s_arc = 0.5
    assert m.base_angle(np.array([0.6]), s_arc)[0] == 0.0                       # 아크 앞: 각변형 없음
    far_behind = m.base_angle(np.array([0.0]), s_arc)[0]
    assert np.isclose(far_behind, 1.5 * DEG, rtol=0.01)                         # 아크 한참 뒤: 최종값
    # 틈새 닫힘: 아크 앞은 진행도만큼(토치 위치에서 절반, 끝에서 최대), 아크 뒤는 지나갈 때 값으로 굳음
    assert m.gap(np.array([1.0]), s_arc)[0] == pytest.approx(0.001 - 0.0008 * 0.5)
    assert m.gap(np.array([0.5]), s_arc)[0] == pytest.approx(0.001 - 0.0008 * 0.5 * 0.5)   # 토치 위치
    assert m.gap(np.array([0.4]), s_arc)[0] == pytest.approx(0.001 - 0.0008 * 0.5 * 0.4)
    assert m.gap(np.array([0.5]), 0.0)[0] == pytest.approx(0.001)                         # 용접 전
    # 세로 휨은 진행도에 비례해 아크 앞쪽도 움직인다
    assert m.label(0.75, 0.5)['e'][2] != 0.0
    assert m.label(0.75, 0.0)['e'][2] == pytest.approx(0.0, abs=1e-12)


def test_web_tilt_rotates_bisector_by_half(cfg):
    scn = zero_scenario(cfg)
    scn['heat'].update(web_tilt=2.0 * DEG, length=1e-4)
    m = PartErrorModel(scn)
    r = m.label(0.2, 1.0)['rotvec']
    assert abs(r[0]) == pytest.approx(1.0 * DEG, rel=1e-3)      # 작업각 = 기울어짐의 절반
    assert np.allclose(r[1:], 0, atol=1e-9)


def test_mesh_seam_corner_matches_true_seam(cfg):
    """틈새가 없을 때 변형된 웹 판의 용접 쪽 아래 모서리(y=0, z=0)가 실제 이음매 경로와 일치."""
    scn = sample_scenario(cfg, np.random.default_rng(3))
    scn['shape'].update(gap0=0.0, gap1=0.0)
    m = PartErrorModel(scn)
    pts, plate, counts, idx = m.nominal_meshes(nx=20)
    assert idx.max() < len(pts) and counts.sum() == len(idx)
    corner = (plate == WEB) & (pts[:, 1] == 0) & (pts[:, 2] == 0)
    for s_arc in (0.0, 0.6, 1.0):
        P = m.deform_points(pts[corner], plate[corner], s_arc)
        Pt, _ = m.seam_true(pts[corner, 0] / m.L, s_arc)
        assert np.allclose(P, Pt, atol=1e-12)


def test_scenario_ranges_and_json_roundtrip(cfg, tmp_path):
    rng = np.random.default_rng(7)
    for _ in range(50):
        scn = sample_scenario(cfg, rng)
        assert all(abs(v) <= 3 * MM for v in scn['place']['xyz'])
        assert 0.3 * MM <= np.linalg.norm(scn['icp']['xyz']) <= 1.5 * MM
        assert 2 <= len(scn['shape']['tacks']) <= 4
        assert scn['shape']['gap1'] >= 0
    p = tmp_path / 's.json'
    save_scenario(scn, str(p))
    back = load_scenario(str(p))
    m1, m2 = PartErrorModel(scn), PartErrorModel(back)
    assert np.allclose(m1.seam_true(S, 0.4)[0], m2.seam_true(S, 0.4)[0])


def test_next_pass_carries_residual_distortion(cfg):
    rng = np.random.default_rng(5)
    s0 = sample_scenario(cfg, rng)
    s1 = next_pass(s0, cfg, rng)
    s2 = next_pass(s1, cfg, rng)
    assert s2['pass_index'] == 2
    assert s1['place'] == s0['place'] and s1['shape']['web_bow'] == s0['shape']['web_bow']
    assert s1['shape']['tacks'] == [] and s1['shape']['gap0'] == 0.0
    assert s2['prior']['bow'] == pytest.approx(s0['heat']['bow'] + s1['heat']['bow'])
    # 앞 패스의 각변형이 다음 패스 시작(s_arc=0)부터 형상에 들어 있음
    m1 = PartErrorModel(s1)
    assert m1.base_angle(np.array([0.5]), 0.0)[0] == pytest.approx(s0['heat']['base_angle'])


def test_summary_keys(cfg):
    out = summarize(sample_scenario(cfg, np.random.default_rng(0)), n=21)
    assert out['lateral_max_mm'] >= 0 and 'travel_angle_max_deg' in out


def test_auto_nominal_places_seam_under_tip():
    T_tip = np.eye(4)
    T_tip[:3, 3] = [0.5, 0.1, 0.3]
    T = auto_nominal(T_tip, 0.3, drop=0.02)
    mid = T[:3, :3] @ np.array([0.15, 0, 0]) + T[:3, 3]
    assert np.allclose(mid, [0.5, 0.1, 0.28])
    assert np.allclose(T[:3, 1], [1, 0, 0])            # 웹 판은 로봇 반대쪽(+X)


def run_expert(scn, dt=0.02):
    m = PartErrorModel(copy.deepcopy(scn))
    T0 = np.eye(4)
    T0[:3, :3] = np.diag([1.0, -1.0, -1.0])            # 토치가 아래를 향한 시작 자세
    T0[:3, 3] = [0.5, 0.0, 0.3]
    m.set_nominal(auto_nominal(T0, m.L))
    ex = ScriptedExpert(m, T0, np.random.default_rng(0))
    out = []
    for _ in range(200000):
        out.append(ex.step(dt))
        if ex.done:
            break
    return m, ex, out


def test_expert_full_episode(cfg):
    scn = sample_scenario(cfg, np.random.default_rng(11))
    scn['sensors']['laser_dropout'] = 0.0
    m, ex, out = run_expert(scn)
    assert ex.done
    phases = [r['phase'] for r in out]
    order = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    assert order == ['approach', 'descend', 'weld', 'retract'] and out[-1]['done']
    # 단계가 바뀌는 주기도 그 목표를 만든 단계로 기록 (첫 용접 주기부터 진행도가 0보다 큼)
    assert all(r['s'] > 0 for r in out if r['arc_on'])
    weld = [r for r in out if r['arc_on']]
    s = np.array([r['s'] for r in out])
    assert np.all(np.diff(s) >= 0) and s.max() == pytest.approx(1.0)
    ex_cfg = scn['expert']
    for r in weld[::50]:
        lab = m.label(r['s'], r['s_arc'])
        d = r['T_target'][:3, 3] - lab['p_true']
        R = lab['R_true']
        # 목표 = 실제 이음매 + 이등분 방향(standoff + 가접) + 좌우 위빙
        lift = ex_cfg['standoff_mm'] * MM + ex_cfg['tack_follow'] * lab['tack']
        assert np.dot(d, R[:, 2]) == pytest.approx(lift, abs=1e-9)
        assert abs(np.dot(d, R[:, 1])) <= r['weave_amp'] + 1e-12
        assert abs(np.dot(d, R[:, 0])) < 1e-9
        # 토치는 이음매 쪽을 향하고 진행 방향으로 travel_angle만큼 기움
        z_tool = r['T_target'][:3, 2]
        assert np.dot(z_tool, -R[:, 2]) == pytest.approx(math.cos(ex_cfg['travel_angle_deg'] * DEG), abs=1e-9)
        assert np.allclose(r['T_target'][:3, :3].T @ r['T_target'][:3, :3], np.eye(3), atol=1e-9)
        assert np.all(np.isfinite(r['laser'])) or r['s'] + scn['sensors']['laser_lookahead'] / m.L > 1
    # 용접 시간 ≈ 길이 / 속도 (틈새가 넓으면 느려짐)
    v_nom = ex_cfg['travel_speed_mm_s'] * MM
    t_weld = len(weld) * 0.02
    assert m.L / v_nom * 0.95 <= t_weld <= m.L / (v_nom * ex_cfg['min_speed_ratio']) * 1.05


def test_expert_gap_rule(cfg):
    scn = sample_scenario(cfg, np.random.default_rng(2))
    ex = ScriptedExpert(PartErrorModel(scn), np.eye(4))
    a0, v0 = ex.process(0.0)
    a2, v2 = ex.process(0.002)
    assert a2 > a0 and v2 < v0
    a_big, v_big = ex.process(0.01)
    assert a_big == pytest.approx(scn['expert']['weave_amp_max_mm'] * MM)
    assert v_big == pytest.approx(scn['expert']['travel_speed_mm_s'] * MM * scn['expert']['min_speed_ratio'])


def test_single_error_scenarios(cfg):
    base = sample_scenario(cfg, np.random.default_rng(0))
    sig = {k: signature(single_error(base, k)) for k in SINGLE_ERRORS}
    col = lambda k, f: np.array([r[f] for r in sig[k]])   # noqa: E731
    # 배치 오차는 ICP가 잡으므로 정답 0, ICP 잔차는 일정한 오프셋
    assert np.allclose(col('place', 'e_y'), 0) and np.allclose(col('place', 'e_z'), 0)
    assert np.allclose(col('icp', 'e_y'), -3 / math.sqrt(2)) and np.allclose(col('icp', 'e_z'), 3 / math.sqrt(2))
    # 밀림은 구간 전 0, 끝난 뒤 10 mm / √2, 튐은 계단
    assert col('drift', 'e_y')[1] == 0 and col('drift', 'e_y')[-1] == pytest.approx(10 / math.sqrt(2))
    assert col('jump', 'e_y')[1] == 0 and col('jump', 'e_y')[2] == pytest.approx(5 / math.sqrt(2))
    # 위치를 안 바꾸는 오차: 틈새·가접·틈새 닫힘
    for k in ('gap', 'tack', 'gap_close'):
        assert np.allclose(col(k, 'e_y'), 0) and np.allclose(col(k, 'e_z'), 0)
    assert col('gap_close', 'gap')[-1] < col('gap_close', 'gap')[0]       # 토치가 보는 틈새가 줄어듦
    # 각변형은 아크 뒤에서만 생겨 한 패스 안의 토치 위치에서는 0, 다음 패스에서는 작업각에 남음
    assert np.allclose(col('base_angle', 'work'), 0) and np.allclose(col('web_tilt', 'work'), 0)
    assert np.all(np.abs(col('residual', 'work')) > 0.4)
    # 단독 시나리오에는 다른 오차가 없음
    z = single_error(base, 'drift')
    assert z['shape']['tacks'] == [] and z['heat']['bow'] == 0 and z['icp']['xyz'] == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        single_error(base, 'nope')
