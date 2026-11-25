"""모재까지 부호 거리 (팔·토치–모재 간격 점검용): 네 시편군에서 알려진 점의 거리."""
import numpy as np
import pytest
import yaml

from sim_errors.clearance import flat_coords, plate_distances
from sim_errors.make_scenarios import DEFAULT_CFG, make_batch
from sim_errors.part_errors import BASE, WEB, PartErrorModel, family_cfg


def model(fam):
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scn = make_batch(family_cfg(cfg, fam), {'clean': 0, 'single': 0, 'mixed': 1, 'hard': 0})[0][1]
    m = PartErrorModel(scn)
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # 아무 배치
    T[:3, 3] = [0.5, 0.2, 0.1]
    m.set_nominal(T)
    return m


def to_base(m, Q):
    T = m.T_place
    return np.asarray(Q, dtype=float) @ T[:3, :3].T + T[:3, 3]


@pytest.mark.parametrize('fam', ['fillet_t', 'fillet_corner'])
def test_fillet_distances(fam):
    m = model(fam)
    g = m.scn['geometry']
    x = 0.5 * m.L
    d = 0.01
    Q = [[x, -d, d],                           # 이음매 위 이등분 방향: 두 판에서 각각 d
         [x, -0.03, 0.05],                     # 바닥 판 위 5 cm, 웹 앞 3 cm
         [x, g['web_thickness'] / 2, 0.05],    # 웹 판 속
         [-0.02, -0.03, -0.005]]               # 이음매 시작 2 cm 앞, 바닥 판 두께 안 높이 (끝면까지 2 cm)
    D = plate_distances(m, to_base(m, Q))
    assert D[0] == pytest.approx([d, d], abs=1e-9)
    assert D[1] == pytest.approx([0.05, 0.03], abs=1e-9)
    assert D[2, WEB] < 0 and D[2, BASE] > 0
    assert D[3, BASE] == pytest.approx(0.02, abs=1e-9)


def test_butt_groove():
    m = model('butt_v')
    g = m.scn['geometry']
    x = 0.5 * m.L
    D = plate_distances(m, to_base(m, [[x, 0.0, g['thickness']], [x, -0.05, g['thickness'] - g['root_face'] + 0.004]]))
    assert np.all(D[0] > 0)                     # 홈 위 가운데는 두 판 밖
    assert D[1, BASE] == pytest.approx(0.004, abs=1e-9) and D[1, WEB] > 0.04


def test_pipe_unwrap_and_distance():
    m = model('pipe_girth')
    g = m.scn['geometry']
    h = g['thickness'] - g['root_face']
    for s in (0.0, 0.3, 0.5, 1.0, 1.4):        # 관 둘레 어디서나 (용접 구간 밖도)
        psi = (s - 0.5) * g['arc']
        r = g['radius'] + h + 0.006
        Q = [[r * np.sin(psi), -0.04, r * np.cos(psi)],      # 왼쪽 관 윗면 6 mm 위
             [0.0, 0.04, 0.0]]                                # 관 축 (속이 빈 관 안)
        P = to_base(m, Q)
        F = flat_coords(m, P[:1])
        assert F[0, 0] == pytest.approx(s * m.L, abs=1e-9) and F[0, 2] == pytest.approx(h + 0.006, abs=1e-9)
        D = plate_distances(m, P)
        assert D[0, BASE] == pytest.approx(0.006, abs=1e-9)
        assert D[1, WEB] > 0.05


@pytest.mark.parametrize('fam', ['fillet_t', 'pipe_girth'])
def test_seam_shift_follows_true_seam(fam):
    """오차가 있는 시나리오: 실제 이음매 바로 위 점은 명목 판 기준으론 어긋나도, 벗어남을 옮기면 제자리 거리가 나온다."""
    m = model(fam)
    for s, s_arc in ((0.2, 0.2), (0.6, 0.9)):
        p, R = m.seam_true(np.array([s]), s_arc)
        d = 0.008
        Q = p[0] + R[0] @ np.array([0.0, 0.0, d])            # 실제 이음매 좌표계 z(이등분·법선) 방향 d
        D = plate_distances(m, Q[None], s_arc=s_arc)[0]
        exp = d / np.sqrt(2) if fam == 'fillet_t' else None   # 필릿: 두 판에 45°
        if exp is not None:
            assert D == pytest.approx([exp, exp], abs=6e-4)
        else:                                                  # V홈 루트 위 d: 두 개선 면까지 d·sin(개선 각)
            bevel = m.scn['geometry']['bevel']
            assert D == pytest.approx([d * np.sin(bevel)] * 2, abs=6e-4)
