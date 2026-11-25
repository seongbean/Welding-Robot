"""모재까지의 부호 거리 (numpy만 사용). Isaac 장면의 모재는 표시 전용이라 충돌이 없어서, 팔·토치가 모재를 뚫어도
시뮬레이션이 막지 않는다. 그 확인을 숫자로 하려고 쓴다 (sim_bridge/clearance_check.py).

- 모재 판 하나 = 명목 단면 볼록 다각형(PartErrorModel.sections)을 이음매 방향으로 늘인 기둥. 파이프는 관 둘레로 감은 것이라
  점을 관 축 둘레 각·반지름으로 풀어(평평한 좌표) 같은 식으로 잰다.
- 변형: s_arc를 주면 점마다 가까운 이음매 진행도에서 실제 이음매가 명목 이음매에서 벗어난 만큼(좌우·높이·밀림 등, mm 단위)
  판을 평행 이동해 잰다. 판의 기울어짐(각변형·웹 기울기)은 넣지 않아, 이음매에서 멀어질수록 그만큼(10 cm에 1°면 1.7 mm) 오차가 있다.
- 부호: 바깥 양수, 안쪽 음수 [m].
"""
from __future__ import annotations

import numpy as np

from .part_errors import PartErrorModel


def _polygon_sd(Q: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """2D 점 (n,2)에서 볼록 다각형 (m,2)까지의 부호 거리."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a                                                    # (m,2)
    ap = Q[:, None, :] - a[None]                                  # (n,m,2)
    t = np.clip(np.einsum('nmk,mk->nm', ap, ab) / np.einsum('mk,mk->m', ab, ab), 0.0, 1.0)
    d = np.linalg.norm(ap - t[..., None] * ab[None], axis=-1).min(axis=1)
    cross = ab[None, :, 0] * ap[..., 1] - ab[None, :, 1] * ap[..., 0]
    inside = np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)
    return np.where(inside, -d, d)


def flat_coords(model: PartErrorModel, P: np.ndarray) -> np.ndarray:
    """로봇 베이스 기준 점 (n,3) → 모재의 평평한 좌표 (x = 이음매 방향, y = 단면 좌우, z = 단면 높이 / 반지름 방향)."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    T = np.asarray(model.T_place)
    Q = (P - T[:3, 3]) @ T[:3, :3]                                # T⁻¹ P
    if not model.pipe:
        return Q
    g = model.scn['geometry']
    psi = np.arctan2(Q[:, 0], Q[:, 2])                            # 꼭대기 = 0, 진행 방향 +
    r = np.hypot(Q[:, 0], Q[:, 2])
    return np.stack([(psi / g['arc'] + 0.5) * model.L, Q[:, 1], r - g['radius']], axis=-1)


def seam_shift(model: PartErrorModel, P: np.ndarray, s_arc: float, n: int = 201) -> np.ndarray:
    """점마다 (가까운 진행도의 실제 이음매 − 명목 이음매) [m, 베이스 좌표]. 진행도는 평평한 좌표 x로 정하고 [0, 1]로 자른다."""
    s = np.clip(flat_coords(model, P)[:, 0] / model.L, 0.0, 1.0)
    grid = np.linspace(0.0, 1.0, n)
    T = np.asarray(model.T_place)
    d = model.seam_true(grid, s_arc)[0] - (model.nominal(grid)[0] @ T[:3, :3].T + T[:3, 3])
    return np.stack([np.interp(s, grid, d[:, k]) for k in range(3)], axis=-1)


def plate_distances(model: PartErrorModel, P: np.ndarray, s_arc: float | None = None) -> np.ndarray:
    """점 (n,3)에서 판마다의 부호 거리 (n, 판 수) [m]. 열 순서는 PartErrorModel.sections() 순서 (판 번호).
    s_arc를 주면 그 아크 진행도의 이음매 벗어남만큼 판을 옮겨 잰다 (seam_shift)."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    if s_arc is not None:
        P = P - seam_shift(model, P, s_arc)
    F = flat_coords(model, P)
    sections, (x0, x1) = model.sections()
    out = []
    for _, poly in sections:
        d2 = _polygon_sd(F[:, 1:], np.asarray(poly, dtype=float))
        if model.pipe:                                            # 관 둘레 전체: 이음매 방향 끝이 없다
            out.append(d2)
            continue
        dx = np.maximum(x0 - F[:, 0], F[:, 0] - x1)
        inside = (d2 <= 0) & (dx <= 0)
        out.append(np.where(inside, np.maximum(d2, dx), np.hypot(np.maximum(d2, 0), np.maximum(dx, 0))))
    return np.stack(out, axis=-1)
