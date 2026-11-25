#!/usr/bin/env python3
"""Isaac 장면(assets/isaac/opti3_welding_scene/posco_torch.usd)의 opti3 링크·토치 표시 메시를 링크 좌표계 점 구름으로 뽑는다.

sim_bridge/clearance_check.py가 팔·토치–모재 간격을 잴 때 쓰는 sim_bridge/geometry/opti3_scene_points.npz를 만든다.
USD 장면의 링크 좌표계는 opti3.urdf의 관절 좌표계와 같고(관절 0에서 원점·방향 일치), 토치는 link6/tcp 아래에 있다.
점은 메시 면 위에 고르게 뿌린 점을 격자(--voxel, 토치는 --voxel-torch)마다 하나씩 남겨 솎은 것이다.

사용법 (pxr 필요: pip install usd-core, 또는 C:\\isaacsim\\python.bat)
  python tools/extract_scene_points.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(REPO, 'assets', 'isaac', 'opti3_welding_scene', 'posco_torch.usd')
OUT = os.path.join(REPO, 'sim_bridge', 'geometry', 'opti3_scene_points.npz')


def voxel_down(P: np.ndarray, size: float) -> np.ndarray:
    """격자 칸마다 실제 꼭짓점 하나 (평균을 내면 표면 안쪽으로 들어가 간격이 커 보인다)."""
    key = np.floor(P / size).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return P[np.sort(first)]


def surface_points(pts: np.ndarray, counts, idx, step: float) -> np.ndarray:
    """메시 면 위에 간격 step 정도로 고르게 점을 뿌린다 (꼭짓점만 쓰면 긴 원통 면 가운데가 비어 간격이 커 보인다)."""
    tris, k = [], 0
    for c in counts:                                   # 다각형 면 → 부채꼴 삼각형
        f = idx[k:k + c]
        k += c
        tris += [(f[0], f[j], f[j + 1]) for j in range(1, c - 1)]
    T = pts[np.array(tris)]                            # (m,3,3)
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    n = np.ceil(area / (0.5 * step * step)).astype(int)
    rng = np.random.default_rng(0)
    tri = np.repeat(np.arange(len(T)), n)
    r1, r2 = rng.random(len(tri)), rng.random(len(tri))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    A, B, C = T[tri, 0], T[tri, 1], T[tri, 2]
    return np.concatenate([pts, A + r1[:, None] * (B - A) + r2[:, None] * (C - A)])


def main():
    from pxr import Usd, UsdGeom
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--usd', default=SCENE)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--voxel', type=float, default=0.008, help='링크 점 간격 [m]')
    ap.add_argument('--voxel-torch', type=float, default=0.003, help='토치 점 간격 [m]')
    args = ap.parse_args()
    st = Usd.Stage.Open(args.usd)
    cache = UsdGeom.XformCache()
    world = lambda p: np.array(cache.GetLocalToWorldTransform(p)).T  # noqa: E731   USD는 행벡터 규약
    out = {}
    for prim in st.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or '/collisions/' in path or not path.startswith('/World/indy/link'):
            continue
        link = path.split('/')[3]                                   # link0 … link6
        name = 'torch' if '/tcp/' in path else link
        mesh = UsdGeom.Mesh(prim)
        size = args.voxel_torch if name == 'torch' else args.voxel
        pts = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        T = np.linalg.inv(world(st.GetPrimAtPath(f'/World/indy/{link}'))) @ world(prim)   # 메시 → 링크 좌표계
        P = surface_points(pts @ T[:3, :3].T + T[:3, 3], np.array(mesh.GetFaceVertexCountsAttr().Get()),
                           np.array(mesh.GetFaceVertexIndicesAttr().Get()), size)
        out[name] = voxel_down(P, size).astype(np.float32)
        print(f'{path}: {len(pts)} 점 → {len(out[name])}')
    tip = st.GetPrimAtPath('/World/indy/link6/tcp/weld_tip')
    out['tip'] = (np.linalg.inv(world(st.GetPrimAtPath('/World/indy/link6'))) @ world(tip))[:3, 3].astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f'저장: {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB), 팁 {out["tip"]}')


if __name__ == '__main__':
    main()
