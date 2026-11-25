#!/usr/bin/env python3
"""팔·토치–모재 간격 점검: Isaac 에피소드 기록의 관절 실측으로 토치와 팔이 모재에 얼마나 가까이 갔는지, 뚫었는지 잰다.

Isaac 장면의 모재는 표시 전용(충돌 없음)이라 토치가 모재를 뚫어도 시뮬레이션이 막지 않고 화면으로만 보인다.
그 확인을 숫자로 한다. 팔·토치 모양은 Isaac 장면의 표시 메시 그대로다(sim_bridge/geometry/opti3_scene_points.npz,
tools/extract_scene_points.py로 posco_torch.usd에서 뽑음. 장면의 링크·tcp·팁 좌표계는 opti3.urdf와 같다).

사용법 (저장소 루트에서, conda activate weld)
  python sim_bridge\\clearance_check.py outputs\\experiments\\20261125-isaac-policy\\runs\\look\\isaac --follower-urdf C:\\...\\opti3.urdf
  (run_batch.py나 policy_to_sim.py가 만든 에피소드 폴더 여럿을 줘도 된다. 결과는 폴더마다 clearance_report.json)

- 부위: 와이어 끝(팁에서 --wire-mm 안의 토치 점), 토치 앞(--nozzle-mm 안, 장면 토치는 와이어가 노즐 밖으로 약 55 mm 나와 있어
  와이어 나머지와 노즐 입구), 토치 뒤(나머지), 손목(link5·6), 팔(link1~4).
- 간격 = 부위 표면 점에서 모재 판까지 부호 거리의 최소 (음수 = 뚫음). 모재는 명목 단면 기둥이고 변형(mm 단위)은 넣지 않는다
  (sim_errors/clearance.py). 와이어 끝 간격은 팁–모재 거리라 1~2 mm가 정상이다 (standoff 2 mm, 필릿은 두 판에 45°).
- 표면 점 간격이 토치 3 mm, 링크 8 mm라 그보다 작은 파고듦은 놓칠 수 있다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, 'weld_teleop')
for p in (REPO, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from sim_errors.clearance import plate_distances  # noqa: E402
from sim_errors.part_errors import PartErrorModel  # noqa: E402

POINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geometry', 'opti3_scene_points.npz')
BODIES = ('wire', 'nozzle', 'torch', 'wrist', 'arm')
BODY_KO = {'wire': '와이어 끝', 'nozzle': '토치 앞', 'torch': '토치 뒤', 'wrist': '손목(link5·6)', 'arm': '팔(link1~4)'}
LINK_BODY = {'link1': 'arm', 'link2': 'arm', 'link3': 'arm', 'link4': 'arm', 'link5': 'wrist', 'link6': 'wrist'}
PLATE_KO = {'fillet_t': ('바닥 판', '웹 판'), 'fillet_corner': ('바닥 판', '웹 판'),
            'butt_v': ('왼쪽 판', '오른쪽 판'), 'pipe_girth': ('왼쪽 관', '오른쪽 관')}


class RobotShape:
    """관절 자세 → 부위별 표면 점 (베이스 좌표). 링크 i 점은 pinocchio 관절 i 좌표계(= URDF link i), 토치는 tcp(= link6)."""

    def __init__(self, kin, points: str = POINTS, wire_mm: float = 15.0, nozzle_mm: float = 100.0):
        import pinocchio as pin
        self.pin, self.kin = pin, kin
        d = np.load(points)
        tip = d['tip'].astype(float)
        torch = d['torch'].astype(float)
        r = np.linalg.norm(torch - tip, axis=1) * 1e3
        self.torch = {'wire': torch[r < wire_mm], 'nozzle': torch[(r >= wire_mm) & (r < nozzle_mm)],
                      'torch': torch[r >= nozzle_mm]}
        names = [kin.model.names[j] for j in range(kin.model.njoints)]
        self.links = []                                    # (관절 번호, 부위, 점)
        for link, body in LINK_BODY.items():
            j = int(link[4:])
            if f'joint{j - 1}' in names and link in d.files:
                self.links.append((names.index(f'joint{j - 1}'), body, d[link].astype(float)))

    def points(self, q) -> dict[str, np.ndarray]:
        pin, kin = self.pin, self.kin
        pin.framesForwardKinematics(kin.model, kin.data, kin._to_pin(np.asarray(q, dtype=float)))
        F = kin.data.oMf[kin.ee_fid]
        out = {b: [P @ F.rotation.T + F.translation] for b, P in self.torch.items()}
        for j, body, P in self.links:
            M = kin.data.oMi[j]
            out.setdefault(body, []).append(P @ M.rotation.T + M.translation)
        return {k: np.concatenate(v) for k, v in out.items()}


def check_episode(path: str, shape: RobotShape, every: int) -> dict:
    d = np.load(path, allow_pickle=False)
    scn = json.loads(str(d['scenario_json']))
    if 'q_meas' not in d.files:
        raise ValueError('관절 실측(q_meas)이 없는 기록 (운동학 에피소드?)')
    model = PartErrorModel(scn)
    fam = scn['geometry'].get('joint', 'fillet_t')
    weld = np.flatnonzero(d['arc_on'].astype(bool))[::every]
    best = {b: (np.inf, 0.0, 0) for b in BODIES}
    for i in weld:
        for b, P in shape.points(d['q_meas'][i]).items():
            D = plate_distances(model, P)
            k = np.unravel_index(np.argmin(D), D.shape)
            if D[k] < best[b][0]:
                best[b] = (float(D[k]), float(d['s'][i]), int(k[1]))
    plates = PLATE_KO.get(fam, ('판 0', '판 1'))
    return {'file': os.path.basename(path), 'family': fam,
            'kind': scn.get('episode', {}).get('kind', ''),
            'min_mm': {b: round(v[0] * 1e3, 2) for b, v in best.items()},
            'at_s': {b: round(v[1], 3) for b, v in best.items()},
            'plate': {b: plates[v[2]] for b, v in best.items()}}


def report(rows: list[dict], near_mm: float) -> None:
    fams = [f for f in PLATE_KO if any(r['family'] == f for r in rows)]
    print(f'\n{"시편군":<14}{"개수":>5}' + ''.join(f'{BODY_KO[b]:>14}' for b in BODIES) + '   (모든 에피소드 중 최소 간격 mm)')
    for fam in fams:
        g = [r for r in rows if r['family'] == fam]
        print(f'{fam:<14}{len(g):>5}' + ''.join(f'{min(r["min_mm"][b] for r in g):>14.1f}' for b in BODIES))
    bad = [(r, b) for r in rows for b in BODIES[1:] if r['min_mm'][b] < 0]
    near = [(r, b) for r in rows for b in BODIES[1:] if 0 <= r['min_mm'][b] < near_mm]
    print(f'\n뚫음(간격 < 0, 와이어 끝 제외): {len(bad)}건, 가까움(0~{near_mm:g} mm): {len(near)}건')
    for r, b in (bad + near)[:20]:
        print(f'  {r["file"]}: {BODY_KO[b]} {r["min_mm"][b]:+.1f} mm, {r["plate"][b]}, s {r["at_s"][b]:.2f}')


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='+', help='Isaac 에피소드 .npz 폴더 (또는 .npz 파일)')
    ap.add_argument('--follower-urdf', required=True)
    ap.add_argument('--follower-config', default=os.path.join(PKG, 'config', 'follower_opti3.yaml'))
    ap.add_argument('--every', type=int, default=10, help='기록 몇 주기마다 볼지 (Isaac 60 Hz면 10 → 6 Hz)')
    ap.add_argument('--points', default=POINTS, help='팔·토치 표면 점 (tools/extract_scene_points.py)')
    ap.add_argument('--wire-mm', type=float, default=15.0, help='팁에서 이 거리 안의 토치 점 = 와이어 끝')
    ap.add_argument('--nozzle-mm', type=float, default=100.0, help='팁에서 이 거리 안의 토치 점 = 노즐')
    ap.add_argument('--near-mm', type=float, default=5.0, help='이보다 가까우면 목록에 보여 줌 (와이어 끝 제외)')
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from expert_to_sim import build_stepper
    kin = build_stepper(args.follower_config, args.follower_urdf).kin
    shape = RobotShape(kin, args.points, args.wire_mm, args.nozzle_mm)
    all_rows = []
    for folder in args.folders:
        files = ([folder] if folder.endswith('.npz') else
                 sorted(f for f in glob.glob(os.path.join(folder, '*.npz')) if not f.endswith('.failed.npz')))
        rows = []
        for k, f in enumerate(files, 1):
            try:
                rows.append(check_episode(f, shape, args.every))
            except (ValueError, KeyError) as e:
                print(f'  건너뜀 {os.path.basename(f)}: {e}')
                continue
            if k % 20 == 0 or k == len(files):
                print(f'[{k}/{len(files)}] {os.path.basename(f)}', flush=True)
        if not folder.endswith('.npz') and rows:
            with open(os.path.join(folder, 'clearance_report.json'), 'w', encoding='utf-8') as fo_:
                json.dump({'shape': {k: v for k, v in vars(args).items() if k not in ('folders',)},
                           'episodes': rows}, fo_, ensure_ascii=False, indent=1)
            print(f'보고서: {os.path.join(os.path.abspath(folder), "clearance_report.json")}')
        all_rows += rows
    if all_rows:
        report(all_rows, args.near_mm)
    return 0 if all(r['min_mm'][b] >= 0 for r in all_rows for b in BODIES[1:]) else 1


if __name__ == '__main__':
    sys.exit(main() or 0)
