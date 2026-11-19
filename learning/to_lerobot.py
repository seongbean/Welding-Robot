#!/usr/bin/env python3
"""에피소드 기록(.npz, 100 Hz) → 학습용 프레임 (10 Hz): frames.npz와 LeRobotDataset.

사용법 (저장소 루트에서)
  python -m learning.to_lerobot episodes/t_small --out datasets/t_small
  python -m learning.to_lerobot episodes/t_small --out datasets/t_small --no-lerobot   # frames.npz만
  python -m learning.to_lerobot episodes/t episodes/corner --out datasets/t_corner     # 여러 시편군을 한 데이터로

- 관측·행동 정의는 learning/features.py (관측 61개, 행동 7개).
- 묶음 실행 보고서(<에피소드 폴더>/batch_report.json)가 있으면 점검에 실패한 에피소드는 뺀다.
- 시나리오에 시작 구간 스캔(sensors.prescan, 아크 끄고 시작 30 mm)이 있으면 그 측정으로 단면 버퍼를 먼저 채운다.
  --no-prescan이면 넣지 않는다. 이전 기록은 sim_errors.resense로 지금 센서 설정을 넣어야 스캔이 생긴다.
- 학습·검증·시험 나눔은 시나리오의 episode.split (에피소드 단위). 가혹 묶음은 시험(평가 전용).
- 에피소드 폴더를 여러 개 주면 한 데이터로 합친다 (파일 이름이 겹치면 안 됨: 코너 묶음은 corner_ 머리말).
  시편군(family = 시나리오 geometry.joint)을 에피소드마다 적어 두므로 train_bc·evaluate의 --family로 나눠 쓸 수 있다.
- frames.npz: obs, action, episode_index, frame_index, timestamp + 에피소드별 file, split, kind, mode, perturb, family,
  prescan(시작 구간 스캔을 넣었는지, train_bc가 policy.pt cfg로 옮김).
  학습(learning/train_bc.py)은 이 파일을 읽는다.
- LeRobotDataset (pip install "lerobot[dataset]"이 있으면): <out>/lerobot, repo_id local/<폴더 이름>, fps 10,
  observation.state(61), action(7), task = "<시편군> weld (weave|straight)" (예: T fillet, corner fillet).
  나눔은 <out>/splits.json.
  나중에 ACT·Diffusion 등 LeRobot 정책과 비교할 때 쓴다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

from .features import ACTION_NAMES, FPS, OBS_NAMES, episode_frames, load_episode


def failed_episodes(ep_dir: str) -> dict[str, str]:
    path = os.path.join(ep_dir, 'batch_report.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        rep = json.load(f)
    return {e['file']: e.get('reason', '') for e in rep.get('episodes', []) if not e.get('ok', False)}


TASK_NAMES = {'fillet_t': 'T fillet', 'fillet_corner': 'corner fillet', 'butt_v': 'butt V-groove'}


def convert(ep_dir, out: str, lerobot: bool = True, fps: int = FPS, prescan: bool = True) -> dict:
    """ep_dir: 에피소드 폴더 하나 또는 여러 개 (리스트)."""
    dirs = [ep_dir] if isinstance(ep_dir, str) else list(ep_dir)
    files, bad = [], {}
    for d in dirs:
        files += sorted(f for f in glob.glob(os.path.join(d, '*.npz')) if not f.endswith('.failed.npz'))
        bad.update(failed_episodes(d))
    names = [os.path.basename(f) for f in files]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f'에피소드 파일 이름이 폴더끼리 겹침: {", ".join(dup[:5])}')
    os.makedirs(out, exist_ok=True)
    OB, A, E, F, T = [], [], [], [], []
    info = {'file': [], 'split': [], 'kind': [], 'mode': [], 'perturb': [], 'family': [], 'n': []}
    skipped = []
    used_prescan = False
    for path in files:
        name = os.path.basename(path)
        if name in bad:
            skipped.append((name, bad[name]))
            continue
        rows, scn, meta = load_episode(path)
        if not rows['arc_on'].any() or not bool(rows['done'][-1]):
            skipped.append((name, '용접 구간이 끝나지 않음'))
            continue
        obs, act, ts = episode_frames(rows, scn, fps, prescan)
        used_prescan = used_prescan or (prescan and scn['sensors'].get('prescan', 0.0) > 0)
        k = len(info['file'])
        epi = scn.get('episode', {})
        OB.append(obs)
        A.append(act)
        T.append(ts)
        E.append(np.full(len(obs), k))
        F.append(np.arange(len(obs)))
        info['file'].append(name)
        info['split'].append(epi.get('split', 'train'))
        info['kind'].append(epi.get('kind', ''))
        info['mode'].append(epi.get('mode', 'weave'))
        info['perturb'].append(bool(epi.get('perturb', False)))
        info['family'].append(scn['geometry'].get('joint', 'fillet_t'))
        info['n'].append(len(obs))
    if not OB:
        raise SystemExit(f'변환할 에피소드가 없음: {ep_dir}')
    np.savez_compressed(os.path.join(out, 'frames.npz'), obs=np.concatenate(OB), action=np.concatenate(A),
                        episode_index=np.concatenate(E), frame_index=np.concatenate(F), timestamp=np.concatenate(T),
                        obs_names=np.array(OBS_NAMES), action_names=np.array(ACTION_NAMES),
                        **{f'ep_{k}': np.array(v) for k, v in info.items()}, fps=fps, prescan=used_prescan)
    splits = {s: [i for i, x in enumerate(info['split']) if x == s] for s in ('train', 'val', 'test')}
    with open(os.path.join(out, 'splits.json'), 'w', encoding='utf-8') as f:
        json.dump({'splits': splits, 'files': info['file'], 'kind': info['kind'], 'mode': info['mode'],
                   'family': info['family'], 'skipped': skipped}, f, ensure_ascii=False, indent=1)
    res = {'prescan': used_prescan, 'families': sorted(set(info['family'])), "episodes": len(OB), 'frames': int(sum(info['n'])), 'skipped': skipped, 'splits': splits, 'lerobot': None}
    if lerobot:
        res['lerobot'] = write_lerobot(out, OB, A, info, fps)
    return res


def write_lerobot(out: str, OB, A, info, fps: int) -> str | None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print('lerobot 없음: LeRobotDataset은 건너뜀 (pip install "lerobot[dataset]")')
        return None
    root = os.path.join(out, 'lerobot')
    if os.path.exists(root):
        shutil.rmtree(root)
    features = {'observation.state': {'dtype': 'float32', 'shape': (len(OBS_NAMES),), 'names': OBS_NAMES},
                'action': {'dtype': 'float32', 'shape': (len(ACTION_NAMES),), 'names': ACTION_NAMES}}
    ds = LeRobotDataset.create(repo_id=f'local/{os.path.basename(os.path.normpath(out))}', fps=fps,
                               features=features, root=root, robot_type='opti3', use_videos=False)
    for obs, act, mode, fam in zip(OB, A, info['mode'], info['family']):
        task = f'{TASK_NAMES.get(fam, fam)} weld ({mode})'
        for o, a in zip(obs, act):
            ds.add_frame({'observation.state': o, 'action': a, 'task': task})
        ds.save_episode()
    ds.finalize()
    return root


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episodes', nargs='+', help='에피소드 .npz 폴더 (run_batch.py 출력), 여러 개면 합침')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-lerobot', action='store_true')
    ap.add_argument('--fps', type=int, default=FPS)
    ap.add_argument('--no-prescan', action='store_true',
                    help='용접 전 시작 구간 스캔을 버퍼에 넣지 않음 (스캔 효과 비교용)')
    args = ap.parse_args(argv)
    r = convert(args.episodes, args.out, not args.no_lerobot, args.fps, not args.no_prescan)
    print(f'에피소드 {r["episodes"]}개, 프레임 {r["frames"]}개 ({args.fps} Hz), '
          f'시작 구간 스캔 {"있음" if r["prescan"] else "없음"}, 시편군 {", ".join(r["families"])} '
          f'→ {os.path.abspath(args.out)}')
    print('나눔: ' + ', '.join(f'{k} {len(v)}' for k, v in r['splits'].items()))
    for name, why in r['skipped']:
        print(f'  제외 {name}: {why}')
    if r['lerobot']:
        print(f'LeRobotDataset: {r["lerobot"]}')


if __name__ == '__main__':
    main()
