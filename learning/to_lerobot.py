#!/usr/bin/env python3
"""에피소드 기록(.npz, 100 Hz) → 학습용 프레임 (10 Hz): frames.npz와 LeRobotDataset.

사용법 (저장소 루트에서)
  python -m learning.to_lerobot episodes/t_small --out datasets/t_small
  python -m learning.to_lerobot episodes/t_small --out datasets/t_small --no-lerobot   # frames.npz만

- 관측·행동 정의는 learning/features.py (관측 61개, 행동 7개).
- 묶음 실행 보고서(<에피소드 폴더>/batch_report.json)가 있으면 점검에 실패한 에피소드는 뺀다.
- 학습·검증·시험 나눔은 시나리오의 episode.split (에피소드 단위). 가혹 묶음은 시험(평가 전용).
- frames.npz: obs, action, episode_index, frame_index, timestamp + 에피소드별 file, split, kind, mode.
  학습(learning/train_bc.py)은 이 파일을 읽는다.
- LeRobotDataset (pip install "lerobot[dataset]"이 있으면): <out>/lerobot, repo_id local/<폴더 이름>, fps 10,
  observation.state(61), action(7), task = "T fillet weld (weave|straight)". 나눔은 <out>/splits.json.
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


def convert(ep_dir: str, out: str, lerobot: bool = True, fps: int = FPS) -> dict:
    files = sorted(f for f in glob.glob(os.path.join(ep_dir, '*.npz')) if not f.endswith('.failed.npz'))
    bad = failed_episodes(ep_dir)
    os.makedirs(out, exist_ok=True)
    OB, A, E, F, T = [], [], [], [], []
    info = {'file': [], 'split': [], 'kind': [], 'mode': [], 'n': []}
    skipped = []
    for path in files:
        name = os.path.basename(path)
        if name in bad:
            skipped.append((name, bad[name]))
            continue
        rows, scn, meta = load_episode(path)
        if not rows['arc_on'].any() or not bool(rows['done'][-1]):
            skipped.append((name, '용접 구간이 끝나지 않음'))
            continue
        obs, act, ts = episode_frames(rows, scn, fps)
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
        info['n'].append(len(obs))
    if not OB:
        raise SystemExit(f'변환할 에피소드가 없음: {ep_dir}')
    np.savez_compressed(os.path.join(out, 'frames.npz'), obs=np.concatenate(OB), action=np.concatenate(A),
                        episode_index=np.concatenate(E), frame_index=np.concatenate(F), timestamp=np.concatenate(T),
                        obs_names=np.array(OBS_NAMES), action_names=np.array(ACTION_NAMES),
                        **{f'ep_{k}': np.array(v) for k, v in info.items()}, fps=fps)
    splits = {s: [i for i, x in enumerate(info['split']) if x == s] for s in ('train', 'val', 'test')}
    with open(os.path.join(out, 'splits.json'), 'w', encoding='utf-8') as f:
        json.dump({'splits': splits, 'files': info['file'], 'kind': info['kind'], 'mode': info['mode'],
                   'skipped': skipped}, f, ensure_ascii=False, indent=1)
    res = {"episodes": len(OB), 'frames': int(sum(info['n'])), 'skipped': skipped, 'splits': splits, 'lerobot': None}
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
    for obs, act, mode in zip(OB, A, info['mode']):
        for o, a in zip(obs, act):
            ds.add_frame({'observation.state': o, 'action': a, 'task': f'T fillet weld ({mode})'})
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
    ap.add_argument('episodes', help='에피소드 .npz 폴더 (run_batch.py 출력)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-lerobot', action='store_true')
    ap.add_argument('--fps', type=int, default=FPS)
    args = ap.parse_args(argv)
    r = convert(args.episodes, args.out, not args.no_lerobot, args.fps)
    print(f'에피소드 {r["episodes"]}개, 프레임 {r["frames"]}개 ({args.fps} Hz) → {os.path.abspath(args.out)}')
    print('나눔: ' + ', '.join(f'{k} {len(v)}' for k, v in r['splits'].items()))
    for name, why in r['skipped']:
        print(f'  제외 {name}: {why}')
    if r['lerobot']:
        print(f'LeRobotDataset: {r["lerobot"]}')


if __name__ == '__main__':
    main()
