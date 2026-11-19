#!/usr/bin/env python3
"""모재 오차 시나리오 N개를 config 분포에서 뽑아 JSON으로 저장하고, 정답 크기 분포를 요약한다.

사용법 (저장소 루트에서)
  python -m sim_errors.make_scenarios --n 20 --seed 0 --out scenarios/
  python -m sim_errors.make_scenarios --n 5 --passes 3 --out scenarios/   # 다층 용접: 패스마다 잔류 변형 누적
  python -m sim_errors.make_scenarios --n 500 --summary-only               # 분포만 확인 (실측과 비교)
  python -m sim_errors.make_scenarios --single all --out scenarios_single/ # 오차별 단독 검증용 (하나만 과장해서 켬)
  python -m sim_errors.make_scenarios --batch --out batches/t_fillet/       # 학습용 묶음 (깨끗함·단독·혼합·가혹)
  python -m sim_errors.make_scenarios --batch --scale 0.2 --out batches/t_small/   # 수량 1/5로 먼저 한 바퀴
  python -m sim_errors.make_scenarios --batch --family fillet_corner --out batches/corner/   # ㄱ자 코너 필릿
  python -m sim_errors.make_scenarios --batch --family butt_v --out batches/butt/           # 맞대기 V홈

--family는 config families 섹션의 시편군을 고른다 (기본 = geometry.joint, 지금은 fillet_t).
T 필릿이 아닌 시편군의 묶음 파일 이름에는 군 머리말(corner_, butt_)이 붙어, 여러 군 에피소드를 한 폴더에 섞어도
이름이 겹치지 않는다. 같은 --seed라도 시편군마다 다른 오차를 뽑는다.

학습용 묶음(--batch)은 config episode 섹션의 개수로 시나리오를 만들고, 각 시나리오에 episode 설정
(강도, 직선/위빙 모드, 시작 어긋남, 실행 섭동, 학습·검증·시험 나눔)을 붙인다. 모두 용접 구간만 담는다.

요약 표는 토치가 진행도 s에 있을 때(s_arc = s)의 정답 크기를 시나리오별로 구한 뒤 백분위로 보여 준다.
실측값이 1~99% 범위 안에 들면 시뮬 분포가 현실을 덮는 것이다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

from .part_errors import (JOINTS, KINDS, SINGLE_ERRORS, episode_meta, family_cfg, next_pass, sample_scenario,
                          save_scenario, scale_cfg, signature, single_error, single_in_range, single_names, summarize, zero_errors)

# 시편군 → 묶음 파일 머리말, 난수 열쇠 (T 필릿은 예전과 같은 이름·난수를 유지)
FAMILY_PREFIX = {'fillet_t': '', 'fillet_corner': 'corner_', 'butt_v': 'butt_'}
FAMILY_KEY = {'fillet_t': (), 'fillet_corner': (1,), 'butt_v': (2,)}

DEFAULT_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'part_errors.yaml')


def main(argv=None):
    for st in (sys.stdout, sys.stderr):   # Windows 콘솔 한글 깨짐 방지
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CFG)
    ap.add_argument('--family', choices=JOINTS, help='시편군 (기본: config geometry.joint)')
    ap.add_argument('--n', type=int, default=10, help='시나리오(모재) 개수')
    ap.add_argument('--passes', type=int, default=1, help='모재당 패스 수 (2 이상이면 잔류 변형을 다음 패스로 넘김)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='scenarios')
    ap.add_argument('--summary-only', action='store_true', help='파일을 쓰지 않고 분포 요약만')
    ap.add_argument('--single', help=f'단독 검증 시나리오: all 또는 쉼표로 ({", ".join(SINGLE_ERRORS)})')
    ap.add_argument('--batch', action='store_true', help='학습용 묶음 (config episode 섹션)')
    ap.add_argument('--scale', type=float, default=1.0, help='--batch 개수 배율 (예: 0.2 = 1/5)')
    ap.add_argument('--counts', help='--batch 개수 직접 지정 clean,single,mixed,hard (예: 4,13,20,4)')
    args = ap.parse_args(argv)

    with open(args.config, encoding='utf-8') as f:
        cfg = family_cfg(yaml.safe_load(f), args.family)
    if args.single:
        return write_single(cfg, args)
    if args.batch:
        return write_batch(cfg, args)
    if not args.summary_only:
        os.makedirs(args.out, exist_ok=True)

    rows = []
    for i in range(args.n):
        rng = np.random.default_rng([args.seed, i])
        scn = sample_scenario(cfg, rng, seed=args.seed * 100000 + i)
        for p in range(args.passes):
            if p:
                scn = next_pass(scn, cfg, rng)
            rows.append(summarize(scn))
            if not args.summary_only:
                save_scenario(scn, os.path.join(args.out, f'part{i:04d}_pass{p}.json'))

    keys = list(rows[0])
    print(f'시나리오 {len(rows)}개 (모재 {args.n} × 패스 {args.passes}) 정답 크기 백분위')
    print(f'{"항목":<24}{"1%":>8}{"50%":>8}{"99%":>8}{"최대":>8}')
    for k in keys:
        v = np.array([r[k] for r in rows])
        print(f'{k:<24}' + ''.join(f'{x:8.2f}' for x in np.percentile(v, [1, 50, 99]).tolist() + [v.max()]))
    if not args.summary_only:
        print(f'저장: {os.path.abspath(args.out)}')


def _pick(i: int, ratio: float) -> bool:
    """i번째가 비율 ratio 쪽인지 (번갈아 고르게: 0.5면 짝수 번째)."""
    return int((i + 1) * ratio) > int(i * ratio)


def _splits(n: int, ratio, rng: np.random.Generator) -> list[str]:
    """n개를 학습·검증·시험으로 나눔 (에피소드 단위, 섞어서). 5개 이상이면 검증·시험을 적어도 1개씩."""
    n_val = int(round(n * ratio[1]))
    n_test = int(round(n * ratio[2]))
    if n >= 5:
        n_val, n_test = max(1, n_val), max(1, n_test)
    lab = ['val'] * n_val + ['test'] * n_test
    lab = ['train'] * (n - len(lab)) + lab
    return [lab[k] for k in rng.permutation(n)]


def make_batch(cfg: dict, counts: dict, seed: int = 0) -> list[tuple[str, dict]]:
    """학습용 묶음 [(파일 이름, 시나리오)]."""
    ep = cfg['episode']
    hard_cfg = scale_cfg(cfg, ep['hard_scale'])
    names = single_names(cfg)
    joint = cfg['geometry'].get('joint', 'fillet_t')
    fk, prefix = FAMILY_KEY[joint], FAMILY_PREFIX[joint]
    out = []
    for k, kind in enumerate(KINDS):
        n = int(counts.get(kind, 0))
        splits = (['test'] * n if kind == 'hard'
                  else _splits(n, ep['split'], np.random.default_rng([seed, k, 99, *fk])))
        for i in range(n):
            rng = np.random.default_rng([seed, k, i, *fk])
            sid = seed * 100000 + k * 10000 + i
            if kind == 'clean':
                src = sample_scenario(cfg, rng, seed=sid)
                scn = zero_errors(src)
                scn['sensors'] = src['sensors']
            elif kind == 'single':
                scn = single_in_range(cfg, names[i % len(names)], rng, seed=sid)
            elif kind == 'mixed':
                scn = sample_scenario(cfg, rng, seed=sid)
            else:
                scn = sample_scenario(hard_cfg, rng, seed=sid)
            mode = 'weave' if _pick(i, ep['weave_ratio']) else 'straight'
            perturb = kind == 'mixed' and _pick(i // 2, ep['perturb_ratio'])
            scn['episode'] = episode_meta(kind, mode, rng, ep, perturb, splits[i])
            out.append((f'{prefix}{kind}_{i:04d}.json', scn))
    return out


def write_batch(cfg: dict, args) -> None:
    if args.counts:
        counts = dict(zip(KINDS, (int(x) for x in args.counts.split(','))))
    else:
        counts = {k: max(1, int(round(v * args.scale))) for k, v in cfg['episode']['counts'].items()}
    batch = make_batch(cfg, counts, args.seed)
    os.makedirs(args.out, exist_ok=True)
    for name, scn in batch:
        save_scenario(scn, os.path.join(args.out, name))
    print(f'학습용 묶음 {len(batch)}개 ({cfg["geometry"]["joint"]}) → {os.path.abspath(args.out)}')
    print(f'{"강도":<8}{"개수":>6}{"위빙":>6}{"직선":>6}{"섭동":>6}{"학습":>6}{"검증":>6}{"시험":>6}')
    for kind in KINDS:
        ep = [s['episode'] for _, s in batch if s['episode']['kind'] == kind]
        if not ep:
            continue
        c = {key: sum(1 for e in ep if pred(e)) for key, pred in (
            ('weave', lambda e: e['mode'] == 'weave'), ('straight', lambda e: e['mode'] == 'straight'),
            ('perturb', lambda e: e['perturb']), ('train', lambda e: e['split'] == 'train'),
            ('val', lambda e: e['split'] == 'val'), ('test', lambda e: e['split'] == 'test'))}
        print(f'{kind:<8}{len(ep):>6}' + ''.join(f'{c[k]:>6}' for k in ('weave', 'straight', 'perturb', 'train', 'val', 'test')))


def write_single(cfg: dict, args) -> None:
    """오차 하나만 과장해서 켠 시나리오를 쓰고, 토치 위치(s_arc = s)의 기대 정답을 표로 보여 준다."""
    names = single_names(cfg) if args.single == 'all' else args.single.split(',')
    base = sample_scenario(cfg, np.random.default_rng([args.seed, 0]), seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    print('기대 정답 (토치가 s에 있을 때): 좌우 e_y / 높이 e_z [mm], 작업각 / 진행각 [deg], 틈새 / 가접 [mm]')
    for name in names:
        scn = single_error(base, name)
        path = os.path.join(args.out, f'single_{name}.json')
        save_scenario(scn, path)
        print(f'\n{name} ({SINGLE_ERRORS[name][0]}) → {path}')
        print(f'  {"s":>5}{"e_y":>8}{"e_z":>8}{"작업각":>8}{"진행각":>8}{"틈새":>7}{"가접":>7}')
        for r in signature(scn):
            print(f'  {r["s"]:5.2f}{r["e_y"]:8.2f}{r["e_z"]:8.2f}{r["work"]:8.2f}{r["travel"]:8.2f}{r["gap"]:7.2f}{r["tack"]:7.2f}')


if __name__ == '__main__':
    main()
