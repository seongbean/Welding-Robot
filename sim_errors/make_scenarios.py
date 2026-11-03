#!/usr/bin/env python3
"""모재 오차 시나리오 N개를 config 분포에서 뽑아 JSON으로 저장하고, 정답 크기 분포를 요약한다.

사용법 (저장소 루트에서)
  python -m sim_errors.make_scenarios --n 20 --seed 0 --out scenarios/
  python -m sim_errors.make_scenarios --n 5 --passes 3 --out scenarios/   # 다층 용접: 패스마다 잔류 변형 누적
  python -m sim_errors.make_scenarios --n 500 --summary-only               # 분포만 확인 (실측과 비교)

요약 표는 토치가 진행도 s에 있을 때(s_arc = s)의 정답 크기를 시나리오별로 구한 뒤 백분위로 보여 준다.
실측값이 1~99% 범위 안에 들면 시뮬 분포가 현실을 덮는 것이다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

from .part_errors import next_pass, sample_scenario, save_scenario, summarize

DEFAULT_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'part_errors.yaml')


def main(argv=None):
    for st in (sys.stdout, sys.stderr):   # Windows 콘솔 한글 깨짐 방지
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CFG)
    ap.add_argument('--n', type=int, default=10, help='시나리오(모재) 개수')
    ap.add_argument('--passes', type=int, default=1, help='모재당 패스 수 (2 이상이면 잔류 변형을 다음 패스로 넘김)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='scenarios')
    ap.add_argument('--summary-only', action='store_true', help='파일을 쓰지 않고 분포 요약만')
    args = ap.parse_args(argv)

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
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


if __name__ == '__main__':
    main()
