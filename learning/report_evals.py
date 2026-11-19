#!/usr/bin/env python3
"""여러 eval.json(learning.evaluate 결과)을 마크다운 표 하나로 모은다. GitHub 실험 워크플로 요약에 쓴다.

사용법 (저장소 루트에서)
  python -m learning.report_evals runs/*/eval.json
  python -m learning.report_evals runs/*/eval.json --kind hard
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

AXES = ('lateral', 'height', 'work', 'travel')


def fmt_gain(v: dict, n: str) -> str:
    g = v.get(n + '_gain', float('nan'))
    return f'{v[n]:.2f} ({g * 100:.0f}%)' if math.isfinite(g) else f'{v[n]:.2f} (-)'


def table(paths: list[str], kind: str = 'all') -> str:
    lines = ['| 정책 | 스캔 | 좌우 mm | 높이 mm | 작업각 deg | 진행각 deg | 높이 시작 30 mm | 높이 나머지 | 위빙 차 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    base = None
    for p in paths:
        with open(p, encoding='utf-8') as f:
            res = json.load(f)
        v = res.get('closed_loop', {}).get(kind)
        if not v:
            continue
        base = base or v
        name = os.path.basename(os.path.dirname(os.path.abspath(p)))
        lines.append(f'| {name} ({res["model"]}) | {"O" if res.get("prescan") else "X"} | '
                     + ' | '.join(fmt_gain(v, n) for n in AXES)
                     + f' | {v.get("height_start", float("nan")):.2f} | {v.get("height_rest", float("nan")):.2f}'
                     + f' | {v["weave_amp_err"]:.2f} |')
    if base:
        lines.append('')
        lines.append(f'시험 {base["n"]}개 ({kind}), 닫힌 루프 RMS (보정 0 대비 개선율). 보정 0: '
                     + ', '.join(f'{n} {base[n + "_base"]:.2f}' for n in AXES))
    return '\n'.join(lines)


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('evals', nargs='+')
    ap.add_argument('--kind', default='all', help='강도 (all, mixed, single, clean, hard)')
    args = ap.parse_args(argv)
    print(table(sorted(args.evals), args.kind))


if __name__ == '__main__':
    main()
