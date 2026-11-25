#!/usr/bin/env python3
"""이미 만든 운동학 에피소드로 실험 워크플로(.github/workflows/experiment.yml)의 변환 → 학습 → 평가 → 표를 한 번에 돌린다.

GitHub Actions를 못 쓸 때 PC(Windows 포함)에서 쓰려고 만들었다. 단계마다 결과가 있으면 건너뛰므로 중단 후 같은 명령으로 이어진다.

사용법 (저장소 루트에서)
  python -m learning.run_experiment --episodes exp/episodes --out exp --jobs 8
  python -m learning.run_experiment --exp 20261123-kin-4fam --jobs 8   # outputs/experiments/<ID>/episodes → 같은 폴더

- 변환: exp/data (episodes/batch_report.json에서 점검 실패한 에피소드는 뺌)
- 학습: mlp_all(네 군), mlp_3(T+코너+V홈), mlp_pipe(파이프만), mlp_all_q(네 군, 학습 수를 mlp_3과 같게), gru_all
- 평가: 위 다섯 + 규칙, 네 군 시험 묶음 닫힌 루프 → exp/runs/<정책>/eval.json
- 표: exp/report.md (시편군별 전체·가혹·혼합)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys

from common import paths

FAMILIES = [('pipe_girth', '파이프 원주'), ('fillet_t', 'T 필릿'), ('fillet_corner', '코너 필릿'), ('butt_v', '맞대기 V홈')]
TRAIN = [('mlp_all', ['--model', 'mlp']),
         ('mlp_3', ['--model', 'mlp', '--family', 'fillet_t,fillet_corner,butt_v']),
         ('mlp_pipe', ['--model', 'mlp', '--family', 'pipe_girth']),
         ('mlp_all_q', None),                      # 학습 수는 mlp_3에서 정함
         ('gru_all', ['--model', 'gru'])]


def run(args: list[str]) -> None:
    print('>', ' '.join(args), flush=True)
    subprocess.run([sys.executable, '-m', *args], check=True)


def main():
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--episodes', help='에피소드 .npz 폴더 (batch_report.json 포함). --exp면 기본 <실험>/episodes')
    ap.add_argument('--out', help='결과 폴더 (data, runs, report.md). --exp면 기본 <실험>')
    ap.add_argument('--jobs', type=int, default=4, help='평가 병렬 프로세스 수 (CPU 코어 수 정도)')
    paths.add_exp_arg(ap)
    a = ap.parse_args()
    if a.exp:
        a.episodes = a.episodes or str(paths.exp_dir(a.exp, 'episodes'))
        a.out = a.out or str(paths.exp_dir(a.exp))
    if not a.episodes or not a.out:
        ap.error('--episodes와 --out이 필요합니다 (또는 --exp)')
    exp = ['--exp', a.exp] if a.exp else []   # 하위 단계도 같은 manifest에 기록 (경로는 아래에서 직접 줌)
    paths.record_stage(a.exp, 'run_experiment', episodes=a.episodes, out=a.out, policies=[t[0] for t in TRAIN])
    data, runs = os.path.join(a.out, 'data'), os.path.join(a.out, 'runs')
    if not os.path.isdir(data):
        run(['learning.to_lerobot', a.episodes, '--out', data, '--no-lerobot', *exp])
    for name, extra in TRAIN:
        out = os.path.join(runs, name)
        if os.path.isfile(os.path.join(out, 'train_log.json')):
            continue
        if extra is None:
            with open(os.path.join(runs, 'mlp_3', 'train_log.json'), encoding='utf-8') as f:
                extra = ['--model', 'mlp', '--max-train', str(json.load(f)['train_episodes'])]
        run(['learning.train_bc', data, *extra, '--out', out, *exp])
    for name, _ in TRAIN:
        if not os.path.isfile(os.path.join(runs, name, 'eval.json')):
            run(['learning.evaluate', os.path.join(runs, name, 'policy.pt'), '--data', data,
                 '--episodes', a.episodes, '--jobs', str(a.jobs), *exp])
    if not os.path.isfile(os.path.join(runs, 'rule', 'eval.json')):
        run(['learning.evaluate', 'rule', '--data', data, '--episodes', a.episodes, '--jobs', str(a.jobs),
             '--rule-dir', runs, *exp])
    from learning.report_evals import table
    evals = sorted(os.path.join(runs, n, 'eval.json') for n in [t[0] for t in TRAIN] + ['rule'])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for fam, title in FAMILIES:
            for kind, label in (('', '시험 (전체)'), ('_hard', '가혹 묶음'), ('_mixed', '혼합 묶음')):
                print(f'## {title} {label}')
                print(table(evals, f'fam_{fam}{kind}'))
                print()
    path = os.path.join(a.out, 'report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(buf.getvalue())
    print(buf.getvalue())
    print(f'표: {path}')


if __name__ == '__main__':
    main()
