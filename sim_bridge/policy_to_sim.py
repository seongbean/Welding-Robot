#!/usr/bin/env python3
"""학습 정책 → Isaac Sim opti3: 규칙 전문가 대신 학습한 보정 정책이 매 주기 토치를 몰고, 닫힌 루프 오차를 잰다.

expert_to_sim.run_episode를 그대로 쓰고 토치 목표를 내는 부분만 정책(learning/policy_driver.py)으로 바꾼다.
모재 자동 배치, 이음매 IK 도달 확인, 시작 자세 바로 놓기, lockstep, 기록은 전문가 실행(run_batch.py)과 같다.
운동학 평가(learning.evaluate)는 로봇이 명령을 바로 따라간다고 가정했다. 여기서는 정책 관측의 토치 자세가 Isaac
실측이고, 오차도 실측 토치로 잰다.

사용법 (저장소 루트에서, conda activate weld. 정책을 불러오려면 torch 필요)
  터미널 1: C:\\isaacsim\\python.bat sim_bridge\\isaac_opti3_receiver.py --lockstep [--headless]
  터미널 2:
    set K=outputs\\experiments\\20251123-kin-4fam
    python sim_bridge\\policy_to_sim.py --policy %K%\\runs\\mlp_all\\policy.pt --scenarios %K%\\episodes ^
        --kin-eval %K%\\runs\\mlp_all\\eval.json --exp 20251125-isaac-policy --follower-urdf C:\\...\\opti3.urdf

- 시나리오: --scenarios 폴더의 *.json(make_scenarios) 또는 에피소드 *.npz(안의 시나리오). --split(기본 test)만 쓴다.
  운동학 실험의 시험 에피소드 폴더를 주면 운동학 평가와 같은 시나리오로 비교된다.
  --only(이름에 든 문자열), --per-family N(시편군마다 앞에서 N개), --limit N으로 줄인다.
- 정책: policy.pt 경로, 또는 학습 없는 규칙 기준선 'rule' / 'rule-noprescan'.
- 출력 (--out, 기본: --exp면 <실험>/runs/<--run>/isaac, 아니면 outputs/scratch/policy_isaac/<--run>):
  에피소드 .npz (끝나지 않으면 .failed.npz), receiver가 읽는 시나리오 사본 scenarios/*.json, 오차 요약 eval_isaac.json.
  --run 기본값은 정책 폴더 이름(예: mlp_all) 또는 rule. 이미 있는 에피소드는 건너뛴다 (--overwrite로 다시).
- 오차: 측정(실측 토치 − 실제 이음매, 위빙 뺌)과 명령(정책 보정 − 실제 이음매, 운동학 평가와 같은 정의).
  --kin-eval(운동학 evaluate의 eval.json)을 주면 같은 에피소드의 운동학 오차를 옆에 놓는다.
- --report-only: Isaac 없이 저장된 에피소드로 요약만 다시 낸다.
- 화면 확인: receiver를 --headless 없이 띄우고 --only pipe_clean --limit 1처럼 하나씩 돌린다.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from common import paths  # noqa: E402
from learning.evaluate import RULES, summarize  # noqa: E402
from learning.features import load_episode  # noqa: E402
from learning.policy_driver import AXES, PolicyDriver, as_cmd, episode_metrics  # noqa: E402
from sim_errors.part_errors import load_scenario, save_scenario  # noqa: E402

HOME_Q = '0,-0.5236,1.7453,0,0.8727,0'    # isaac_opti3_receiver.py --initial-q 기본값
FAMILIES = ('fillet_t', 'fillet_corner', 'butt_v', 'pipe_girth')
FAMILY_KO = {'fillet_t': 'T 필릿', 'fillet_corner': '코너 필릿', 'butt_v': '맞대기 V홈', 'pipe_girth': '파이프 원주'}


def load_policy(path: str):
    from learning.evaluate import load_policy as _load
    return _load(path)


def read_scenario(path: str) -> dict:
    """시나리오 .json 또는 에피소드 .npz 안의 시나리오."""
    if path.endswith('.npz'):
        with np.load(path, allow_pickle=False) as d:      # 시나리오만 읽는다 (기록 배열은 안 풂)
            return json.loads(str(d['scenario_json']))
    return load_scenario(path)


def family(scn: dict) -> str:
    return scn['geometry'].get('joint', 'fillet_t')


def select(folder: str, split: str | None, only: str | None, per_family: int | None, limit: int | None
           ) -> list[tuple[str, dict]]:
    """(이름, 시나리오) 목록. .failed.npz는 뺀다. 이름 = 확장자 없는 파일 이름."""
    files = sorted(f for f in glob.glob(os.path.join(folder, '*.json')) + glob.glob(os.path.join(folder, '*.npz'))
                   if not f.endswith('.failed.npz'))
    if only:
        files = [f for f in files if only in os.path.basename(f)]
    out, count = [], {}
    for f in files:
        try:
            scn = read_scenario(f)
        except (ValueError, KeyError, OSError):      # batch_report.json 같은 다른 파일
            continue
        if split and split != 'all' and scn.get('episode', {}).get('split') != split:
            continue
        fam = family(scn)
        if per_family and count.get(fam, 0) >= per_family:
            continue
        count[fam] = count.get(fam, 0) + 1
        out.append((os.path.splitext(os.path.basename(f))[0], scn))
        if limit and len(out) >= limit:
            break
    return out


def sensor_seed(name: str) -> int:
    """단면 센서 노이즈 시드: 이름마다 고정 (고르는 조건이 바뀌어도 같은 값)."""
    return zlib.crc32(name.encode()) % 100000


def kin_rows(path: str | None) -> dict:
    """운동학 eval.json의 에피소드별 닫힌 루프 행 (이름 → 행)."""
    if not path:
        return {}
    with open(path, encoding='utf-8') as f:
        res = json.load(f)
    return {os.path.splitext(r['file'])[0]: r for r in res.get('closed_loop_episodes', [])}


def load_skipped(out: str) -> dict:
    """도달·배치 실패로 시작하지 못한 시나리오 (이름 → 이유). 다음 실행에서 건너뛴다 (--overwrite면 다시)."""
    p = os.path.join(out, 'skipped.json')
    if not os.path.isfile(p):
        return {}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def save_skipped(out: str, skipped: dict) -> None:
    with open(os.path.join(out, 'skipped.json'), 'w', encoding='utf-8') as f:
        json.dump(skipped, f, ensure_ascii=False, indent=1)


def build_report(out: str, policy: str, kin_eval: str | None = None) -> dict:
    """out 폴더의 에피소드 .npz → eval_isaac.json (시편군·강도별 측정·명령·운동학 요약)."""
    kin = kin_rows(kin_eval)
    rows, failed = [], sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, '*.failed.npz')))
    for p in sorted(glob.glob(os.path.join(out, '*.npz'))):
        if p.endswith('.failed.npz'):
            continue
        rec, scn, meta = load_episode(p)
        if 'policy_c' not in rec:
            continue
        r = episode_metrics(rec, scn)
        epi = scn.get('episode', {})
        r.update(file=os.path.basename(p), family=family(scn), kind=epi.get('kind', ''), mode=epi.get('mode', ''),
                 ik_not_ok=float(np.mean(rec['status'].astype(str) != 'ok')))
        rows.append(r)
    res = {'policy': policy, 'kin_eval': kin_eval, 'n': len(rows), 'failed_runs': failed, 'skipped': load_skipped(out),
           'groups': {}, 'episodes': rows}

    def group(name: str, sel: list[dict]) -> None:
        if not sel:
            return
        g = {'isaac': summarize(sel), 'isaac_cmd': summarize([as_cmd(r) for r in sel]),
             'track_p95_mm': float(np.median([r['track_p95_mm'] for r in sel])),
             'track_p95_max_mm': float(max(r['track_p95_mm'] for r in sel)),
             'lag_mm': float(np.mean([r['lag_mm'] for r in sel])),
             'ik_not_ok': float(np.mean([r['ik_not_ok'] for r in sel]))}
        same = [kin[os.path.splitext(r['file'])[0]] for r in sel if os.path.splitext(r['file'])[0] in kin]
        if same:
            g['kinematic'] = summarize(same)
        res['groups'][name] = g

    group('all', rows)
    for fam in FAMILIES:
        sel = [r for r in rows if r['family'] == fam]
        group(f'fam_{fam}', sel)
        for kind in ('clean', 'single', 'mixed', 'hard'):
            group(f'fam_{fam}_{kind}', [r for r in sel if r['kind'] == kind])
    with open(os.path.join(out, 'eval_isaac.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def print_report(res: dict) -> None:
    print(f'\n정책 {res["policy"]}: Isaac 닫힌 루프 {res["n"]}개'
          + (f', 끝나지 않은 실행 {len(res["failed_runs"])}개' if res['failed_runs'] else '')
          + (f', 시작 못 함(도달·배치) {len(res["skipped"])}개' if res.get('skipped') else ''))
    for name, why in (res.get('skipped') or {}).items():
        print(f'  시작 못 함 {name}: {why}')
    print('RMS 좌우·높이 mm, 작업각·진행각 deg. 측정 = 실측 토치, 명령 = 정책 보정, 운동학 = 같은 에피소드 운동학 평가,'
          ' 괄호 = 보정 0 대비 개선율')
    head = (f'{"묶음":<26}{"개수":>4}{"완료":>6}' + ''.join(f'{n:>16}' for n in ('좌우', '높이', '작업각', '진행각'))
            + f'{"추종95% mm":>11}{"뒤짐 mm":>8}')
    print(head)
    for name, g in res['groups'].items():
        if name != 'all' and not name.startswith('fam_'):
            continue
        if name.count('_') > 2 and not name.endswith(('_hard', '_mixed')):    # 강도별은 가혹·혼합만 보여 줌
            continue
        label = name.replace('fam_', '')
        for src, tag in (('isaac', '측정'), ('isaac_cmd', '명령'), ('kinematic', '운동학')):
            v = g.get(src)
            if v is None:
                continue
            cells = ''.join(f'{v[n]:>8.2f}' + (f' ({v[n + "_gain"] * 100:+4.0f}%)' if math.isfinite(v[n + '_gain'])
                                                else '  (  - )') for n in AXES)
            tail = f'{g["track_p95_mm"]:>11.2f}{g["lag_mm"]:>8.2f}' if src == 'isaac' else ''
            print(f'{(label + " " + tag):<26}{v["n"]:>4}{v["done"] * 100:>5.0f}%{cells}{tail}')


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', required=True, help="policy.pt 경로, 또는 규칙 기준선 'rule' / 'rule-noprescan'")
    ap.add_argument('--scenarios', help='시나리오 *.json 또는 에피소드 *.npz 폴더 (--report-only면 없어도 됨)')
    ap.add_argument('--split', default='test', help="이 나눔의 시나리오만 (train, val, test, all)")
    ap.add_argument('--only', help='이름에 이 문자열이 든 시나리오만 (예: pipe_)')
    ap.add_argument('--per-family', type=int, help='시편군마다 앞에서 N개만')
    ap.add_argument('--limit', type=int, help='전체 앞에서 N개만')
    ap.add_argument('--run', help='결과 폴더 이름 (기본: 정책 폴더 이름 또는 rule)')
    ap.add_argument('--out', help='출력 폴더 (기본: --exp면 <실험>/runs/<run>/isaac, 아니면 outputs/scratch/policy_isaac/<run>)')
    ap.add_argument('--kin-eval', help='같은 정책의 운동학 eval.json (같은 에피소드 비교)')
    ap.add_argument('--follower-urdf', help='opti3.urdf 경로')
    ap.add_argument('--follower-config', default=os.path.join(REPO, 'weld_teleop', 'config', 'follower_opti3.yaml'))
    ap.add_argument('--home-q', default=HOME_Q, help='rad, receiver --initial-q와 같게 (모재 배치 기준)')
    ap.add_argument('--send', default='127.0.0.1:9870')
    ap.add_argument('--listen', default='127.0.0.1:9871')
    ap.add_argument('--max-time', type=float, default=600.0, help='에피소드 하나의 최대 실제 시간 [s]')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--report-only', action='store_true', help='실행 없이 저장된 에피소드로 요약만')
    paths.add_exp_arg(ap)
    args = ap.parse_args(argv)

    run = args.run or (args.policy if args.policy in RULES else
                       os.path.basename(os.path.dirname(os.path.abspath(args.policy))) or 'policy')
    out = args.out or str(paths.exp_dir(args.exp, 'runs') / run / 'isaac' if args.exp
                          else paths.scratch_dir('policy_isaac', run))
    os.makedirs(out, exist_ok=True)

    if not args.report_only:
        if not args.scenarios or not args.follower_urdf:
            ap.error('--scenarios와 --follower-urdf가 필요합니다 (요약만 다시 내려면 --report-only)')
        if args.policy not in RULES and not os.path.isfile(args.policy):
            sys.exit(f'정책 파일이 없음: {args.policy}')
        todo_all = select(args.scenarios, args.split, args.only, args.per_family, args.limit)
        out_of = lambda name: os.path.join(out, name + '.npz')  # noqa: E731
        skipped = {} if args.overwrite else load_skipped(out)
        todo = [(n, s) for n, s in todo_all
                if args.overwrite or not (os.path.exists(out_of(n)) or n in skipped)]
        print(f'시나리오 {len(todo_all)}개 (나눔 {args.split}), 새로 돌릴 것 {len(todo)}개 → {os.path.abspath(out)}')
        paths.record_stage(args.exp, 'policy_to_sim', policy=args.policy, policy_sha256=paths.file_sha256(args.policy),
                           scenarios=args.scenarios, split=args.split, only=args.only, per_family=args.per_family,
                           limit=args.limit, out=out, follower_urdf=args.follower_urdf,
                           follower_config_sha256=paths.file_sha256(args.follower_config), n=len(todo))
        if todo:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from expert_to_sim import SimLink, build_stepper, run_episode
            policy = load_policy(args.policy)
            stepper = build_stepper(args.follower_config, args.follower_urdf)
            link = SimLink(args.listen, args.send, stepper.kin.nq)
            home = np.array([float(x) for x in args.home_q.split(',')])
            base_id = int(time.time()) % 100000 * 1000
            scn_dir = os.path.join(out, 'scenarios')
            os.makedirs(scn_dir, exist_ok=True)
            try:
                for k, (name, scn) in enumerate(todo, 1):
                    scn_path = os.path.join(scn_dir, name + '.json')   # receiver는 JSON 경로로 모재를 만든다
                    save_scenario(scn, scn_path)
                    seed = sensor_seed(name)
                    print(f'\n[{k}/{len(todo)}] {name}')
                    t1 = time.time()
                    rec, scn_out, result = run_episode(
                        link, stepper, scn, scn_path, home_q=home, sensor_seed=seed, reset_id=base_id + k,
                        max_time=args.max_time,
                        make_driver=lambda m, e, seed=seed: PolicyDriver(policy, m, e, sensor_seed=seed))
                    print(f'\n  결과 {result}, {len(rec)} 주기, {time.time() - t1:.0f} s')
                    if result == 'reach':          # 배치 실패(place)는 receiver 문제일 수 있어 다음에 다시 함
                        skipped[name] = result
                        save_skipped(out, skipped)
                    elif len(rec):
                        path = out_of(name) if result == 'done' else out_of(name).replace('.npz', '.failed.npz')
                        rec.save(path, scn_out, result=result, runner='isaac_policy', policy=args.policy,
                                 sensor_seed=seed)
            except KeyboardInterrupt:
                print('\n중단 (같은 명령으로 이어서 실행)')
            finally:
                link.close()

    res = build_report(out, args.policy, args.kin_eval)
    print_report(res)
    print(f'요약: {os.path.join(os.path.abspath(out), "eval_isaac.json")}')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
