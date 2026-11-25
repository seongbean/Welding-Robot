#!/usr/bin/env python3
"""학습용 묶음 일괄 실행: 시나리오 폴더의 에피소드를 연달아 만들고, 하나씩 자동 점검하고, 묶음 통계를 낸다.

사용법 (저장소 루트에서)
  python -m sim_errors.make_scenarios --batch --scale 0.2 --out batches/t_small
  터미널 1 (Isaac Sim, 한 번만 띄움):
    C:\\isaacsim\\python.bat sim_bridge\\isaac_opti3_receiver.py --lockstep
  터미널 2 (conda activate weld):
    python sim_bridge/run_batch.py batches/t_small --out episodes/t_small --follower-urdf C:/.../opti3.urdf
  Isaac 없이 흐름만 시험 (완벽 추종 가정, pinocchio·URDF 불필요):
    python sim_bridge/run_batch.py batches/t_small --out episodes/t_small_kin --kinematic --jobs 4

- 이미 있는 에피소드(.npz)는 건너뛴다 (중단 후 같은 명령으로 이어서). 다시 만들려면 --overwrite.
- 에피소드마다 check_episode 8항목을 돌려 <out>/batch_report.json에 남긴다. 실패한 에피소드는 ok=false와 이유가
  적히고 learning/to_lerobot.py가 뺀다. 주의는 통과로 본다.
- 묶음 통계 (강도별): 개수·통과, 정답 크기 백분위, 단면 측정 실패율, IK 경고 비율, 추종 오차 95%, 용접 시간.
- receiver는 --scenario 없이 띄운다. 시나리오는 패킷으로 바뀐다. --headless --lockstep이면 실시간보다 빠르다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from common import paths  # noqa: E402
from sim_errors.check_episode import FAIL, WARN, check, load, tracking_error  # noqa: E402
from sim_errors.episode import run_kinematic  # noqa: E402
from sim_errors.part_errors import DEG, MM, load_scenario  # noqa: E402


def check_file(path: str) -> dict:
    rows, scn = load(path)
    res = check(rows, scn)
    epi = scn.get('episode', {})
    weld = rows['arc_on'].astype(bool)
    e = rows['e'][weld]
    rv = rows['rotvec'][weld]
    las = rows['laser'][weld]
    in_range = rows['s'][weld] + scn['sensors']['laser_lookahead'] / scn['geometry']['length'] <= 1.0   # 끝 너머 제외
    status = rows['status'].astype(str)
    pos, _ = tracking_error(rows)
    fails = [f'{n}: {m}' for n, lv, m in res if lv == FAIL]
    return {'file': os.path.basename(path), 'kind': epi.get('kind', ''), 'mode': epi.get('mode', ''),
            'split': epi.get('split', ''), 'single_error': scn.get('single_error'), 'perturb': epi.get('perturb', False),
            'ok': not fails, 'reason': '; '.join(fails),
            'warn': [n for n, lv, _ in res if lv == WARN],
            'checks': {n: lv for n, lv, _ in res},
            'lateral_max_mm': float(np.abs(e[:, 1]).max() / MM) if len(e) else 0.0,
            'height_max_mm': float(np.abs(e[:, 2]).max() / MM) if len(e) else 0.0,
            'work_max_deg': float(np.abs(rv[:, 0]).max() / DEG) if len(rv) else 0.0,
            'travel_max_deg': float(np.abs(rv[:, 1]).max() / DEG) if len(rv) else 0.0,
            'gap_max_mm': float(rows['gap'][weld].max() / MM) if weld.any() else 0.0,
            'dropout': float(np.mean(~np.isfinite(las[in_range, 0]))) if in_range.any() else 1.0,
            'ik_not_ok': float(np.mean(status != 'ok')),
            'track_p95_mm': float(np.percentile(pos[weld], 95)) if weld.any() else float('nan'),
            'weld_time_s': float(weld.sum() * np.median(np.diff(rows['t']))) if len(rows['t']) > 1 else 0.0}


def summary(eps: list[dict]) -> str:
    lines = []
    kinds = sorted({e['kind'] for e in eps}, key=lambda k: ['clean', 'single', 'mixed', 'hard', ''].index(k)
                   if k in ('clean', 'single', 'mixed', 'hard', '') else 9)
    lines.append(f'{"강도":<8}{"개수":>5}{"통과":>5}{"주의":>5}'
                 f'{"좌우 최대 mm":>18}{"높이 최대 mm":>18}{"작업각 deg":>18}'
                 f'{"측정 실패":>9}{"IK 경고":>8}{"추종 95% mm":>16}{"용접 s":>12}')
    pct = lambda v: f'{np.percentile(v, 50):5.2f}/{np.percentile(v, 99):5.2f}'  # noqa: E731
    for k in kinds + ['전체']:
        g = eps if k == '전체' else [e for e in eps if e['kind'] == k]
        if not g:
            continue
        col = lambda key: np.array([e[key] for e in g])  # noqa: E731
        lines.append(f'{k or "-":<8}{len(g):>5}{sum(e["ok"] for e in g):>5}{sum(bool(e["warn"]) for e in g):>5}'
                     f'{pct(col("lateral_max_mm")):>18}{pct(col("height_max_mm")):>18}{pct(col("work_max_deg")):>18}'
                     f'{np.mean(col("dropout")) * 100:>8.1f}%{np.mean(col("ik_not_ok")) * 100:>7.1f}%'
                     f'{pct(col("track_p95_mm")):>16}{np.mean(col("weld_time_s")):>12.1f}')
    lines.append('(백분위 칸은 중앙/99%)')
    bad = [e for e in eps if not e['ok']]
    for e in bad:
        lines.append(f'  실패 {e["file"]}: {e["reason"]}')
    return '\n'.join(lines)


def write_report(out: str, eps: list[dict]) -> None:
    with open(os.path.join(out, 'batch_report.json'), 'w', encoding='utf-8') as f:
        json.dump({'episodes': eps, 'n': len(eps), 'ok': sum(e['ok'] for e in eps)}, f, ensure_ascii=False, indent=1)


def _kin_job(args) -> str:
    path, out_path, seed = args
    scn = load_scenario(path)
    rec = run_kinematic(scn, sensor_seed=seed)
    rec.save(out_path, scn, result='done' if rec.rows['done'][-1] else 'timeout', runner='kinematic')
    return out_path


def family_of(name: str) -> str:
    for pre, fam in (('corner_', 'fillet_corner'), ('butt_', 'butt_v'), ('pipe_', 'pipe_girth')):
        if name.startswith(pre):
            return fam
    return 'fillet_t'


def reach_only(files: list[str], args) -> int:
    """시나리오마다 자동 배치 → 이음매 41점 IK·시작 자세 IK. Isaac 실행 전에 몇 분이면 끝난다."""
    if not args.follower_urdf:
        sys.exit('--reach-only에는 --follower-urdf가 필요합니다')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from expert_to_sim import build_stepper, reach_scan
    from sim_errors.expert import ScriptedExpert, isaac_nominal
    from sim_errors.part_errors import PartErrorModel
    stepper = build_stepper(args.follower_config, args.follower_urdf)
    kin = stepper.kin
    home = np.array([float(x) for x in args.home_q.split(',')])
    T_home = kin.fk(home).homogeneous
    rows = []
    for k, f in enumerate(files, 1):
        model = PartErrorModel(load_scenario(f))
        model.set_nominal(isaac_nominal(model, T_home))
        expert = ScriptedExpert(model, T_home, np.random.default_rng(0))
        r = reach_scan(kin, expert, home, stepper.tol, stepper.accept)
        r.update(file=os.path.basename(f), family=family_of(os.path.basename(f)))
        rows.append(r)
        if not r['ok'] or k % 20 == 0 or k == len(files):
            print(f'[{k}/{len(files)}] {r["file"]}: {"통과" if r["ok"] else "실패"} 최대 오차 {r["max_err"]:.4f}, '
                  f'시작 {r["start_err"]:.4f}, 관절 변화 {r["joint_jump_rad"]:.3f} rad, 리밋 여유 {r["limit_margin_rad"]:.2f} rad',
                  flush=True)
    with open(os.path.join(args.out, 'reach_report.json'), 'w', encoding='utf-8') as fo:
        json.dump({'n': len(rows), 'ok': sum(r['ok'] for r in rows), 'tol': stepper.tol, 'accept': stepper.accept,
                   'scenarios': rows}, fo, ensure_ascii=False, indent=1)
    print(f'\n{"시편군":<14}{"개수":>5}{"통과":>5}{"최대 오차":>10}{"관절 변화 rad":>14}{"리밋 여유 rad":>14}  관절 범위 (rad, 최소..최대)')
    for fam in ('fillet_t', 'fillet_corner', 'butt_v', 'pipe_girth'):
        g = [r for r in rows if r['family'] == fam]
        if not g:
            continue
        lo = np.min([r['q_min'] for r in g], axis=0)
        hi = np.max([r['q_max'] for r in g], axis=0)
        rng = ' '.join(f'{a:+.2f}..{b:+.2f}' for a, b in zip(lo, hi))
        print(f'{fam:<14}{len(g):>5}{sum(r["ok"] for r in g):>5}{max(r["max_err"] for r in g):>10.4f}'
              f'{max(r["joint_jump_rad"] for r in g):>14.3f}{min(r["limit_margin_rad"] for r in g):>14.2f}  {rng}')
    print(f'보고서: {os.path.join(os.path.abspath(args.out), "reach_report.json")}')
    return 0 if all(r['ok'] for r in rows) else 1


def resolve_args(ap: argparse.ArgumentParser, args) -> None:
    """--exp가 있으면 비어 있는 scenarios·--out을 실험 폴더 배치로 채운다. 없으면 둘 다 필수."""
    if args.exp:
        if not args.scenarios:
            args.scenarios = str(paths.exp_dir(args.exp, 'batch'))
        if not args.out:
            if args.reach_only:
                args.out = str(paths.exp_dir(args.exp, 'reach'))
            elif args.kinematic:
                args.out = str(paths.exp_dir(args.exp, 'episodes'))
            else:
                args.out = str(paths.exp_dir(args.exp, 'episodes_isaac') / args.group)
    if not args.scenarios or not args.out:
        ap.error('시나리오 폴더와 --out이 필요합니다 (또는 --exp)')


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scenarios', nargs='?', help='make_scenarios --batch 출력 폴더 (--exp면 기본 <실험>/batch)')
    ap.add_argument('--out', help='에피소드 .npz 폴더 (--exp면 기본: --kinematic → <실험>/episodes, '
                                  '--reach-only → <실험>/reach, Isaac → <실험>/episodes_isaac/<--group>)')
    ap.add_argument('--group', default='all', help='--exp + Isaac 실행일 때 episodes_isaac 아래 폴더 이름 (예: butt)')
    ap.add_argument('--follower-urdf', help='opti3.urdf 경로 (Isaac 실행)')
    ap.add_argument('--follower-config', default=os.path.join(REPO, 'weld_teleop', 'config', 'follower_opti3.yaml'))
    ap.add_argument('--home-q', default='0,-0.5236,1.7453,0,0.8727,0', help='receiver --initial-q와 같게')
    ap.add_argument('--send', default='127.0.0.1:9870')
    ap.add_argument('--listen', default='127.0.0.1:9871')
    ap.add_argument('--kinematic', action='store_true', help='Isaac 없이 완벽 추종 가정')
    ap.add_argument('--jobs', type=int, default=1, help='--kinematic 병렬 프로세스 수')
    ap.add_argument('--limit', type=int, help='앞에서 N개만')
    ap.add_argument('--only', help='이름에 이 문자열이 든 시나리오만 (예: mixed)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--report-only', action='store_true', help='실행 없이 있는 에피소드로 보고서만')
    ap.add_argument('--reach-only', action='store_true',
                    help='Isaac 없이 opti3 IK 도달만 점검 (--follower-urdf 필요). <out>/reach_report.json')
    paths.add_exp_arg(ap)
    args = ap.parse_args(argv)
    resolve_args(ap, args)
    if not args.report_only:
        paths.record_stage(args.exp, 'run_batch', scenarios=args.scenarios, out=args.out,
                           runner='reach' if args.reach_only else 'kinematic' if args.kinematic else 'isaac',
                           follower_urdf=args.follower_urdf, follower_config=args.follower_config,
                           follower_config_sha256=paths.file_sha256(args.follower_config),
                           only=args.only, limit=args.limit)

    files = sorted(glob.glob(os.path.join(args.scenarios, '*.json')))
    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
    if args.limit:
        files = files[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    out_of = lambda f: os.path.join(args.out, os.path.splitext(os.path.basename(f))[0] + '.npz')  # noqa: E731
    todo = [f for f in files if args.overwrite or not os.path.exists(out_of(f))]
    seed_of = lambda f: files.index(f)  # noqa: E731   # 단면 센서 노이즈 시드
    print(f'시나리오 {len(files)}개, 새로 만들 것 {0 if args.report_only else len(todo)}개 → {os.path.abspath(args.out)}')
    t0 = time.time()

    if args.reach_only:
        return reach_only(files, args)

    if not args.report_only and todo:
        if args.kinematic:
            jobs = [(f, out_of(f), seed_of(f)) for f in todo]
            if args.jobs > 1:
                with ProcessPoolExecutor(args.jobs) as ex:
                    for k, p in enumerate(ex.map(_kin_job, jobs), 1):
                        print(f'[{k}/{len(jobs)}] {os.path.basename(p)}', flush=True)
            else:
                for k, j in enumerate(jobs, 1):
                    _kin_job(j)
                    print(f'[{k}/{len(jobs)}] {os.path.basename(j[1])}', flush=True)
        else:
            if not args.follower_urdf:
                sys.exit('--follower-urdf가 필요합니다 (Isaac 없이 시험하려면 --kinematic)')
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from expert_to_sim import SimLink, build_stepper, run_episode
            stepper = build_stepper(args.follower_config, args.follower_urdf)
            link = SimLink(args.listen, args.send, stepper.kin.nq)
            home = np.array([float(x) for x in args.home_q.split(',')])
            base_id = int(time.time()) % 100000 * 1000
            try:
                for k, f in enumerate(todo, 1):
                    name = os.path.basename(f)
                    print(f'\n[{k}/{len(todo)}] {name}')
                    t1 = time.time()
                    rec, scn_out, result = run_episode(link, stepper, load_scenario(f), f, home_q=home,
                                                       sensor_seed=seed_of(f), reset_id=base_id + k)
                    print(f'\n  결과 {result}, {len(rec)} 주기, {time.time() - t1:.0f} s')
                    if len(rec) and result == 'done':
                        rec.save(out_of(f), scn_out, result=result, runner='isaac')
                    elif len(rec):
                        rec.save(out_of(f).replace('.npz', '.failed.npz'), scn_out, result=result, runner='isaac')
            except KeyboardInterrupt:
                print('\n중단 (같은 명령으로 이어서 실행)')
            finally:
                link.close()

    done = [out_of(f) for f in files if os.path.exists(out_of(f))]
    eps = []
    for p in done:
        try:
            eps.append(check_file(p))
        except Exception as e:   # 깨진 파일
            eps.append({'file': os.path.basename(p), 'kind': '', 'mode': '', 'split': '', 'ok': False,
                        'reason': f'읽기 실패: {e}', 'warn': [], 'checks': {}})
    failed_runs = sorted(glob.glob(os.path.join(args.out, '*.failed.npz')))
    good = [e for e in eps if 'lateral_max_mm' in e]
    write_report(args.out, eps)
    print(f'\n묶음 점검 ({len(eps)}개, {time.time() - t0:.0f} s)')
    if good:
        print(summary(good))
    for e in eps:
        if 'lateral_max_mm' not in e:
            print(f'  실패 {e["file"]}: {e["reason"]}')
    if failed_runs:
        print(f'끝나지 않은 실행 {len(failed_runs)}개 (*.failed.npz, 학습에서 제외): '
              + ', '.join(os.path.basename(p) for p in failed_runs[:10]))
    print(f'보고서: {os.path.join(os.path.abspath(args.out), "batch_report.json")}')
    return 0 if all(e['ok'] for e in eps) else 1


if __name__ == '__main__':
    sys.exit(main() or 0)
