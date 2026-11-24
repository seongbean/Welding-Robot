#!/usr/bin/env python3
"""기준 정책 평가: 보정 0 기준선 대비 정답 오차.

사용법 (저장소 루트에서)
  python -m learning.evaluate runs/mlp_full/policy.pt --data datasets/t_small --episodes episodes/t_small
  python -m learning.evaluate runs/*/policy.pt --data datasets/t_small --episodes episodes/t_small   # 여러 정책 비교
  python -m learning.evaluate mlp_all rule --exp 20251201-weave --jobs 8   # <실험>/runs/<이름>/policy.pt, <실험>/data, <실험>/episodes
  python -m learning.evaluate runs/mlp_t/policy.pt --data datasets/t_corner --episodes episodes/t episodes/corner \
      --family fillet_corner --name corner        # T로 학습한 정책을 코너 시험에서 (→ runs/mlp_t/eval_corner.json)

두 가지를 본다.
1. 열린 루프: 시험 에피소드의 기록된 관측으로 행동을 예측해 전문가 정답과 비교 (행동별 RMSE, 보정 0과 비교).
2. 닫힌 루프 (운동학): 시험 에피소드의 시나리오에서 정책이 직접 토치를 움직인다. 정책 출력이 다음 관측(토치 자세,
   진행도, 단면 버퍼 위치)을 바꾸므로 열린 루프에서 안 보이는 누적 오차·지름길 학습이 드러난다.
   로봇은 명령을 바로 따라간다고 가정한다(Isaac 추종 오차 없음). 오차 = 정책이 놓은 토치 중심 − 실제 이음매
   (좌우, 높이+가접, 작업각, 진행각). 기준선 = 보정 0 (추정 경로 그대로). 개선율 = 1 − 정책 RMS / 기준선 RMS.
   위빙·속도는 전문가 규칙값과의 차이로 본다.
결과 표는 강도(혼합·단독·깨끗함·가혹)별로 나오고, <정책 폴더>/eval.json에 저장된다.
닫힌 루프는 에피소드마다 오래 걸리므로(운동학 한 에피소드 수십 초) --jobs N으로 나눠 돌린다.
정책 자리에 'rule'(또는 'rule-noprescan')을 주면 학습 없는 규칙 기준선(버퍼의 토치 위치 값을 그대로 씀)을 평가한다.
시편군이 둘 이상 섞인 데이터는 강도별 표에 시편군별 줄(fam_<군>, fam_<군>_<강도>)을 더한다. --family로 한 군만 평가할 수 있다.
높이 오차는 시작 30 mm와 나머지로도 나눠 보여 준다(시작 구간은 단면 센서가 토치 앞 30 mm만 재서 사각지대가 된다).
보정 0 오차가 0.01(mm 또는 deg)보다 작은 강도(깨끗함, 각도 오차가 없는 단독 오차)는 개선율 칸에 -를 쓴다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import paths
from sim_errors.expert import ScriptedExpert
from sim_errors.part_errors import DEG, MM, PartErrorModel, rotvec_to_R

from .features import (ACTION_NAMES, BUF_OFFSETS_MM, FPS, OBS_NAMES, SectionBuffer, fill_prescan, load_episode,
                       make_obs, torch_offset)

AXES = ('lateral', 'height', 'work', 'travel')
RULES = {'rule': True, 'rule-noprescan': False}   # 정책 자리에 쓰는 규칙 기준선 이름 → 시작 구간 스캔 여부


class RulePolicy:
    """학습 없는 기준선: 단면 버퍼에서 토치 위치(0 mm) 값을 그대로 보정으로 쓴다 (없으면 가장 가까운 유효 점).
    속도·위빙은 잰 틈새로 전문가 규칙을 그대로 쓴다. 정책이 센서 한계에 닿았는지 볼 때 비교한다."""

    K0 = int(np.flatnonzero(BUF_OFFSETS_MM == 0)[0])

    def __init__(self, name: str = 'rule'):
        self.cfg = {'model': 'rule', 'groups': ['section', 'profile', 'mode'], 'prescan': RULES[name]}
        self.ex = None
        self.last = np.zeros(4)

    def bind(self, scn: dict) -> None:
        self.ex = ScriptedExpert(PartErrorModel(scn), np.eye(4), weld_only=True)

    def reset(self):
        self.last = np.zeros(4)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        n = len(BUF_OFFSETS_MM)
        valid = obs[3 * n:4 * n]
        if valid.any():
            j = np.flatnonzero(valid)
            k = int(j[np.argmin(np.abs(j - self.K0))])
            self.last = np.array([obs[3 * k], obs[3 * k + 1], obs[4 * n + 2 * k], obs[4 * n + 2 * k + 1]])
            gap = float(obs[3 * k + 2])
        else:
            gap = 0.0
        mode = 'weave' if obs[OBS_NAMES.index('mode_weave')] > 0.5 else 'straight'
        if self.ex is None:
            raise RuntimeError('RulePolicy.bind(scn)를 먼저 불러야 합니다')
        self.ex.mode = mode
        amp, v = self.ex.process(gap * MM)
        return np.r_[self.last, v / MM, amp / MM, self.ex.weave_hz].astype(np.float32)

    def predict_episode(self, obs: np.ndarray) -> np.ndarray:
        self.reset()
        return np.array([self(o) for o in obs])


def load_frames(path: str) -> dict:
    """frames.npz (train_bc.load_frames와 같음. 규칙 기준선은 torch 없이 돌도록 여기 둠)."""
    d = np.load(os.path.join(path, 'frames.npz'), allow_pickle=False)
    return {k: d[k] for k in d.files}


def load_policy(path: str):
    if path in RULES:
        return RulePolicy(path)
    from .train_bc import Policy
    return Policy(path)


def open_loop(policy, fr: dict, eps: list[int], scenarios: dict | None = None) -> dict:
    if [str(n) for n in fr['obs_names']] != OBS_NAMES:
        raise SystemExit('데이터 관측 형식이 지금 코드와 다릅니다. learning.to_lerobot으로 다시 변환하세요.')
    P, Y = [], []
    for e in eps:
        m = fr['episode_index'] == e
        if hasattr(policy, 'bind'):
            if not scenarios:
                continue
            policy.bind(scenarios[e])
        policy.reset()
        P.append(policy.predict_episode(fr['obs'][m]))
        Y.append(fr['action'][m])
    if not P:
        return {}
    P, Y = np.concatenate(P), np.concatenate(Y)
    return {n: {'rmse': float(np.sqrt(np.mean((P[:, k] - Y[:, k]) ** 2))),
                'zero_rmse': float(np.sqrt(np.mean(Y[:, k] ** 2))) if k < 4 else None}
            for k, n in enumerate(ACTION_NAMES)}


def torch_pose(model: PartErrorModel, s: float, c, weave: float) -> np.ndarray:
    """추정 경로 + 보정 c = [좌우, 높이 mm, 작업각, 진행각 deg] + 위빙 [mm] → 토치 포즈 (롤은 임의)."""
    ex = model.scn['expert']
    p_e, R_e = model.seam_est(s)
    p_e, R_e = p_e[0], R_e[0]
    Rs = R_e @ rotvec_to_R([c[2] * DEG, c[3] * DEG, 0.0])
    a = ex['travel_angle_deg'] * DEG
    z = -Rs[:, 2] * math.cos(a) + Rs[:, 0] * math.sin(a)
    x = np.cross(Rs[:, 1], z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], axis=-1)
    T[:3, 3] = p_e + R_e @ np.array([0.0, (c[0] + weave) * MM, ex['standoff_mm'] * MM + c[1] * MM])
    return T


def truth(model: PartErrorModel, s: float) -> np.ndarray:
    lab = model.label(s, s)
    tf = model.scn['expert']['tack_follow']
    return np.array([lab['e'][1] / MM, (lab['e'][2] + tf * lab['tack']) / MM,
                     lab['rotvec'][0] / DEG, lab['rotvec'][1] / DEG]), lab['gap']


def closed_loop(policy, scn: dict, sensor_seed: int = 0, dt: float = 0.01, fps: int = FPS,
                start_zone_mm: float = 30.0) -> dict:
    """운동학 닫힌 루프 한 에피소드. 정책 cfg의 prescan이 참이면(그렇게 변환한 데이터로 학습) 용접 전 시작 구간
    스캔으로 버퍼를 먼저 채운다. 높이 오차는 시작 구간(처음 start_zone_mm)과 나머지로도 나눠 낸다."""
    model = PartErrorModel(scn)
    ex = scn['expert']
    epi = scn.get('episode', {})
    mode = epi.get('mode', 'weave')
    rule = ScriptedExpert(model, np.eye(4), mode=mode, weld_only=True)
    rng = np.random.default_rng(sensor_seed)
    look = scn['sensors']['laser_lookahead'] / model.L
    v_nom = ex['travel_speed_mm_s']
    buf = SectionBuffer(model.L)
    if policy.cfg.get('prescan', False) and not fill_prescan(buf, model, rng):
        print('경고: 정책은 시작 구간 스캔으로 학습했는데 시나리오에 스캔 설정(sensors.prescan)이 없음 '
              '(이전 시나리오면 sim_errors.resense로 지금 센서 설정을 넣으세요)', file=sys.stderr)
    if hasattr(policy, 'bind'):
        policy.bind(scn)
    so = np.array(epi.get('start_offset', [0.0] * 4))
    c = truth(model, 0.0)[0] + np.array([so[0] / MM, so[1] / MM, so[2] / DEG, so[3] / DEG])   # 놓인 자세
    speed, amp, hz = v_nom, 0.0, 0.0
    s = t = phase = 0.0
    every = int(round(1.0 / (fps * dt)))
    policy.reset()
    err, base, proc, s_log = [], [], [], []
    k = 0
    t_max = 3 * model.L / (v_nom * MM)
    while s < 1.0 and t < t_max:
        buf.add(s + look, model.laser(s, s, rng))
        weave = amp * math.sin(phase)
        if k % every == 0:
            vals, valid = buf.query(s)
            a = policy(make_obs(vals, valid, torch_offset(model, s, torch_pose(model, s, c, weave)), s, mode))
            c = a[:4].astype(float)
            speed = float(np.clip(a[4], 0.3 * v_nom, 2.0 * v_nom))
            amp = float(np.clip(a[5], 0.0, 2 * ex['weave_amp_max_mm']))
            hz = float(np.clip(a[6], 0.0, 3.0))
        tr, gap = truth(model, s)
        err.append(c - tr)
        base.append(-tr)
        s_log.append(s)
        amp_r, v_r = rule.process(gap)
        proc.append([speed - v_r / MM, amp - amp_r / MM, hz - rule.weave_hz])
        s = min(1.0, s + speed * MM * dt / model.L)
        phase += 2 * math.pi * hz * dt
        t += dt
        k += 1
    err, base, proc = np.array(err), np.array(base), np.array(proc)
    out = {'done': bool(s >= 1.0), 'time': t}
    for j, n in enumerate(AXES):
        out[n] = float(np.sqrt(np.mean(err[:, j] ** 2)))
        out[n + '_base'] = float(np.sqrt(np.mean(base[:, j] ** 2)))
        out[n + '_p95'] = float(np.percentile(np.abs(err[:, j]), 95))
    for j, n in enumerate(('speed', 'weave_amp', 'weave_hz')):
        out[n + '_err'] = float(np.sqrt(np.mean(proc[:, j] ** 2)))
    start = np.array(s_log) * model.L < start_zone_mm * MM
    out['height_sq_start'] = float(np.sum(err[start, 1] ** 2))      # 시작 구간 높이 제곱오차 합 (나눔 요약용)
    out['height_sq_rest'] = float(np.sum(err[~start, 1] ** 2))
    out['n_start'], out['n_rest'] = int(start.sum()), int((~start).sum())
    return out


_WORKER_POLICY: dict = {}


def _closed_loop_job(job: tuple) -> dict:
    """프로세스 하나에서 닫힌 루프 한 에피소드 (정책은 프로세스마다 한 번만 불러옴)."""
    policy_path, ep_path, seed = job
    if policy_path not in _WORKER_POLICY:
        if policy_path not in RULES:
            import torch
            torch.set_num_threads(1)
        _WORKER_POLICY.clear()
        _WORKER_POLICY[policy_path] = load_policy(policy_path)
    _, scn, _ = load_episode(ep_path)
    r = closed_loop(_WORKER_POLICY[policy_path], scn, sensor_seed=seed)
    r['file'] = os.path.basename(ep_path)
    return r


def summarize(rows: list[dict]) -> dict:
    out = {'n': len(rows), 'done': float(np.mean([r['done'] for r in rows]))}
    for n in AXES:
        p = float(np.sqrt(np.mean([r[n] ** 2 for r in rows])))
        b = float(np.sqrt(np.mean([r[n + '_base'] ** 2 for r in rows])))
        out[n] = p
        out[n + '_base'] = b
        out[n + '_gain'] = 1 - p / b if b >= 0.01 else float('nan')   # 고칠 오차가 거의 없으면 개선율은 의미 없음
        out[n + '_p95'] = float(np.mean([r[n + '_p95'] for r in rows]))
    for n in ('speed', 'weave_amp', 'weave_hz'):
        out[n + '_err'] = float(np.sqrt(np.mean([r[n + '_err'] ** 2 for r in rows])))
    if all('n_start' in r for r in rows):
        ss, sr = sum(r['height_sq_start'] for r in rows), sum(r['height_sq_rest'] for r in rows)
        ns, nr = sum(r['n_start'] for r in rows), sum(r['n_rest'] for r in rows)
        out['height_start'] = float(np.sqrt(ss / max(1, ns)))    # 시작 30 mm의 높이 RMS
        out['height_rest'] = float(np.sqrt(sr / max(1, nr)))
        out['height_start_share'] = float(ss / max(1e-12, ss + sr))   # 높이 제곱오차 중 시작 구간 몫
    return out


def _family(fr: dict, e: int) -> str:
    return str(fr['ep_family'][e]) if 'ep_family' in fr else 'fillet_t'


def _find(episodes, name: str) -> str:
    """에피소드 폴더(하나 또는 여러 개)에서 파일 찾기."""
    dirs = [episodes] if isinstance(episodes, str) else list(episodes)
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise SystemExit(f'에피소드 파일을 못 찾음: {name} (폴더: {", ".join(dirs)})')


def evaluate(policy_path: str, data: str, episodes, splits=('test',), verbose: bool = True,
             jobs: int = 1, rule_dir: str = 'runs', families=None, name: str | None = None) -> dict:
    """policy_path가 'rule' 또는 'rule-noprescan'이면 규칙 기준선을 평가하고 <rule_dir>/<이름>/eval.json에 쓴다.
    episodes: 에피소드 폴더 하나 또는 여러 개. families: 평가할 시편군 (None = 전부).
    name을 주면 결과 파일이 eval_<name>.json (같은 정책을 여러 시험 묶음에서 볼 때)."""
    pol = load_policy(policy_path)
    fr = load_frames(data)
    eps = [i for i, s in enumerate(fr['ep_split']) if s in splits and (not families or _family(fr, i) in families)]
    if families and not eps:
        raise SystemExit(f'평가할 에피소드가 없음 (나눔 {",".join(splits)}, 시편군 {families or "전부"})')
    scenarios = ({e: load_episode(_find(episodes, str(fr['ep_file'][e])))[1] for e in eps}
                 if episodes and hasattr(pol, 'bind') else None)
    res = {'policy': policy_path, 'groups': pol.cfg['groups'], 'model': pol.cfg['model'],
           'prescan': bool(pol.cfg.get('prescan', False)),
           'train_families': pol.cfg.get('families', ['fillet_t']) if pol.cfg['model'] != 'rule' else [],
           'eval_families': sorted({_family(fr, e) for e in eps}),
           'open_loop': open_loop(pol, fr, eps, scenarios) if eps else {}}
    if episodes:
        per_kind: dict[str, list] = {}
        job_list = [(policy_path, _find(episodes, str(fr['ep_file'][e])), 1000 + e) for e in eps]
        if jobs > 1:
            with ProcessPoolExecutor(jobs) as pool:
                rows = list(pool.map(_closed_loop_job, job_list))
        else:
            rows = [_closed_loop_job(j) for j in job_list]
        for e, r in zip(eps, rows):
            per_kind.setdefault(str(fr['ep_kind'][e]), []).append(r)
        res['closed_loop'] = {k: summarize(v) for k, v in per_kind.items()}
        res['closed_loop']['all'] = summarize([r for v in per_kind.values() for r in v])
        if len(res['eval_families']) > 1:
            for fam in res['eval_families']:
                sel = [(e, r) for e, r in zip(eps, rows) if _family(fr, e) == fam]
                res['closed_loop'][f'fam_{fam}'] = summarize([r for _, r in sel])
                for kind in sorted({str(fr['ep_kind'][e]) for e, _ in sel}):   # 예: fam_fillet_corner_hard
                    res['closed_loop'][f'fam_{fam}_{kind}'] = summarize(
                        [r for e, r in sel if str(fr['ep_kind'][e]) == kind])
        res['closed_loop_episodes'] = [r for v in per_kind.values() for r in v]
    out_dir = os.path.join(rule_dir, policy_path) if policy_path in RULES else os.path.dirname(policy_path)
    os.makedirs(out_dir or '.', exist_ok=True)
    with open(os.path.join(out_dir, f'eval_{name}.json' if name else 'eval.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    if verbose:
        report(res)
    return res


def report(res: dict) -> None:
    print(f'\n정책 {res["policy"]} ({res["model"]}, 입력 {"+".join(res["groups"])}, '
          f'시작 구간 스캔 {"있음" if res.get("prescan") else "없음"}, '
          f'학습 시편군 {"+".join(res.get("train_families") or ["-"])}, 시험 시편군 {"+".join(res.get("eval_families", []))})')
    if res['open_loop']:
        print('  열린 루프 RMSE (보정 0):  ' + '  '.join(
            f'{n} {v["rmse"]:.3f}' + (f' ({v["zero_rmse"]:.3f})' if v['zero_rmse'] is not None else '')
            for n, v in res['open_loop'].items()))
    if 'closed_loop' in res:
        print('  닫힌 루프 RMS: 정책 / 보정 0 (개선율)   좌우·높이 mm, 각도 deg')
        print(f'  {"강도":<8}{"개수":>4}{"완료":>6}' + ''.join(f'{n:>22}' for n in AXES)
              + f'{"속도 차":>9}{"위빙 차":>9}')
        for k, v in res['closed_loop'].items():
            print(f'  {k:<8}{v["n"]:>4}{v["done"] * 100:>5.0f}%' + ''.join(
                f'{v[n]:>8.2f} / {v[n + "_base"]:<5.2f}' + (f'({v[n + "_gain"] * 100:+4.0f}%)'
                                                            if math.isfinite(v[n + '_gain']) else '(   - )')
                for n in AXES)
                + f'{v["speed_err"]:>9.2f}{v["weave_amp_err"]:>9.2f}')
        a = res['closed_loop'].get('all', {})
        if 'height_start' in a:
            print(f'  높이 RMS: 시작 30 mm {a["height_start"]:.2f}, 나머지 {a["height_rest"]:.2f} mm '
                  f'(제곱오차 중 시작 구간 {a["height_start_share"] * 100:.0f}%)')


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policies', nargs='+',
                    help="train_bc.py가 만든 policy.pt, 또는 규칙 기준선 'rule' (시작 구간 스캔 있음) / 'rule-noprescan'")
    ap.add_argument('--data', help='to_lerobot.py 출력 폴더 (--exp면 기본 <실험>/data)')
    ap.add_argument('--episodes', nargs='+', help='에피소드 .npz 폴더 (닫힌 루프 평가용 시나리오), 여러 개 가능')
    ap.add_argument('--family', help='이 시편군 시험만 평가 (쉼표로 여러 개)')
    ap.add_argument('--name', help='결과 파일 이름 eval_<name>.json (기본 eval.json)')
    ap.add_argument('--splits', default='test', help='평가할 나눔 (쉼표로, 예: test,val)')
    ap.add_argument('--jobs', type=int, default=1, help='닫힌 루프 병렬 프로세스 수')
    ap.add_argument('--rule-dir', help='규칙 기준선 결과 폴더 (<폴더>/rule/eval.json). 기본 runs, --exp면 <실험>/runs')
    paths.add_exp_arg(ap)
    args = ap.parse_args(argv)
    if args.exp:
        runs = paths.exp_dir(args.exp, 'runs')
        args.data = args.data or str(paths.exp_dir(args.exp, 'data'))
        args.episodes = args.episodes or [str(paths.exp_dir(args.exp, 'episodes'))]
        args.rule_dir = args.rule_dir or str(runs)
        # 경로가 아닌 정책 이름(예: mlp_all)은 <실험>/runs/<이름>/policy.pt
        args.policies = [p if p in RULES or os.path.exists(p) else str(runs / p / 'policy.pt') for p in args.policies]
    args.rule_dir = args.rule_dir or 'runs'
    if not args.data:
        ap.error('--data가 필요합니다 (또는 --exp)')
    paths.record_stage(args.exp, 'evaluate', policies=args.policies, data=args.data, episodes=args.episodes,
                       family=args.family, splits=args.splits, name=args.name)
    for p in args.policies:
        evaluate(p, args.data, args.episodes, tuple(args.splits.split(',')), jobs=args.jobs, rule_dir=args.rule_dir,
                 families=args.family.split(',') if args.family else None, name=args.name)


if __name__ == '__main__':
    main()
