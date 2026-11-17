#!/usr/bin/env python3
"""기준 정책 평가: 보정 0 기준선 대비 정답 오차.

사용법 (저장소 루트에서)
  python -m learning.evaluate runs/mlp_full/policy.pt --data datasets/t_small --episodes episodes/t_small
  python -m learning.evaluate runs/*/policy.pt --data datasets/t_small --episodes episodes/t_small   # 여러 정책 비교

두 가지를 본다.
1. 열린 루프: 시험 에피소드의 기록된 관측으로 행동을 예측해 전문가 정답과 비교 (행동별 RMSE, 보정 0과 비교).
2. 닫힌 루프 (운동학): 시험 에피소드의 시나리오에서 정책이 직접 토치를 움직인다. 정책 출력이 다음 관측(토치 자세,
   진행도, 단면 버퍼 위치)을 바꾸므로 열린 루프에서 안 보이는 누적 오차·지름길 학습이 드러난다.
   로봇은 명령을 바로 따라간다고 가정한다(Isaac 추종 오차 없음). 오차 = 정책이 놓은 토치 중심 − 실제 이음매
   (좌우, 높이+가접, 작업각, 진행각). 기준선 = 보정 0 (추정 경로 그대로). 개선율 = 1 − 정책 RMS / 기준선 RMS.
   위빙·속도는 전문가 규칙값과의 차이로 본다.
결과 표는 강도(혼합·단독·깨끗함·가혹)별로 나오고, <정책 폴더>/eval.json에 저장된다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from sim_errors.expert import ScriptedExpert
from sim_errors.part_errors import DEG, MM, PartErrorModel, rotvec_to_R

from .features import ACTION_NAMES, FPS, SectionBuffer, load_episode, make_obs, torch_offset
from .train_bc import Policy, load_frames

AXES = ('lateral', 'height', 'work', 'travel')


def open_loop(policy: Policy, fr: dict, eps: list[int]) -> dict:
    P, Y = [], []
    for e in eps:
        m = fr['episode_index'] == e
        policy.reset()
        P.append(policy.predict_episode(fr['obs'][m]))
        Y.append(fr['action'][m])
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


def closed_loop(policy: Policy, scn: dict, sensor_seed: int = 0, dt: float = 0.01, fps: int = FPS) -> dict:
    """운동학 닫힌 루프 한 에피소드."""
    model = PartErrorModel(scn)
    ex = scn['expert']
    epi = scn.get('episode', {})
    mode = epi.get('mode', 'weave')
    rule = ScriptedExpert(model, np.eye(4), mode=mode, weld_only=True)
    rng = np.random.default_rng(sensor_seed)
    look = scn['sensors']['laser_lookahead'] / model.L
    v_nom = ex['travel_speed_mm_s']
    buf = SectionBuffer(model.L)
    so = np.array(epi.get('start_offset', [0.0] * 4))
    c = truth(model, 0.0)[0] + np.array([so[0] / MM, so[1] / MM, so[2] / DEG, so[3] / DEG])   # 놓인 자세
    speed, amp, hz = v_nom, 0.0, 0.0
    s = t = phase = 0.0
    every = int(round(1.0 / (fps * dt)))
    policy.reset()
    err, base, proc = [], [], []
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
    return out


def summarize(rows: list[dict]) -> dict:
    out = {'n': len(rows), 'done': float(np.mean([r['done'] for r in rows]))}
    for n in AXES:
        p = float(np.sqrt(np.mean([r[n] ** 2 for r in rows])))
        b = float(np.sqrt(np.mean([r[n + '_base'] ** 2 for r in rows])))
        out[n] = p
        out[n + '_base'] = b
        out[n + '_gain'] = 1 - p / b if b > 1e-9 else float('nan')
        out[n + '_p95'] = float(np.mean([r[n + '_p95'] for r in rows]))
    for n in ('speed', 'weave_amp', 'weave_hz'):
        out[n + '_err'] = float(np.sqrt(np.mean([r[n + '_err'] ** 2 for r in rows])))
    return out


def evaluate(policy_path: str, data: str, episodes: str | None, splits=('test',), verbose: bool = True) -> dict:
    pol = Policy(policy_path)
    fr = load_frames(data)
    eps = [i for i, s in enumerate(fr['ep_split']) if s in splits]
    res = {'policy': policy_path, 'groups': pol.cfg['groups'], 'model': pol.cfg['model'],
           'open_loop': open_loop(pol, fr, eps) if eps else {}}
    if episodes:
        per_kind: dict[str, list] = {}
        for e in eps:
            _, scn, _ = load_episode(os.path.join(episodes, str(fr['ep_file'][e])))
            r = closed_loop(pol, scn, sensor_seed=1000 + e)
            r['file'] = str(fr['ep_file'][e])
            per_kind.setdefault(str(fr['ep_kind'][e]), []).append(r)
        res['closed_loop'] = {k: summarize(v) for k, v in per_kind.items()}
        res['closed_loop']['all'] = summarize([r for v in per_kind.values() for r in v])
        res['closed_loop_episodes'] = [r for v in per_kind.values() for r in v]
    with open(os.path.join(os.path.dirname(policy_path), 'eval.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    if verbose:
        report(res)
    return res


def report(res: dict) -> None:
    print(f'\n정책 {res["policy"]} ({res["model"]}, 입력 {"+".join(res["groups"])})')
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
                f'{v[n]:>8.2f} / {v[n + "_base"]:<5.2f}({v[n + "_gain"] * 100:+4.0f}%)' for n in AXES)
                + f'{v["speed_err"]:>9.2f}{v["weave_amp_err"]:>9.2f}')


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policies', nargs='+', help='train_bc.py가 만든 policy.pt')
    ap.add_argument('--data', required=True, help='to_lerobot.py 출력 폴더')
    ap.add_argument('--episodes', help='에피소드 .npz 폴더 (닫힌 루프 평가용 시나리오)')
    ap.add_argument('--splits', default='test', help='평가할 나눔 (쉼표로, 예: test,val)')
    args = ap.parse_args(argv)
    for p in args.policies:
        evaluate(p, args.data, args.episodes, tuple(args.splits.split(',')))


if __name__ == '__main__':
    main()
