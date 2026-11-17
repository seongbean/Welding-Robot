#!/usr/bin/env python3
"""규칙 전문가 에피소드 기록(.npz)이 의도대로 만들어졌는지 점검하고 그래프를 그린다.

사용법 (저장소 루트에서)
  python -m sim_errors.check_episode episodes/part0000_pass0.npz
  python -m sim_errors.check_episode episodes/*.npz --no-plot      # 여러 개, 표만

점검 항목 (통과/주의/실패)
  1. 단계 순서      approach → descend → weld → retract → 완료 (용접 구간만 담은 에피소드는 weld → 완료)
  2. IK 상태        FollowerStepper 상태가 ok인 비율 (approx·ik_failed·tracking_error 등은 주의)
  3. 진행도         s가 줄지 않고 1까지 감
  4. 정답 재계산    기록된 정답(e, rotvec, 틈새)이 시나리오로 다시 계산한 값과 같음
  5. 목표 기하      용접 중 전문가 목표 = 실제 이음매 + standoff(+가접) + 위빙, 토치 방향 = 이음매 쪽 + 미는 각,
                    보낸 목표 = 전문가 목표 + 기록된 어긋남(시작 어긋남·섭동)
  6. 추종 오차      실측 관절로 구한 토치 팁(T_meas)과 목표(T_target)의 차이. 로봇이 실제로 따라갔는지
  7. 단면 센서      측정 실패율, 좌우·높이·틈새 폭 측정값과 정답의 차이가 노이즈 수준인지
  8. 용접 시간      이음매 길이 / 속도와 맞는지
그래프(matplotlib 있으면): 진행도에 따른 정답·레이저, 각도, 틈새·위빙·속도, 추종 오차 → <npz 이름>.png

주의: T_meas는 URDF FK라서 Isaac 장면의 토치 팁(weld_tip)이 URDF와 다르면 여기서는 안 보인다.
그건 Isaac 화면에서 토치 끝이 빨간 선(실제 이음매)을 따라가는지로 확인한다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from .expert import apply_offset
from .part_errors import DEG, MM, PartErrorModel, R_to_rotvec

OK, WARN, FAIL = '통과', '주의', '실패'
PHASES = ['approach', 'descend', 'weld', 'retract', 'done']


def load(path: str) -> tuple[dict, dict]:
    d = np.load(path, allow_pickle=False)
    rows = {k: d[k] for k in d.files if k != 'scenario_json'}
    return rows, json.loads(str(d['scenario_json']))


def check(rows: dict, scn: dict, track_tol_mm: float = 2.0) -> list[tuple[str, str, str]]:
    """점검 결과 [(항목, 통과/주의/실패, 설명)]."""
    out = []
    m = PartErrorModel(scn)
    ex = scn['expert']
    phase = rows['phase'].astype(str)
    weld = rows['arc_on'].astype(bool)
    n = len(phase)

    # 1. 단계 순서 (이전 기록은 단계가 바뀌는 주기를 다음 단계 이름으로 적어서 끝에 done이 있음)
    seq = [p for i, p in enumerate(phase) if i == 0 or p != phase[i - 1]]
    finished = 'done' in seq or ('done' in rows and bool(rows['done'][-1]))
    if seq[0] == 'weld':                      # 용접 구간만 담은 에피소드
        out.append(('단계 순서', OK if seq in (['weld'], ['weld', 'done']) and finished else FAIL,
                    'weld → 완료 (용접 구간만)' if finished else f'{" → ".join(seq)}, 완료 안 됨'))
    elif seq[:4] == PHASES[:4] and finished:
        out.append(('단계 순서', OK, ' → '.join(seq[:4]) + ' → 완료'))
    elif seq == PHASES[:len(seq)]:
        out.append(('단계 순서', FAIL, f'{seq[-1]}에서 끝남 (중단 또는 IK 실패)'))
    else:
        out.append(('단계 순서', FAIL, ' → '.join(seq)))

    # 2. IK 상태
    status = rows['status'].astype(str)
    names, counts = np.unique(status, return_counts=True)
    frac_ok = float(np.mean(status == 'ok'))
    detail = ', '.join(f'{a} {c}' for a, c in zip(names, counts))
    out.append(('IK 상태', OK if frac_ok == 1.0 else WARN if frac_ok > 0.95 else FAIL,
                f'ok {frac_ok * 100:.1f}% ({detail})'))

    # 3. 진행도
    s = rows['s']
    mono = bool(np.all(np.diff(s) >= -1e-12))
    out.append(('진행도', OK if mono and s.max() >= 1 - 1e-9 else FAIL,
                f'최대 {s.max():.3f}, {"단조 증가" if mono else "감소 구간 있음"}'))

    if not weld.any():
        out.append(('용접 구간', FAIL, '아크 ON 구간이 없음'))
        return out
    # 이전 기록에서 첫 'weld' 행은 하강 단계의 마지막 목표(진행도 0)라서 제외
    idx = np.flatnonzero(weld & (rows['s'] > 0))

    labs = {int(i): m.label(float(s[i]), float(rows['s_arc'][i])) for i in idx}

    # 4. 정답 재계산
    de = dr = dg = 0.0
    for i in idx:
        lab = labs[int(i)]
        de = max(de, float(np.max(np.abs(lab['e'] - rows['e'][i]))))
        dr = max(dr, float(np.max(np.abs(lab['rotvec'] - rows['rotvec'][i]))))
        dg = max(dg, abs(lab['gap'] - float(rows['gap'][i])))
    good = de < 1e-9 and dr < 1e-9 and dg < 1e-9
    out.append(('정답 재계산', OK if good else FAIL,
                f'최대 차이 위치 {de / MM:.2e} mm, 각도 {dr / DEG:.2e} deg, 틈새 {dg / MM:.2e} mm'))

    # 5. 목표 기하
    standoff = ex['standoff_mm'] * MM
    cos_travel = math.cos(ex['travel_angle_deg'] * DEG)
    worst_lift = worst_lat = worst_x = worst_ang = worst_cmd = 0.0
    T_exp = rows.get('T_expert', rows['T_target'])
    for i in idx:
        lab = labs[int(i)]
        R, T = lab['R_true'], T_exp[i]
        if 'perturb' in rows:
            Tc = apply_offset(T, R, rows['perturb'][i])
            worst_cmd = max(worst_cmd, float(np.abs(Tc - rows['T_target'][i]).max()))
        d = T[:3, 3] - lab['p_true']
        lift = standoff + ex['tack_follow'] * lab['tack']
        worst_lift = max(worst_lift, abs(float(d @ R[:, 2]) - lift))
        worst_lat = max(worst_lat, abs(float(d @ R[:, 1])) - float(rows['weave_amp'][i]))
        worst_x = max(worst_x, abs(float(d @ R[:, 0])))
        worst_ang = max(worst_ang, abs(float(T[:3, 2] @ -R[:, 2]) - cos_travel))
    good = worst_lift < 1e-6 and worst_lat < 1e-6 and worst_x < 1e-6 and worst_ang < 1e-6 and worst_cmd < 1e-6
    out.append(('목표 기하', OK if good else FAIL,
                f'높이 {worst_lift / MM:.1e} mm, 위빙 초과 {max(0.0, worst_lat) / MM:.1e} mm, '
                f'진행 방향 {worst_x / MM:.1e} mm, 토치 각 {worst_ang:.1e}, 어긋남 적용 {worst_cmd:.1e}'))

    # 6. 추종 오차
    pos, ang = tracking_error(rows)
    p95, pmax, a95 = np.percentile(pos[idx], 95), pos[idx].max(), np.percentile(ang[idx], 95)
    level = OK if p95 < track_tol_mm else WARN if p95 < 3 * track_tol_mm else FAIL
    out.append(('추종 오차 (용접 중)', level,
                f'위치 중앙 {np.median(pos[idx]):.2f} / 95% {p95:.2f} / 최대 {pmax:.2f} mm, 각도 95% {a95:.2f} deg'))

    # 7. 단면 센서 (코드 이름 laser): [좌우, 높이(+가접), 틈새 폭]. 이전 기록은 두 열
    las = rows['laser'][idx]
    valid = np.isfinite(las[:, 0])
    look = scn['sensors']['laser_lookahead'] / m.L
    in_range = s[idx] + look <= 1.0
    drop = 1.0 - float(np.mean(valid[in_range])) if in_range.any() else float('nan')
    exp_drop = scn['sensors']['laser_dropout']
    noise = scn['sensors']['laser_noise']
    res, res_gap = [], []
    for k in np.flatnonzero(valid):
        i = idx[k]
        lab = m.label(float(s[i]) + look, float(rows['s_arc'][i]))
        res.append(las[k, :2] - np.array([lab['e'][1], lab['e'][2] + lab['tack']]))
        if las.shape[1] > 2 and lab['gap'] > 3 * noise:   # 틈새 0 근처는 0에서 잘려 편향되므로 제외
            res_gap.append(las[k, 2] - lab['gap'])
    if res:
        sd = np.std(np.array(res), axis=0)
        sd_gap = float(np.std(res_gap)) if res_gap else 0.0
        good = abs(drop - exp_drop) < max(0.05, 3 * math.sqrt(exp_drop * (1 - exp_drop) / max(1, in_range.sum())))
        good = good and np.all(sd < 2 * noise + 1e-6) and sd_gap < 2 * noise + 1e-6
        gap_txt = f' 틈새 {sd_gap / MM:.3f}' if las.shape[1] > 2 else ''
        out.append(('단면 센서', OK if good else WARN,
                    f'실패율 {drop * 100:.1f}% (설정 {exp_drop * 100:.0f}%), 정답 대비 표준편차 '
                    f'좌우 {sd[0] / MM:.3f} 높이 {sd[1] / MM:.3f}{gap_txt} mm (설정 {noise / MM:.3f})'))
    else:
        out.append(('단면 센서', WARN, '유효 측정 없음'))

    # 8. 용접 시간
    t = rows['t']
    dt = float(np.median(np.diff(t))) if n > 1 else 0.0
    t_weld = len(idx) * dt
    v = ex['travel_speed_mm_s'] * MM
    lo, hi = m.L / v, m.L / (v * ex['min_speed_ratio'])
    out.append(('용접 시간', OK if lo * 0.95 <= t_weld <= hi * 1.05 else WARN,
                f'{t_weld:.1f} s (예상 {lo:.1f}~{hi:.1f} s)'))
    return out


def tracking_error(rows: dict) -> tuple[np.ndarray, np.ndarray]:
    """주기별 토치 팁 위치 오차 [mm], 방향 오차 [deg] (실측 FK vs 목표)."""
    Tt, Tm = rows['T_target'], rows['T_meas']
    pos = np.linalg.norm(Tm[:, :3, 3] - Tt[:, :3, 3], axis=1) / MM
    ang = np.array([np.linalg.norm(R_to_rotvec(a[:3, :3].T @ b[:3, :3])) for a, b in zip(Tt, Tm)]) / DEG
    return pos, ang


def plot(rows: dict, scn: dict, path: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    weld = rows['arc_on'].astype(bool)
    s = rows['s'][weld]
    e = rows['e'][weld] / MM
    r = rows['rotvec'][weld] / DEG
    las = rows['laser'][weld] / MM
    pos, _ = tracking_error(rows)
    for name in ('Malgun Gothic', 'AppleGothic', 'NanumGothic'):   # 한글 글꼴 (없으면 기본)
        if any(name in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False   # 한글 글꼴에 없는 유니코드 빼기 기호 대신 '-' 사용
    look = scn['sensors']['laser_lookahead'] / scn['geometry']['length']
    fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    ax[0].plot(s, e[:, 1], label='정답 좌우 e_y')
    ax[0].plot(s, e[:, 2], label='정답 높이 e_z')
    ax[0].plot(s + look, las[:, 0], '.', ms=2, alpha=0.5, label='단면 좌우 (측정 위치)')
    ax[0].plot(s + look, las[:, 1], '.', ms=2, alpha=0.5, label='단면 높이+가접 (측정 위치)')
    ax[0].set_ylabel('mm')
    ax[1].plot(s, r[:, 0], label='작업각 보정')
    ax[1].plot(s, r[:, 1], label='진행각 보정')
    ax[1].set_ylabel('deg')
    ax[2].plot(s, rows['gap'][weld] / MM, label='틈새 [mm]')
    if las.shape[1] > 2:
        ax[2].plot(s + look, las[:, 2], '.', ms=2, alpha=0.4, label='단면 틈새 폭 (측정 위치)')
    ax[2].plot(s, rows['weave_amp'][weld] / MM, label='위빙 진폭 [mm]')
    ax[2].plot(s, rows['speed'][weld] / MM, label='진행 속도 [mm/s]')
    ax[2].plot(s, rows['tack'][weld] / MM, label='가접 높이 [mm]')
    ax[3].plot(s, pos[weld], label='토치 팁 추종 오차')
    ax[3].set_ylabel('mm')
    ax[3].set_xlabel('진행도 s')
    sl = scn['slip']
    for a in ax:
        if np.linalg.norm(sl['drift']) > 0:
            a.axvspan(sl['drift_s0'], sl['drift_s0'] + sl['drift_span'], color='orange', alpha=0.1)
        if np.linalg.norm(sl['jump']) > 0:
            a.axvline(sl['jump_s'], color='red', ls=':', lw=1)
        a.legend(fontsize=8, loc='upper right')
        a.grid(alpha=0.3)
    ax[0].set_title('주황 구간 = 서서히 밀림, 빨간 점선 = 갑자기 튐')
    fig.tight_layout()
    out = os.path.splitext(path)[0] + '.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='expert_to_sim.py가 만든 .npz')
    ap.add_argument('--track-tol-mm', type=float, default=2.0, help='추종 오차 95% 통과 기준')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args(argv)

    worst = OK
    for path in args.files:
        rows, scn = load(path)
        res = check(rows, scn, args.track_tol_mm)
        print(f'\n{path}  (패스 {scn.get("pass_index", 0)}, {len(rows["t"])} 주기)')
        for name, lv, msg in res:
            print(f'  [{lv}] {name:<14} {msg}')
            worst = FAIL if FAIL in (worst, lv) else WARN if WARN in (worst, lv) else OK
        if not args.no_plot:
            png = plot(rows, scn, path)
            print(f'  그래프: {png}' if png else '  그래프: matplotlib 없음 (conda install matplotlib)')
    return 0 if worst != FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
