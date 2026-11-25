#!/usr/bin/env python3
"""SO-102 임시 URDF 가정 검증 스크립트 (자/줄자 실측 vs URDF FK)

검증 대상 (so102.urdf 주석의 임시 가정):
  가정 1: wrist_yaw → wrist_roll 횡방향 오프셋 0.0181 m 의 부호 (+ / -)
  가정 4: wrist_yaw 회전 방향 부호

원리
  - 두 부호 가설(A: +0.0181, B: -0.0181)로 각각 FK를 계산해 "자로 잴 수 있는 값"을 예측한다.
  - 좌표계에 의존하지 않도록 '두 자세 사이 TCP(고정 조 끝) 이동 거리'를 주로 쓴다.
  - 실측값을 입력하면 어느 가설이 더 맞는지 판정한다.

의존성: numpy 만 필요 (ROS2 / pinocchio 불필요, Windows에서 바로 실행)

사용법
  python tools/so102_urdf_check.py                 # 측정 절차와 예측값 출력
  python tools/so102_urdf_check.py --measure       # 측정값을 입력받아 판정
  python tools/so102_urdf_check.py --urdf path/to/so102.urdf --tcp-offset-mm 0 0 0
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

ARM = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll']
EE = 'gripper_frame_link'
YAW_TO_ROLL_JOINT = 'wrist_roll'   # 이 관절 origin의 z 성분이 가정 1 대상


# ----------------------------------------------------------------------------- FK
def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle_R(axis, th):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def T(R=np.eye(3), p=(0, 0, 0)):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


class Urdf:
    def __init__(self, path: str, yaw_roll_z_override: float | None = None):
        root = ET.parse(path).getroot()
        self.joints = {}
        self.child_of = {}
        for j in root.findall('joint'):
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz', '0 0 0').split() if o is not None else [0, 0, 0])]
            rpy = [float(v) for v in (o.get('rpy', '0 0 0').split() if o is not None else [0, 0, 0])]
            ax = j.find('axis')
            axis = [float(v) for v in ax.get('xyz').split()] if ax is not None else [0, 0, 1]
            lim = j.find('limit')
            limits = (float(lim.get('lower')), float(lim.get('upper'))) if lim is not None else None
            name = j.get('name')
            if name == YAW_TO_ROLL_JOINT and yaw_roll_z_override is not None:
                xyz[2] = yaw_roll_z_override
            self.joints[name] = dict(type=j.get('type'), parent=j.find('parent').get('link'),
                                     child=j.find('child').get('link'), xyz=xyz, rpy=rpy,
                                     axis=axis, limits=limits)
            self.child_of[j.find('child').get('link')] = name
        missing = [n for n in ARM if n not in self.joints]
        if missing:
            raise ValueError(f'URDF에 없는 관절: {missing}')

    def chain_to(self, link):
        names = []
        while link in self.child_of:
            jn = self.child_of[link]
            names.append(jn)
            link = self.joints[jn]['parent']
        return names[::-1]

    def fk(self, q: dict, link=EE, tcp_offset=np.zeros(3)):
        M = np.eye(4)
        for jn in self.chain_to(link):
            j = self.joints[jn]
            M = M @ T(rpy_to_R(*j['rpy']), j['xyz'])
            if j['type'] in ('revolute', 'continuous'):
                M = M @ T(axis_angle_R(j['axis'], q.get(jn, 0.0)))
        return (M @ T(p=tcp_offset))[:3, 3]

    def joint_frame(self, jn, q: dict):
        """관절 회전축 위치/방향 (base 기준)"""
        M = np.eye(4)
        for n in self.chain_to(self.joints[jn]['child']):
            j = self.joints[n]
            M = M @ T(rpy_to_R(*j['rpy']), j['xyz'])
            if n == jn:
                return M[:3, 3], M[:3, :3] @ np.asarray(j['axis'], float)
            if j['type'] in ('revolute', 'continuous'):
                M = M @ T(axis_angle_R(j['axis'], q.get(n, 0.0)))


# ------------------------------------------------------------------------- 측정 항목
def pose(**deg):
    return {k: math.radians(v) for k, v in deg.items()}


# 측정 항목
#   kind='dist': 두 자세 사이 TCP 직선 이동 거리 (좌표계 무관)
#   kind='dz'  : 한 자세에서 'TCP 높이 - wrist_flex 축 중심 높이' (직각자로 수직 거리)
#   kind='z'   : 한 자세에서 'TCP 높이 - 베이스 바닥면(base_link 원점) 높이'
TESTS = [
    ('H1', 'dz',   '모든 관절 0°: TCP 높이 - wrist_flex 축 높이', pose(), None),
    ('H2', 'z',    '모든 관절 0°: TCP 높이 (베이스 바닥면 기준*)', pose(), None),
    ('T1', 'dist', 'yaw +90°: wrist_flex -90° → +90°', pose(wrist_yaw=90, wrist_flex=-90),
     pose(wrist_yaw=90, wrist_flex=90)),
    ('T2', 'dist', 'yaw -90°: wrist_flex -90° → +90°', pose(wrist_yaw=-90, wrist_flex=-90),
     pose(wrist_yaw=-90, wrist_flex=90)),
    ('T3', 'dist', 'wrist_yaw -90° → +90° (나머지 0°)', pose(wrist_yaw=-90), pose(wrist_yaw=90)),
    ('T4', 'dist', 'wrist_flex -45° → +45° (나머지 0°)', pose(wrist_flex=-45), pose(wrist_flex=45)),
]


def predict(u: 'Urdf', kind, p1, p2, off):
    if kind == 'dist':
        return 1000 * np.linalg.norm(u.fk(p2, tcp_offset=off) - u.fk(p1, tcp_offset=off))
    tcp_z = u.fk(p1, tcp_offset=off)[2]
    if kind == 'dz':
        return 1000 * (tcp_z - u.joint_frame('wrist_flex', p1)[0][2])
    return 1000 * tcp_z


def describe_dir(v):
    """base 기준 이동 방향을 말로 (x: 로봇 정면, y: 로봇 왼쪽, z: 위) — URDF base_link 기준"""
    names = [('정면(+x)', '뒤(-x)'), ('왼쪽(+y)', '오른쪽(-y)'), ('위(+z)', '아래(-z)')]
    i = int(np.argmax(np.abs(v)))
    return names[i][0 if v[i] > 0 else 1]


def main():
    for st in (sys.stdout, sys.stderr):  # Windows 콘솔 한글 깨짐 방지
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 저장소 루트 (이 파일은 tools/)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_urdf = next((c for c in (os.path.join(here, 'weld_teleop', 'urdf', 'so102.urdf'),
                                     os.path.join(here, 'so102.urdf')) if os.path.exists(c)),
                        os.path.join(here, 'weld_teleop', 'urdf', 'so102.urdf'))
    ap.add_argument('--urdf', default=default_urdf)
    ap.add_argument('--tcp-offset-mm', type=float, nargs=3, default=[0, 0, 0],
                    help='측정점이 gripper_frame_link와 다를 때, gripper_frame_link 기준 오프셋 [mm]')
    ap.add_argument('--measure', action='store_true', help='실측값을 입력받아 판정')
    ap.add_argument('--tol-mm', type=float, default=3.0, help='자 측정 허용 오차 [mm]')
    args = ap.parse_args()

    if not os.path.exists(args.urdf):
        sys.exit(f'URDF를 찾을 수 없음: {args.urdf}')
    base = Urdf(args.urdf)
    z0 = base.joints[YAW_TO_ROLL_JOINT]['xyz'][2]
    hyp = {'A (+%.4f, 현재 URDF)' % abs(z0) if z0 >= 0 else 'A (%.4f, 현재 URDF)' % z0: Urdf(args.urdf),
           'B (%+.4f, 부호 반전)' % -z0: Urdf(args.urdf, yaw_roll_z_override=-z0)}
    off = np.asarray(args.tcp_offset_mm) / 1000.0

    print(f'URDF: {args.urdf}')
    print(f'측정점: {EE} + offset {args.tcp_offset_mm} mm  (기본 = 고정 조 끝)\n')

    # --- 0. 리밋 확인
    for tid, _, _, p1, p2 in TESTS:
        for qq in (p1, p2) if p2 is not None else (p1,):
            for jn, v in qq.items():
                lo, hi = base.joints[jn]['limits']
                if not lo - 1e-6 <= v <= hi + 1e-6:
                    print(f'  [주의] {tid}: {jn}={math.degrees(v):.0f}° 가 URDF 리밋 밖')

    # --- 1. 예측값
    print('=' * 78)
    print(' 1) 측정 항목별 예측 [mm]  (dist=두 자세 간 직선거리, dz/z=수직 높이차)')
    print('=' * 78)
    print(f'{"ID":<4}{"종류":<6}{"동작":<40}' + ''.join(f'{k[:1]:>9}' for k in hyp) + f'{"|A-B|":>9}')
    pred = {}
    for tid, kind, desc, p1, p2 in TESTS:
        row = {k: predict(u, kind, p1, p2, off) for k, u in hyp.items()}
        pred[tid] = row
        a, b = row.values()
        mark = '  ← 판별력 높음' if abs(a - b) >= 2 * args.tol_mm else ('  (부호와 무관: 길이 검증용)' if abs(a - b) < 0.5 else '')
        print(f'{tid:<4}{kind:<6}{desc:<40}' + ''.join(f'{v:9.1f}' for v in row.values()) + f'{abs(a - b):9.1f}{mark}')
    for k in hyp:
        print(f'  {k[:1]} = 가설 {k}')

    # --- 2. 회전 방향(가정 4)
    print('\n' + '=' * 78)
    print(' 2) wrist_yaw 회전 방향 확인 (가정 4)')
    print('=' * 78)
    u = list(hyp.values())[0]
    d = u.fk(pose(wrist_yaw=30), tcp_offset=off) - u.fk(pose(), tcp_offset=off)
    pos, ax = u.joint_frame('wrist_yaw', pose())
    print(f'  모든 관절 0°에서 wrist_yaw 축 방향(base 기준): {np.round(ax, 3)}')
    print(f'  wrist_yaw 를 +30° (lerobot 값이 증가하는 방향) 돌리면 그리퍼 끝이 '
          f'"{describe_dir(d)}" 쪽으로 약 {1000 * np.linalg.norm(d):.0f} mm 이동해야 함')
    print('  → 실제로 반대쪽으로 가면: yaml signs_f 의 wrist_yaw 를 -1.0 으로 (URDF는 그대로)')

    # --- 3. 측정 절차
    print('\n' + '=' * 78)
    print(' 3) 측정 절차')
    print('=' * 78)
    print('''  준비: 리더 팔을 책상에 고정, lerobot으로 관절값을 실시간 출력(도 단위)해 두고
        목표 각도 ±2° 이내로 손으로 맞춘다. 측정점 = 고정 조(움직이지 않는 턱) 끝.
        리더 암(권총 손잡이)에는 고정 조가 없으므로 손잡이 위 점을 쓰고 --tcp-offset-mm 지정.
        가정 1은 2026-10-06 roll 축 높이(손잡이 표시점의 roll 0°/180° 높이 평균)로 확정됨.
  H1 (가장 판별력 큼): 모든 관절 0° 자세에서 wrist_yaw 축이 수직이 됨.
        직각자를 책상에 세워 wrist_flex 서보 혼 중심 높이와 측정점 높이를 각각 재고 뺀다
        (측정점이 더 낮으면 음수로 입력).
  H2:   같은 자세에서 측정점의 높이를 베이스 바닥면(책상 면) 기준으로 잰다.
        shoulder/elbow 캘리브레이션 오차가 섞이므로 H1의 보조 확인용.
        (*base_link 원점 = 베이스 바닥면이라는 가정. 아니면 H2는 무시)
  T1~T4 (dist): 자세1에서 측정점 위치를 표시 → 자세2에서 다시 표시 → 두 점 사이 직선거리.
        높이가 다르면 직각자로 높이차도 재서 √(수평² + 높이²) 로 계산.
        T3은 부호와 무관 → 0.056 m 링크 길이 검증용.''')

    if not args.measure:
        print('\n실측 후:  python tools/so102_urdf_check.py --measure')
        return

    # --- 4. 판정
    print('\n' + '=' * 78)
    print(' 4) 실측값 입력 (Enter = 건너뛰기)')
    print('=' * 78)
    err = {k: [] for k in hyp}
    for tid, kind, desc, _, _ in TESTS:
        s = input(f'  {tid} {desc} 실측 [mm]: ').strip()
        if not s:
            continue
        try:
            m = float(re.sub(r'[^0-9.\-]', '', s))
        except ValueError:
            print('    숫자가 아님 → 건너뜀')
            continue
        line = []
        for k in hyp:
            e = m - pred[tid][k]
            err[k].append(e)
            line.append(f'{k[:1]} 오차 {e:+6.1f}')
        print('    ' + ' | '.join(line))

    n = len(next(iter(err.values())))
    if n == 0:
        print('\n입력된 값이 없음.')
        return
    print('\n  결과 (RMS 오차)')
    rms = {k: math.sqrt(sum(e * e for e in v) / n) for k, v in err.items()}
    for k, v in rms.items():
        print(f'    가설 {k}: {v:5.1f} mm')
    (ka, va), (kb, vb) = sorted(rms.items(), key=lambda kv: kv[1])
    if vb - va >= 2 * args.tol_mm:
        print(f'\n  ▶ 가설 {ka} 채택 (차이 {vb - va:.1f} mm).')
        if va > 2 * args.tol_mm:
            print(f'    단, 채택 가설도 오차 {va:.1f} mm — 부호 외에 링크 길이(0.056/0.0181), '
                  '각도 맞춤, 측정점 위치(--tcp-offset-mm)도 확인 권장.')
        if ka.startswith('B'):
            print(f'    so102.urdf 의 {YAW_TO_ROLL_JOINT} origin z 를 {-z0:+.4f} 로 수정하세요 '
                  f'(weld_teleop/urdf/so102.urdf 도 함께).')
        else:
            print('    현재 URDF 유지.')
    else:
        print('\n  ▶ 판별 불충분 — 판별력 높은 항목(H1, T1, T2)을 더 측정하거나 측정을 반복하세요.')


if __name__ == '__main__':
    main()
