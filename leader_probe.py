#!/usr/bin/env python3
"""리더 암(SO-102) 서보 확인 도구: 버스 스캔 · 실시간 관절값 · 영점 tick 기록

ROS2 없이 실행 (Windows/Linux). 필요: pip install feetech-servo-sdk

사용법
  python leader_probe.py --port COM5 scan          # 응답하는 서보 ID와 현재 위치
  python leader_probe.py --port COM5 watch         # 실시간 값. 관절을 하나씩 움직여 ID↔관절 대응 확인
  python leader_probe.py --port COM5 snapshot      # 지금 자세의 tick 값을 config용 offsets 줄로 출력
  (Linux 포트 예: /dev/ttyACM0)

STS3215 기준: 1회전 = 4096 tick, 중앙 = 2048. watch/snapshot의 각도는 (tick - 2048) 기준이며
캘리브레이션 전 참고값이다.
"""
from __future__ import annotations

import argparse
import sys
import time

ADDR_PRESENT_POSITION = 56   # STS3215
TICKS_PER_REV = 4096
CENTER = 2048


def ticks_to_deg(ticks: int, center: int = CENTER) -> float:
    return (ticks - center) * 360.0 / TICKS_PER_REV


def scan(port, pkt, ids, comm_success: int = 0) -> dict[int, tuple[int, int]]:
    """응답한 ID → (모델 번호, 현재 tick)."""
    found = {}
    for sid in ids:
        model, comm, err = pkt.ping(port, sid)
        if comm != comm_success:
            continue
        pos, comm, err = pkt.read2ByteTxRx(port, sid, ADDR_PRESENT_POSITION)
        found[sid] = (model, pos if comm == comm_success else -1)
    return found


def read_positions(port, pkt, ids, comm_success: int = 0) -> dict[int, int]:
    out = {}
    for sid in ids:
        pos, comm, err = pkt.read2ByteTxRx(port, sid, ADDR_PRESENT_POSITION)
        out[sid] = pos if comm == comm_success else -1
    return out


def most_moved(prev: dict[int, int], cur: dict[int, int], min_ticks: int = 20) -> int | None:
    """직전 대비 가장 많이 움직인 ID (min_ticks 미만이면 None)."""
    best, best_d = None, min_ticks
    for sid, v in cur.items():
        if v < 0 or prev.get(sid, -1) < 0:
            continue
        d = abs(v - prev[sid])
        if d >= best_d:
            best, best_d = sid, d
    return best


def open_bus(port_name: str, baud: int):
    try:
        from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler
    except ImportError:
        sys.exit('scservo_sdk 없음: pip install feetech-servo-sdk')
    port = PortHandler(port_name)
    if not port.openPort():
        sys.exit(f'포트 열기 실패: {port_name} (장치 관리자/ls /dev/tty* 로 포트 확인)')
    if not port.setBaudRate(baud):
        sys.exit(f'보레이트 설정 실패: {baud}')
    return port, PacketHandler(0), COMM_SUCCESS


def main():
    for st in (sys.stdout, sys.stderr):  # Windows 콘솔 한글 깨짐 방지
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', required=True)
    ap.add_argument('--baud', type=int, default=1000000)
    ap.add_argument('--ids', default='0-20', help='스캔할 ID 범위 (예: 0-20 또는 1,2,3)')
    ap.add_argument('mode', choices=['scan', 'watch', 'snapshot'])
    args = ap.parse_args()

    if '-' in args.ids:
        a, b = map(int, args.ids.split('-'))
        ids = list(range(a, b + 1))
    else:
        ids = [int(x) for x in args.ids.split(',')]

    port, pkt, ok = open_bus(args.port, args.baud)
    try:
        found = scan(port, pkt, ids, ok)
        if not found:
            sys.exit('응답한 서보 없음: 전원, 포트, 보레이트(--baud) 확인')
        sids = sorted(found)

        if args.mode == 'scan':
            print(f'응답한 서보 {len(sids)}개')
            print(f'{"ID":>4}{"모델":>8}{"tick":>8}{"deg":>9}')
            for sid in sids:
                model, pos = found[sid]
                print(f'{sid:>4}{model:>8}{pos:>8}{ticks_to_deg(pos):>9.1f}')
            print('\n다음: watch 모드에서 관절을 하나씩 움직여 각 ID가 어느 관절인지 확인하세요.')

        elif args.mode == 'snapshot':
            pos = read_positions(port, pkt, sids, ok)
            print('# 지금 자세의 tick (URDF 영점 자세에서 실행하면 그대로 offsets)')
            print(f'servo_ids: {sids}')
            print(f'offsets: {[pos[s] for s in sids]}')

        else:
            print('Ctrl+C로 종료. 움직인 서보 ID에 * 표시.')
            prev = read_positions(port, pkt, sids, ok)
            while True:
                cur = read_positions(port, pkt, sids, ok)
                moved = most_moved(prev, cur)
                line = '  '.join(f'{"*" if s == moved else " "}{s}:{ticks_to_deg(cur[s]):7.1f}' for s in sids)
                print('\r' + line, end='', flush=True)
                prev = cur
                time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        port.closePort()


if __name__ == '__main__':
    main()
