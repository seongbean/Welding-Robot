#!/usr/bin/env python3
"""SO-102 리더암 서보 모니터 — 서보 ID가 어느 관절인지 확인용 (읽기 전용)

원리
  - 버스의 서보를 스캔한 뒤, 모든 서보의 현재 위치를 계속 읽어 표로 보여준다.
  - 관절을 하나씩 손으로 움직이면 값이 바뀌는 ID 줄에 '◀ 움직이는 중'이 뜬다.
  - 서보에는 읽기 명령(ping, Torque_Enable/Present_Position 읽기)만 보내고 아무것도 쓰지 않는다.

의존성: feetech-servo-sdk (scservo_sdk) — lerobot conda 환경에 설치되어 있음

사용법
  conda activate lerobot
  python so102_servo_monitor.py                       # /dev/ttyACM0, 1 Mbps, ID 0~20 스캔
  python so102_servo_monitor.py --port /dev/ttyACM1 --max-id 30
  실행 중: Enter = 기준 초기화(변화량을 0으로), Ctrl+C = 종료
"""
from __future__ import annotations

import argparse
import collections
import os
import select
import sys
import time
import unicodedata

ADDR_TORQUE_ENABLE = 40      # STS3215 컨트롤 테이블 주소
ADDR_PRESENT_POSITION = 56   # 2 bytes
TICKS_PER_REV = 4096
MODELS = {777: 'STS3215'}
JOINT_ORDER = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper']

MOVE_WINDOW_S = 0.3   # 최근 이 시간 동안
MOVE_DEG = 1.5        # 이 각도 이상 바뀌었으면 '움직이는 중'


# ------------------------------------------------------------------------- 위치 추적
def to_deg(ticks):
    return ticks * 360.0 / TICKS_PER_REV


def decode_position(raw):
    """Present_Position의 bit 15 = 부호 (sign-magnitude, lerobot과 같은 해석)"""
    return -(raw & 0x7FFF) if raw & 0x8000 else raw


class Track:
    """서보 하나의 위치 이력. 0↔4095 경계를 넘어도 끊기지 않게 펼친(unwrap) 위치로 누적"""

    def __init__(self, raw, now):
        self.raw = raw
        self.pos = float(raw)   # 펼친 위치 [tick]
        self.hist = collections.deque([(now, self.pos)])
        self.reset()

    def reset(self):
        self.start = self.lo = self.hi = self.pos

    def update(self, raw, now):
        half = TICKS_PER_REV // 2
        self.pos += (raw - self.raw + half) % TICKS_PER_REV - half
        self.raw = raw
        self.lo, self.hi = min(self.lo, self.pos), max(self.hi, self.pos)
        self.hist.append((now, self.pos))
        while now - self.hist[0][0] > MOVE_WINDOW_S:
            self.hist.popleft()

    def delta_deg(self):
        return to_deg(self.pos - self.start)

    def range_deg(self):
        return to_deg(self.hi - self.lo)

    def recent_deg(self):
        """최근 MOVE_WINDOW_S 동안 움직인 폭"""
        ps = [p for _, p in self.hist]
        return to_deg(max(ps) - min(ps))


# ----------------------------------------------------------------------------- 화면
def text_width(s):
    """터미널 표시 폭 (한글 등 전각 문자 = 2칸)"""
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)


def rjust(s, w):
    return ' ' * max(0, w - text_width(s)) + s


COLS = [('ID', 3), ('모델', 8), ('현재[tick]', 11), ('기준 대비[°]', 13), ('움직인 범위[°]', 15)]


def render(found, tracks, failed, torque_on, ansi):
    lines = ['SO-102 서보 모니터 (읽기 전용)    Enter: 기준 초기화    Ctrl+C: 종료',
             '관절을 하나씩 크게 움직여 보세요 (베이스 → 끝):',
             '  ' + ', '.join(JOINT_ORDER)]
    if torque_on:
        lines.append(f'[주의] 토크 ON: ID {", ".join(map(str, torque_on))} — 손으로 잘 안 움직일 수 있음 '
                     '(이 도구는 토크를 바꾸지 않음)')
    lines += ['', '  '.join(rjust(name, w) for name, w in COLS) + '  움직임']
    for sid, model in found.items():
        t = tracks.get(sid)
        if t is None:  # 아직 한 번도 못 읽음
            lines.append(rjust(str(sid), COLS[0][1]) + '  읽기 실패')
            continue
        recent = t.recent_deg()
        moving = sid not in failed and recent >= MOVE_DEG
        cells = [str(sid), MODELS.get(model, str(model)), str(t.raw), f'{t.delta_deg():+.1f}', f'{t.range_deg():.1f}']
        row = '  '.join(rjust(c, w) for c, (_, w) in zip(cells, COLS)) + '  '
        if sid in failed:
            row += '읽기 실패'
        else:
            row += '█' * min(20, int(recent / 2)) + (' ◀ 움직이는 중' if moving else '')
        if moving and ansi:
            row = f'\x1b[1;32m{row}\x1b[0m'
        lines.append(row)
    return lines


def enter_pressed():
    """Enter가 눌렸으면 True (리눅스/맥 터미널에서만 지원)"""
    if os.name == 'nt' or not sys.stdin.isatty():
        return False
    if not select.select([sys.stdin], [], [], 0)[0]:
        return False
    return sys.stdin.readline() != ''  # '' = EOF(Ctrl+D)


def main():
    for st in (sys.stdout, sys.stderr):  # Windows 콘솔 한글 깨짐 방지
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--baud', type=int, default=1_000_000)
    ap.add_argument('--max-id', type=int, default=20, help='ID 0 ~ max-id 를 스캔')
    args = ap.parse_args()

    try:
        from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler
    except ImportError:
        sys.exit('scservo_sdk 없음 → conda activate lerobot 후 실행 (또는 pip install feetech-servo-sdk)')

    port = PortHandler(args.port)
    try:
        opened = port.openPort() and port.setBaudRate(args.baud)
    except Exception as e:  # 포트 없음, 권한 없음 등 (pyserial 예외)
        sys.exit(f'포트 열기 실패: {args.port} ({e})')
    if not opened:
        sys.exit(f'포트 열기 실패: {args.port}')
    pkt = PacketHandler(0)  # STS/SMS 계열 = little endian

    def read_pos(sid):
        raw, res, _ = pkt.read2ByteTxRx(port, sid, ADDR_PRESENT_POSITION)
        return decode_position(raw) if res == COMM_SUCCESS else None

    try:
        print(f'{args.port} @ {args.baud} bps, ID 0~{args.max_id} 스캔 중...')
        found = {}
        for sid in range(args.max_id + 1):
            model, res, _ = pkt.ping(port, sid)
            if res == COMM_SUCCESS:
                found[sid] = model
        if not found:
            sys.exit('응답한 서보가 없음 → 포트, 보레이트, 서보 전원을 확인하세요.')
        torque_on = []
        for sid in found:
            val, res, _ = pkt.read1ByteTxRx(port, sid, ADDR_TORQUE_ENABLE)
            if res == COMM_SUCCESS and val:
                torque_on.append(sid)

        ansi = sys.stdout.isatty()  # 터미널이면 제자리 갱신, 아니면(파이프 등) 1초마다 출력
        if ansi:
            sys.stdout.write('\x1b[2J')
        tracks, last_print = {}, 0.0
        while True:
            now = time.monotonic()
            failed = set()
            for sid in found:
                raw = read_pos(sid)
                if raw is None:
                    failed.add(sid)
                elif sid in tracks:
                    tracks[sid].update(raw, now)
                else:
                    tracks[sid] = Track(raw, now)
            if enter_pressed():
                for t in tracks.values():
                    t.reset()
            lines = render(found, tracks, failed, torque_on, ansi)
            if ansi:
                sys.stdout.write('\x1b[H' + ''.join(line + '\x1b[K\n' for line in lines) + '\x1b[J')
                sys.stdout.flush()
            elif now - last_print >= 1.0:
                print('\n'.join(lines) + '\n', flush=True)
                last_print = now
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('\n종료')
    except OSError as e:  # pyserial SerialException 포함
        sys.exit(f'\n포트 통신 오류: {e}\n'
                 '→ 같은 포트를 쓰는 다른 프로그램(leader_monitor.py, lerobot, ROS 노드 등)을 먼저 종료하세요.')
    finally:
        port.closePort()


if __name__ == '__main__':
    main()
