#!/usr/bin/env python3
"""리더 암(SO-102) → Isaac Sim opti3 텔레옵 (ROS 없이, Windows/Linux)

리더 서보 읽기 → 리더 FK → 클러치·상대 매핑 → opti3 IK → 관절 목표를 UDP로 Isaac Sim에 보낸다.
Isaac Sim 쪽은 isaac_opti3_receiver.py가 받아서 /World/indy 관절에 넣고, 실측 관절값을 돌려보낸다.
config는 ROS 노드와 같은 파일(weld_teleop/config/*.yaml)을 쓴다.

사용법 (저장소 루트에서, pinocchio가 설치된 환경)
  python sim_bridge/leader_to_sim.py --port COM3 --follower-urdf C:/.../opti3.urdf
  python sim_bridge/leader_to_sim.py --mock --follower-urdf ...     # 리더 없이 사인파로 확인

조작: Enter = 클러치 ON/OFF (시작은 OFF). 클러치 ON 순간의 자세를 기준으로 상대 이동만 따라간다.
     Ctrl+C = 종료
"""
from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, 'weld_teleop')
sys.path.insert(0, PKG)

from weld_teleop.leader_drivers import FeetechDriver, MockDriver  # noqa: E402
from weld_teleop.teleop_loop import build_loop, decode, encode, ros_params  # noqa: E402


def cfg(name: str) -> str:
    return os.path.join(PKG, 'config', name)


def addr(s: str) -> tuple[str, int]:
    host, port = s.rsplit(':', 1)
    return host, int(port)


def main():
    for st in (sys.stdout, sys.stderr):  # Windows 콘솔 한글 깨짐 방지
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', help='리더 시리얼 포트 (예: COM3)')
    ap.add_argument('--mock', action='store_true', help='리더 없이 사인파 입력')
    ap.add_argument('--follower-urdf', required=True, help='opti3.urdf 경로')
    ap.add_argument('--leader-urdf', default=os.path.join(PKG, 'urdf', 'so102.urdf'))
    ap.add_argument('--leader-config', default=cfg('leader_so101_7axis.yaml'))
    ap.add_argument('--follower-config', default=cfg('follower_opti3.yaml'))
    ap.add_argument('--send', default='127.0.0.1:9870', help='Isaac Sim 수신 주소')
    ap.add_argument('--listen', default='127.0.0.1:9871', help='실측 관절값 수신 주소')
    ap.add_argument('--rate', type=float, default=100.0)
    args = ap.parse_args()
    if not args.mock and not args.port:
        ap.error('--port 또는 --mock 중 하나가 필요합니다')

    loop = build_loop(args.leader_config, cfg('mapping.yaml'), args.follower_config,
                      args.leader_urdf, args.follower_urdf)
    n_follower = loop.follower_kin.nq

    st = ros_params(args.leader_config, 'leader_state_node')
    aux_id = int(st['aux_servo_id'])
    if args.mock:
        driver = MockDriver(len(st['joint_names']), has_aux=aux_id >= 0)
    else:
        driver = FeetechDriver(args.port, int(st['baudrate']), list(st['servo_ids']),
                               list(st['offsets']), list(st['signs']),
                               aux_id=aux_id if aux_id >= 0 else None,
                               aux_range=tuple(st['aux_range']))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(addr(args.listen))
    sock.setblocking(False)
    dest = addr(args.send)

    clutch = threading.Event()

    def toggle_on_enter():
        for _ in sys.stdin:
            if clutch.is_set():
                clutch.clear()
            else:
                clutch.set()
    threading.Thread(target=toggle_on_enter, daemon=True).start()

    print(f'리더: {"mock" if args.mock else args.port} → Isaac Sim {args.send}')
    print('Isaac Sim 실측값을 기다리는 중… Enter = 클러치 ON/OFF, Ctrl+C = 종료')
    dt = 1.0 / args.rate
    t_next = time.monotonic()
    t_print = 0.0
    aux = None
    try:
        while True:
            now = time.monotonic()
            while True:   # 쌓인 실측 패킷 중 마지막 것만 사용
                try:
                    data, _ = sock.recvfrom(65536)
                except (BlockingIOError, ConnectionResetError):
                    break
                msg = decode(data, n_follower)
                if msg is not None:
                    if loop.stepper.q_meas is None:   # 첫 실측: 자세 확인용 출력
                        tip = loop.follower_kin.fk(msg['q']).translation
                        print(f'\nIsaac Sim 첫 실측 [deg]: {[round(math.degrees(float(v)), 1) for v in msg["q"]]}'
                              f'  토치 팁 [m]: {[round(float(v), 3) for v in tip]}')
                        out = loop.stepper.out_of_limits(msg['q'])
                        if out:
                            print(f'경고: URDF 리밋 밖 관절 {out} → 장면과 URDF의 관절 영점/범위 확인 필요')
                    loop.on_measured(msg['q'], now)
            try:
                q_leader, aux = driver.read()
            except (IOError, KeyError):
                q_leader = None
            q = loop.tick(now, q_leader, clutch.is_set())
            if q is not None:
                sock.sendto(encode(q), dest)

            if now - t_print > 0.1:
                g = f'{aux * 100:5.1f}%' if aux is not None else '  -  '
                print(f'\r클러치 {"ON " if clutch.is_set() else "OFF"} | 그리퍼 {g} | {loop.status:<32}',
                      end='', flush=True)
                t_print = now
            t_next = max(t_next + dt, time.monotonic() - dt)   # 밀리면 따라잡지 않고 버림
            time.sleep(max(0.0, t_next - time.monotonic()))
    except KeyboardInterrupt:
        print()
    finally:
        driver.close()
        sock.close()


if __name__ == '__main__':
    main()
