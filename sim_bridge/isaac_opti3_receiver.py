"""Isaac Sim 쪽: UDP로 받은 관절 목표를 opti3(/World/indy)에 넣고 실측 관절값을 돌려보낸다.

Isaac Sim 5.1 standalone 스크립트. Isaac Sim에 들어 있는 파이썬으로 실행한다(추가 설치 없음).
  C:\\isaacsim\\python.bat sim_bridge\\isaac_opti3_receiver.py --usd C:\\...\\opti3_welding_scene\\posco_torch.usd

- 장면 README대로 관절 drive 게인을 kp=1e4, kd=1e3 으로 올린다(그냥 Play하면 중력에 처짐).
- 시작 자세(--initial-q)는 토치가 로봇 앞에서 아래를 향하는 자세(관절 0, -30, 100, 0, 50, 0도).
  follower_opti3.yaml의 initial_q(토치 팁이 로봇 뒤 바닥 근처)보다 사방으로 움직일 여유가 크다.
  목표가 안 오면 마지막 목표를 유지한다.
- 리더 쪽은 leader_to_sim.py. 종료는 창 닫기 또는 Ctrl+C.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--usd', required=True, help='posco_torch.usd 경로 (폴더 구조 그대로)')
ap.add_argument('--prim', default='/World/indy')
ap.add_argument('--joints', default='joint0,joint1,joint2,joint3,joint4,joint5')
ap.add_argument('--initial-q', default='0,-0.5236,1.7453,0,0.8727,0', help='rad, joint0~5 시작 자세')
ap.add_argument('--listen', default='127.0.0.1:9870', help='관절 목표 수신 주소')
ap.add_argument('--reply', default='127.0.0.1:9871', help='실측 관절값 보낼 주소 (leader_to_sim.py)')
ap.add_argument('--kp', type=float, default=1e4)
ap.add_argument('--kd', type=float, default=1e3)
ap.add_argument('--headless', action='store_true')
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': args.headless})

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402


def addr(s: str) -> tuple[str, int]:
    host, port = s.rsplit(':', 1)
    return host, int(port)


JOINTS = args.joints.split(',')
Q0 = np.array([float(x) for x in args.initial_q.split(',')])

if not open_stage(args.usd):
    sys.exit(f'장면 열기 실패: {args.usd}')
world = World(stage_units_in_meters=1.0)
robot = world.scene.add(SingleArticulation(prim_path=args.prim, name='indy'))


def setup() -> tuple[list[int], np.ndarray]:
    """reset 직후 호출: 게인 설정, 시작 자세. (관절 인덱스, 전체 목표 벡터) 반환."""
    world.reset()
    missing = [j for j in JOINTS if j not in robot.dof_names]
    if missing:
        sys.exit(f'{args.prim}에 없는 관절: {missing} (있는 관절: {robot.dof_names})')
    idx = [robot.dof_names.index(j) for j in JOINTS]
    n = robot.num_dof
    robot.get_articulation_controller().set_gains(kps=np.full(n, args.kp), kds=np.full(n, args.kd))
    target = robot.get_joint_positions().copy()
    target[idx] = Q0
    robot.set_joint_positions(target)
    return idx, target


idx, target = setup()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(addr(args.listen))
sock.setblocking(False)
reply = addr(args.reply)
print(f'[receiver] {args.prim} {JOINTS} 대기 중: {args.listen} → 실측 {args.reply}')

n_rx = 0
try:
    while app.is_running():
        if world.is_stopped():           # GUI에서 Stop 후 Play 하면 다시 초기화
            world.step(render=True)
            if world.is_playing():
                idx, target = setup()
            continue
        while True:                      # 쌓인 목표 중 마지막 것만 사용
            try:
                data, _ = sock.recvfrom(65536)
            except (BlockingIOError, ConnectionResetError):
                break
            try:
                q = np.asarray(json.loads(data.decode())['q'], dtype=float)
            except (ValueError, KeyError, TypeError):
                continue
            if q.shape == (len(idx),) and np.all(np.isfinite(q)):
                target[idx] = q
                n_rx += 1
        robot.apply_action(ArticulationAction(joint_positions=target))
        world.step(render=not args.headless)
        if world.is_playing():
            q_meas = robot.get_joint_positions()[idx]
            sock.sendto(json.dumps({'q': q_meas.tolist(), 'rx': n_rx}).encode(), reply)
except KeyboardInterrupt:
    pass
finally:
    sock.close()
    app.close()
