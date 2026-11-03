#!/usr/bin/env python3
"""규칙 전문가 → Isaac Sim opti3: 모재 오차 시나리오 하나로 에피소드 1개를 만들고 정답을 기록한다.

규칙 전문가(sim_errors/expert.py)가 실제 이음매를 따라 토치 목표를 만들고, opti3 IK(FollowerStepper)로
관절 목표를 계산해 isaac_opti3_receiver.py에 보낸다. 패킷에 아크 진행도(s_arc)와 모재 배치를 함께 보내므로
receiver(--scenario 같은 파일)가 모재 변형을 같은 진행도로 그린다.

사용법 (저장소 루트에서, pinocchio가 설치된 conda 환경)
  python -m sim_errors.make_scenarios --n 20 --out scenarios
  터미널 1: C:\\isaacsim\\python.bat sim_bridge\\isaac_opti3_receiver.py --usd ...\\posco_torch.usd --scenario scenarios\\part0000_pass0.json
  터미널 2: python sim_bridge/expert_to_sim.py --scenario scenarios/part0000_pass0.json --follower-urdf C:/.../opti3.urdf

- 모재 배치: 기본은 시작 자세의 토치 팁 근처에 자동 배치(이음매 방향 = 베이스 -Y). 시나리오의 nominal_pose를
  쓰려면 --use-scenario-pose.
- 시작 전에 이음매 전체에 대해 IK 도달 여부를 확인하고, 안 되면 시작하지 않는다.
- 기록: --out(기본 episodes/<시나리오 이름>.npz). 정답(e, rotvec, 틈새, 위빙, 속도), 레이저 측정, 목표 포즈,
  관절 명령·실측, 실측 툴 포즈. 각 필드 설명은 sim_errors/README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, 'weld_teleop')
for p in (REPO, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

import pinocchio as pin  # noqa: E402

from sim_errors.expert import ScriptedExpert, auto_nominal  # noqa: E402
from sim_errors.part_errors import PartErrorModel, load_scenario  # noqa: E402
from weld_teleop.follower_control import FollowerStepper  # noqa: E402
from weld_teleop.kinematics import ChainKinematics, se3_from_xyz_rpy, se3_to_xyzquat  # noqa: E402
from weld_teleop.teleop_loop import decode, encode, ros_params  # noqa: E402


def addr(s: str) -> tuple[str, int]:
    host, port = s.rsplit(':', 1)
    return host, int(port)


def build_stepper(follower_yaml: str, follower_urdf: str) -> FollowerStepper:
    fo = ros_params(follower_yaml, 'follower_ik_node')
    kin = ChainKinematics(follower_urdf, list(fo['joint_names']), fo['ee_frame'],
                          se3_from_xyz_rpy(fo['tool_xyz'], fo['tool_rpy']))
    return FollowerStepper(kin, max_joint_step=fo['max_joint_step'], ik_tolerance=fo['ik_tolerance'],
                           max_track_error=fo['max_track_error'], ik_accept_error=fo.get('ik_accept_error'),
                           input_timeout=fo['input_timeout'])


def to_se3(T: np.ndarray) -> pin.SE3:
    return pin.SE3(np.asarray(T[:3, :3]), np.asarray(T[:3, 3]))


def check_reach(kin: ChainKinematics, expert: ScriptedExpert, q0: np.ndarray, tol: float, n: int = 21):
    """이음매 전체(가장 변형된 상태 s_arc=s)에 IK가 닿는지 미리 확인. 반환: 실패한 s 목록."""
    q = q0.copy()
    bad = []
    lift = expert.ex['standoff_mm'] * 1e-3
    for s in np.linspace(0, 1, n):
        q_sol, ok, err = kin.ik(to_se3(expert.seam_pose(s, s, lift)), q, iters=100)
        if ok or err <= tol:
            q = q_sol
        else:
            bad.append((round(float(s), 2), round(err, 4)))
    return bad


class Recorder:
    def __init__(self):
        self.rows: dict[str, list] = {}

    def add(self, **kw):
        for k, v in kw.items():
            self.rows.setdefault(k, []).append(v)

    def save(self, path: str, scn: dict):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        out = {}
        for k, v in self.rows.items():
            out[k] = np.array(v) if k != 'phase' else np.array(v, dtype='U8')
        np.savez_compressed(path, scenario_json=json.dumps(scn), **out)


def main():
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', required=True, help='make_scenarios.py가 만든 JSON')
    ap.add_argument('--follower-urdf', required=True, help='opti3.urdf 경로')
    ap.add_argument('--follower-config', default=os.path.join(PKG, 'config', 'follower_opti3.yaml'))
    ap.add_argument('--use-scenario-pose', action='store_true', help='자동 배치 대신 시나리오의 nominal_pose 사용')
    ap.add_argument('--send', default='127.0.0.1:9870')
    ap.add_argument('--listen', default='127.0.0.1:9871')
    ap.add_argument('--rate', type=float, default=100.0)
    ap.add_argument('--sensor-seed', type=int, default=0, help='레이저 노이즈 난수 시드')
    ap.add_argument('--out', help='기록 파일 (.npz). 기본 episodes/<시나리오 이름>.npz')
    args = ap.parse_args()

    scn = load_scenario(args.scenario)
    model = PartErrorModel(scn)
    stepper = build_stepper(args.follower_config, args.follower_urdf)
    kin = stepper.kin
    out_path = args.out or os.path.join('episodes', os.path.splitext(os.path.basename(args.scenario))[0] + '.npz')

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(addr(args.listen))
    sock.setblocking(False)
    dest = addr(args.send)

    print(f'시나리오 {args.scenario} (패스 {scn.get("pass_index", 0)}) → Isaac Sim {args.send}')
    print('Isaac Sim 실측값을 기다리는 중… (receiver를 같은 --scenario로 실행)')
    expert = None
    rec = Recorder()
    dt = 1.0 / args.rate
    t_next = time.monotonic()
    t_print = 0.0
    try:
        while True:
            now = time.monotonic()
            while True:
                try:
                    data, _ = sock.recvfrom(65536)
                except (BlockingIOError, ConnectionResetError):
                    break
                msg = decode(data, kin.nq)
                if msg is not None:
                    stepper.on_measured(msg['q'], now)

            if expert is None and stepper.q_meas is not None:
                T0 = kin.fk(stepper.q_meas).homogeneous
                if not args.use_scenario_pose:
                    model.set_nominal(auto_nominal(T0, model.L))
                expert = ScriptedExpert(model, T0, np.random.default_rng(args.sensor_seed))
                print(f'시작 토치 팁 [m]: {np.round(T0[:3, 3], 3).tolist()}')
                bad = check_reach(kin, expert, stepper.q_meas, stepper.accept)
                if bad:
                    print(f'IK 도달 실패 (s, 오차): {bad}\n모재 배치(nominal_pose 또는 시작 자세)를 바꿔 다시 실행하세요.')
                    return
                print('이음매 전체 IK 도달 확인. 시작합니다.')

            if expert is not None:
                r = expert.step(dt)
                stepper.on_target(to_se3(r['T_target']), now)
                q = stepper.step(now)
                scene = {'s_arc': r['s_arc'], 'T_nominal': model.scn['T_nominal'], 'phase': r['phase']}
                if q is not None:
                    sock.sendto(encode(q, target=se3_to_xyzquat(to_se3(r['T_target'])).tolist(), scene=scene), dest)
                T_meas = kin.fk(stepper.q_meas).homogeneous
                rec.add(t=r['t'], phase=r['phase'], arc_on=r['arc_on'], done=r['done'], s=r['s'], s_arc=r['s_arc'],
                        e=r['e'], rotvec=r['rotvec'], gap=r['gap'], tack=r['tack'],
                        weave_amp=r['weave_amp'], weave_hz=r['weave_hz'], speed=r['speed'], laser=r['laser'],
                        p_true=r['p_true'], p_est=r['p_est'], T_target=r['T_target'], T_meas=T_meas,
                        q_cmd=q if q is not None else np.full(kin.nq, np.nan), q_meas=stepper.q_meas.copy(),
                        status=stepper.status)
                if stepper.status == 'ik_failed':
                    print(f'\nIK 실패로 중단 ({stepper.detail}), 진행도 {r["s"]:.3f}')
                    break
                if expert.done:
                    print('\n에피소드 완료')
                    break
                if now - t_print > 0.2:
                    err = np.linalg.norm(T_meas[:3, 3] - r['T_target'][:3, 3]) * 1e3
                    print(f'\r{r["phase"]:<8} s {r["s"]:.3f} | 좌우 {r["e"][1] * 1e3:+5.2f} 높이 {r["e"][2] * 1e3:+5.2f} mm'
                          f' | 틈새 {r["gap"] * 1e3:4.2f} mm | 추종 오차 {err:5.1f} mm | {stepper.status:<10}',
                          end='', flush=True)
                    t_print = now
            t_next = max(t_next + dt, time.monotonic() - dt)
            time.sleep(max(0.0, t_next - time.monotonic()))
    except KeyboardInterrupt:
        print('\n중단')
    finally:
        sock.close()
        if rec.rows:
            rec.save(out_path, model.scn)
            print(f'기록: {os.path.abspath(out_path)} ({len(rec.rows["t"])} 주기)')


if __name__ == '__main__':
    main()
