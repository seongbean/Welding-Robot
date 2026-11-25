#!/usr/bin/env python3
"""규칙 전문가 → Isaac Sim opti3: 모재 오차 시나리오 하나로 에피소드 1개를 만들고 정답을 기록한다.

규칙 전문가(sim_errors/expert.py)가 실제 이음매를 따라 토치 목표를 만들고, opti3 IK(FollowerStepper)로
관절 목표를 계산해 isaac_opti3_receiver.py에 보낸다. 패킷에 시나리오 경로, 아크 진행도(s_arc), 모재 배치를
함께 보내므로 receiver가 재시작 없이 모재를 바꾸고 변형을 같은 진행도로 그린다. 여러 개는 run_batch.py.

사용법 (저장소 루트에서, pinocchio가 설치된 conda 환경)
  터미널 1: C:\\isaacsim\\python.bat sim_bridge\\isaac_opti3_receiver.py
  터미널 2: python sim_bridge/expert_to_sim.py --scenario scenarios/part0000_pass0.json --follower-urdf C:/.../opti3.urdf

- 시간: receiver가 보내는 시뮬 시각으로 전문가를 진행한다. 실측 하나에 한 주기라서, Isaac이 멈추면 전문가도
  멈춘다(joint_state_timeout 동안 진행도가 토치보다 앞서가던 문제). 시각이 없는 이전 receiver면 받은 시각 사용.
- 모재 배치: 홈 자세(--home-q, receiver --initial-q와 같아야 함)의 토치 팁 근처에 고정 배치(이음매 방향 = 베이스 -Y).
  시나리오의 nominal_pose를 쓰려면 --use-scenario-pose.
- 시작 전에 이음매 전체에 대해 IK 도달 여부를 확인하고, 안 되면 시작하지 않는다.
- 용접 구간만 담는 시나리오(episode.weld_only, make_scenarios --batch)는 로봇을 이음매 시작점 + 시작 어긋남
  관절 자세로 바로 놓고(reset_q) 자리 잡으면 용접부터 시작한다. 아니면 지금 자세에서 접근부터 한다.
- 기록: --out(기본 outputs/scratch/episodes/<시나리오 이름>.npz). 필드 설명은 sim_errors/README.md.
"""
from __future__ import annotations

import argparse
import os
import select
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

from common import paths  # noqa: E402
from sim_errors.episode import Recorder  # noqa: E402
from sim_errors.expert import ScriptedExpert, isaac_nominal  # noqa: E402
from sim_errors.part_errors import PartErrorModel, load_scenario  # noqa: E402
from weld_teleop.follower_control import FollowerStepper  # noqa: E402
from weld_teleop.kinematics import ChainKinematics, se3_from_xyz_rpy, se3_to_xyzquat  # noqa: E402
from weld_teleop.teleop_loop import decode, encode, ros_params  # noqa: E402

HOME_Q = '0,-0.5236,1.7453,0,0.8727,0'    # isaac_opti3_receiver.py --initial-q 기본값


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


def _walk(kin: ChainKinematics, expert: ScriptedExpert, q: np.ndarray, ss, lift: float, accept: float):
    """이음매를 s 순서로 따라가며 이전 해로 IK 시드. 반환: (관절 (n,6), 오차 (n,))."""
    qs, errs = [], []
    for s in ss:
        q_sol, ok, err = kin.ik(to_se3(expert.seam_pose(s, s, lift)), q, iters=200)
        errs.append(0.0 if ok else err)
        if ok or err <= accept:
            q = q_sol
        qs.append(q.copy())
    return np.array(qs), np.array(errs)


def seam_path_q(kin: ChainKinematics, expert: ScriptedExpert, q0: np.ndarray, accept: float, n: int = 41,
                n_seeds: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """이음매 시작점 IK를 여러 시드(q0 + 고정 난수 섭동)로 풀고, 이음매를 따라갔을 때 도달 못 한 점이 가장 적고
    이웃 점 관절 변화가 0.2 rad 미만인 해 중 q0에 가장 가까운 해를 고른다. 시드 하나(q0)만 쓰면 opti3가 어깨 특이점 근처의
    해로 들어가 이음매 중간에 손목이 뒤집혔다. 반환: (관절 (n,6), 오차 (n,))."""
    lift = expert.ex['standoff_mm'] * 1e-3
    rng = np.random.default_rng(0)
    seeds = [q0] + [np.clip(q0 + rng.uniform(-1.5, 1.5, len(q0)), kin.lower, kin.upper) for _ in range(n_seeds)]
    T0 = to_se3(expert.seam_pose(0.0, 0.0, lift))
    starts = []
    for sd in seeds:
        q, ok, err = kin.ik(T0, sd, iters=300)
        if (ok or err <= accept) and not any(np.abs(q - o).max() < 1e-2 for o in starts):
            starts.append(q)
    if not starts:
        starts = [q0]
    ss = np.linspace(0, 1, n)
    best = None
    for q in starts:
        qs, errs = _walk(kin, expert, q, ss, lift, accept)
        jump = float(np.abs(np.diff(qs, axis=0)).max())
        key = (int((errs > accept).sum()), jump >= 0.2, float(np.abs(qs[0] - q0).max()))   # 매끄러우면 q0에 가까운 해
        if best is None or key < best[0]:
            best = (key, qs, errs)
    return best[1], best[2]


def check_reach(kin: ChainKinematics, expert: ScriptedExpert, q0: np.ndarray, tol: float, n: int = 41):
    """이음매 전체(가장 변형된 상태 s_arc=s)에 IK가 닿는지 미리 확인. 반환: (실패한 s 목록, 시작점 관절 자세)."""
    qs, errs = seam_path_q(kin, expert, q0, tol, n)
    ss = np.linspace(0, 1, n)
    bad = [(round(float(s), 2), round(float(e), 4)) for s, e in zip(ss, errs) if e > tol]
    return bad, (qs[0].copy() if errs[0] <= tol else None)


def reach_scan(kin: ChainKinematics, expert: ScriptedExpert, q0: np.ndarray, tol: float, accept: float,
               n: int = 41) -> dict:
    """Isaac 없이 이음매 전체의 IK 도달만 본다 (run_batch --reach-only). 해 고르기는 실제 실행과 같다(seam_path_q).
    반환: 최대 IK 오차, tol·accept 넘는 점 수, 이웃 점 사이 관절 최대 변화(손목 뒤집힘 확인), 리밋 여유, 시작 자세 IK."""
    qs, errs = seam_path_q(kin, expert, q0, accept, n)
    jump = float(np.abs(np.diff(qs, axis=0)).max()) if len(qs) > 1 else 0.0
    margin = float(np.minimum(qs - kin.lower, kin.upper - qs).min())
    _, ok0, err0 = kin.ik(to_se3(expert.start_pose()), qs[0], iters=200)
    start_err = 0.0 if ok0 else err0
    return {'max_err': float(errs.max()), 'n_over_tol': int((errs > tol).sum()),
            'n_over_accept': int((errs > accept).sum()), 'joint_jump_rad': jump,
            'limit_margin_rad': margin, 'start_err': float(start_err),
            'q_start': qs[0].round(3).tolist(),
            'q_min': qs.min(axis=0).round(3).tolist(), 'q_max': qs.max(axis=0).round(3).tolist(),
            'ok': bool((errs <= accept).all() and start_err <= tol and jump < 0.3)}


class SimLink:
    """receiver와의 UDP 연결. 실측 메시지: q, rx, (새 receiver) seq, t(시뮬 시각), reset_id."""

    def __init__(self, listen: str, send: str, nq: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(addr(listen))
        self.sock.setblocking(False)
        self.dest = addr(send)
        self.nq = nq
        self.t0_wall = time.monotonic()

    def recv_latest(self, timeout: float = 0.5) -> dict | None:
        """쌓인 실측 중 마지막 것 (없으면 timeout까지 기다림). 시뮬 시각이 없으면 받은 시각."""
        r, _, _ = select.select([self.sock], [], [], timeout)
        if not r:
            return None
        last = None
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except (BlockingIOError, ConnectionResetError):
                break
            msg = decode(data, self.nq)
            if msg is not None:
                last = msg
        if last is not None and 't' not in last:
            last['t'] = time.monotonic() - self.t0_wall
        return last

    def send(self, q, **extra) -> None:
        self.sock.sendto(encode(q, **extra), self.dest)

    def close(self):
        self.sock.close()


def place_robot(link: SimLink, q_start: np.ndarray, scene: dict, reset_id: int, tol: float = 2e-3,
                settle: int = 15, timeout: float = 20.0) -> bool:
    """receiver에 관절 자세를 바로 놓게 하고(reset_q), 실측이 settle번 연속 tol 안이면 True."""
    t_end = time.monotonic() + timeout
    good = 0
    t_send = 0.0
    while time.monotonic() < t_end:
        if time.monotonic() - t_send > 0.1:
            link.send(q_start, reset_q=[float(x) for x in q_start], reset_id=reset_id, scene=scene)
            t_send = time.monotonic()
        msg = link.recv_latest(0.1)
        if msg is None:
            continue
        if msg.get('reset_id', reset_id) != reset_id:      # 이전 receiver는 reset_id를 안 보냄
            continue
        good = good + 1 if np.max(np.abs(msg['q'] - q_start)) < tol else 0
        if good >= settle:
            return True
    return False


def run_episode(link: SimLink, stepper: FollowerStepper, scn: dict, scenario_path: str, *, home_q: np.ndarray,
                use_scenario_pose: bool = False, sensor_seed: int = 0, reset_id: int = 1, max_time: float = 400.0,
                log=print) -> tuple[Recorder, dict, str]:
    """에피소드 하나. 반환 (기록, 모재 배치가 반영된 시나리오, 결과: done | reach | place | ik_failed | timeout | stopped)."""
    kin = stepper.kin
    model = PartErrorModel(scn)
    T_home = kin.fk(home_q).homogeneous
    if not use_scenario_pose:
        model.set_nominal(isaac_nominal(model, T_home))
    expert = ScriptedExpert(model, T_home, np.random.default_rng(sensor_seed))
    rec = Recorder()
    bad, q_seam0 = check_reach(kin, expert, home_q, stepper.accept)
    if bad:
        log(f'IK 도달 실패 (s, 오차): {bad}')
        return rec, model.scn, 'reach'
    scene = {'scenario': os.path.abspath(scenario_path), 's_arc': 0.0, 'T_nominal': model.scn['T_nominal'],
             'phase': expert.phase}

    msg = None
    if expert.weld_only:
        q_start, ok, err = kin.ik(to_se3(expert.start_pose()), q_seam0, iters=200)
        if not (ok or err <= stepper.tol):
            log(f'시작 자세 IK 실패 (오차 {err:.4f})')
            return rec, model.scn, 'reach'
        if not place_robot(link, q_start, scene, reset_id):
            log('로봇을 시작 자세에 놓지 못함 (receiver가 reset_q를 지원하는지, 실행 중인지 확인)')
            return rec, model.scn, 'place'
        stepper.q_cmd = q_start.copy()
        stepper.q_ik = q_start.copy()
    else:
        link.send(home_q, scene=scene)
    stepper.target = None

    t_prev = None
    t_print = 0.0
    result = 'timeout'
    t_wall_end = time.monotonic() + max_time
    while time.monotonic() < t_wall_end:
        msg = link.recv_latest(1.0)
        if msg is None:
            log('\rIsaac Sim 실측 대기 중…' + ' ' * 40)
            continue
        if msg.get('reset_id', reset_id) != reset_id:
            continue
        t = float(msg['t'])
        stepper.on_measured(msg['q'], t)
        if t_prev is None:
            t_prev = t
            if not expert.weld_only and stepper.q_cmd is None:
                stepper.set_initial(msg['q'])
            continue
        dt = t - t_prev
        if dt <= 0:
            continue
        t_prev = t
        r = expert.step(min(dt, 0.05))
        stepper.on_target(to_se3(r['T_target']), t)
        q = stepper.step(t)
        scene.update(s_arc=r['s_arc'], phase=r['phase'])
        q_send = q if q is not None else stepper.q_cmd    # 명령을 못 내도 응답(ack)은 보냄 (잠금 단계)
        if q_send is not None:
            link.send(q_send, target=se3_to_xyzquat(to_se3(r['T_target'])).tolist(), scene=scene,
                      ack=msg.get('seq'), reset_id=reset_id)
        T_meas = kin.fk(stepper.q_meas).homogeneous
        rec.add_expert(r, T_meas=T_meas, q_cmd=q if q is not None else np.full(kin.nq, np.nan),
                       q_meas=stepper.q_meas.copy(), status=stepper.status, t_sim=t)
        if stepper.status == 'ik_failed':
            log(f'\nIK 실패로 중단 ({stepper.detail}), 진행도 {r["s"]:.3f}')
            result = 'ik_failed'
            break
        if expert.done:
            result = 'done'
            break
        if time.monotonic() - t_print > 0.2:
            err = np.linalg.norm(T_meas[:3, 3] - r['T_target'][:3, 3]) * 1e3
            log(f'\r{r["phase"]:<8} s {r["s"]:.3f} | 좌우 {r["e"][1] * 1e3:+5.2f} 높이 {r["e"][2] * 1e3:+5.2f} mm'
                f' | 틈새 {r["gap"] * 1e3:4.2f} mm | 추종 오차 {err:5.1f} mm | {stepper.status:<10}', end='', flush=True)
            t_print = time.monotonic()
    return rec, model.scn, result


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
    ap.add_argument('--use-scenario-pose', action='store_true', help='고정 자동 배치 대신 시나리오의 nominal_pose 사용')
    ap.add_argument('--home-q', default=HOME_Q, help='rad, receiver --initial-q와 같게 (모재 배치 기준)')
    ap.add_argument('--send', default='127.0.0.1:9870')
    ap.add_argument('--listen', default='127.0.0.1:9871')
    ap.add_argument('--sensor-seed', type=int, default=0, help='단면 센서 노이즈 난수 시드')
    ap.add_argument('--out', help='기록 파일 (.npz). 기본 outputs/scratch/episodes/<시나리오 이름>.npz')
    args = ap.parse_args()

    scn = load_scenario(args.scenario)
    stepper = build_stepper(args.follower_config, args.follower_urdf)
    out_path = args.out or str(paths.scratch_dir('episodes') / (os.path.splitext(os.path.basename(args.scenario))[0] + '.npz'))
    link = SimLink(args.listen, args.send, stepper.kin.nq)
    print(f'시나리오 {args.scenario} (패스 {scn.get("pass_index", 0)}) → Isaac Sim {args.send}')
    rec, scn_out, result = Recorder(), scn, 'stopped'
    try:
        rec, scn_out, result = run_episode(link, stepper, scn, args.scenario,
                                           home_q=np.array([float(x) for x in args.home_q.split(',')]),
                                           use_scenario_pose=args.use_scenario_pose, sensor_seed=args.sensor_seed,
                                           reset_id=int(time.time()) % 100000)
    except KeyboardInterrupt:
        print('\n중단')
    finally:
        link.close()
        print(f'\n결과: {result}')
        if len(rec):
            rec.save(out_path, scn_out, result=result)
            print(f'기록: {os.path.abspath(out_path)} ({len(rec)} 주기)')


if __name__ == '__main__':
    main()
