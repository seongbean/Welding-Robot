#!/usr/bin/env python3
"""공유 제어 → Isaac Sim opti3: 고정한 기본 정책이 용접하고, 사람이 SO-102 리더로 보정을 더하며 개입 에피소드를 기록한다.

policy_to_sim.py(정책 실행기)와 같은 시나리오·배치·lockstep을 쓰고, 토치 목표만 learning/shared_control.py의
SharedDriver로 낸다: 로봇 보정 = 기본 정책 + (선택) 잔차 모델 + 사람 보정 (이음매 좌표계).
기본 정책은 개입 데이터로 다시 학습하지 않는다. 기록의 human_c(유지 오프셋 포함)가 잔차 모델의 정답이다.

사용법 (저장소 루트에서, conda activate weld. torch·pinocchio·scservo_sdk 필요)
  터미널 1: C:\\isaacsim\\python.bat sim_bridge\\isaac_opti3_receiver.py --lockstep
  터미널 2:
    set K=outputs\\experiments\\20261123-kin-4fam
    python sim_bridge\\shared_to_sim.py --policy %K%\\runs\\mlp_all\\policy.pt --base-tag base-v1 ^
        --scenarios %K%\\episodes --per-family 1 --port COM3 --exp 20261125-shared ^
        --follower-urdf C:\\workspace\\indy-ros2-jazzy\\indy_description\\urdf_files\\opti3.urdf
  리더 없이 시험: --port 대신 --mock (사인파 리더, 클러치는 키보드 c).

조작 (이 터미널에 포커스. Windows는 키 하나씩, 다른 OS는 키 + Enter)
  s = 시작 트리거 (로봇을 시작 자세에 놓은 뒤 누르면 용접 시작)   n = 이 시나리오 건너뜀
  클러치: --clutch grip(기본) = 리더 그리퍼를 쥐고 있는 동안 ON (값이 --grip-on 아래), --clutch key = c로 켜고 끔
  r = 오프셋 초기화 (유지 오프셋을 --reset-time 초 동안 0으로)   q = 지금 에피소드 멈춤 (대기 중이면 전체 종료)

- 리더 매핑: 클러치 ON 순간 대비 리더 끝 변화를 이음매 좌표로 바로 읽는다 (리더 베이스 x = 진행, y = 좌우, z = 높이,
  --leader-to-seam-rpy로 돌림). 좌우·높이 = 위치 × --pos-scale, 작업각·진행각 = x·y축 회전 × --ang-scale.
  진행 방향 이동은 버린다. --gain으로 축별 권한(0~1), --limit으로 한도, 속도·위빙은 모델이 맡는다.
- 출력 (--out, 기본: --exp면 <실험>/runs/<--run>/shared, 아니면 outputs/scratch/shared/<--run>):
  <시나리오>__<시각>.npz (끝나지 않거나 멈추면 .failed.npz). 필드는 learning/shared_control.py RECORD_FIELDS.
  meta에 정책 경로·sha256·고정 이름(learning/base_policies.json), 잔차 모델, 사람 보정 설정을 남긴다.
  끝나면 policy_to_sim과 같은 오차 요약(eval_isaac.json)과 개입 요약(interventions.json)을 낸다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, 'weld_teleop')
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, PKG, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import paths  # noqa: E402
from learning.base_policy import identify  # noqa: E402
from learning.features import load_episode  # noqa: E402
from learning.shared_control import HumanCorrection, SharedDriver, intervention_stats  # noqa: E402
from policy_to_sim import HOME_Q, build_report, load_policy, print_report, select, sensor_seed  # noqa: E402
from sim_errors.part_errors import save_scenario  # noqa: E402


def floats(s: str, n: int | None = None) -> list[float]:
    v = [float(x) for x in s.split(',')]
    if n is not None and len(v) == 1:
        v = v * n
    if n is not None and len(v) != n:
        raise argparse.ArgumentTypeError(f'값 {n}개 필요: {s}')
    return v


class Keys:
    """키 입력 스레드. Windows는 msvcrt로 키 하나씩, 다른 OS는 줄 단위(각 글자를 키로)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pending: list[str] = []
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            import msvcrt
        except ImportError:
            msvcrt = None
        if msvcrt is not None:
            while True:
                ch = msvcrt.getwch()
                self.push(ch.lower())
        else:
            try:
                for line in sys.stdin:
                    for ch in line.strip().lower():
                        self.push(ch)
            except (OSError, ValueError):      # 입력이 없는 실행 (시험 등)
                pass

    def push(self, ch: str) -> None:
        with self.lock:
            self.pending.append(ch)

    def take(self, ch: str) -> bool:
        """ch가 눌렸으면 하나 꺼내고 True."""
        with self.lock:
            if ch in self.pending:
                self.pending.remove(ch)
                return True
        return False

    def clear(self) -> None:
        with self.lock:
            self.pending.clear()


class Leader:
    """리더 읽기 스레드: 관절 → FK 포즈(4x4)와 그리퍼 값. 최근 stale초 안에 읽은 값만 돌려준다."""

    def __init__(self, driver, kin, rate: float = 100.0, stale: float = 0.2):
        self.driver, self.kin = driver, kin
        self.rate, self.stale = rate, stale
        self.lock = threading.Lock()
        self.T, self.aux, self.q, self.t = None, None, None, -1e9
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop:
            try:
                q, aux = self.driver.read()
                T = self.kin.fk(np.asarray(q, dtype=float)).homogeneous
                with self.lock:
                    self.T, self.aux, self.q, self.t = T, aux, np.asarray(q, dtype=float), time.monotonic()
            except (IOError, KeyError, RuntimeError):
                pass
            time.sleep(1.0 / self.rate)

    def latest(self):
        with self.lock:
            if time.monotonic() - self.t > self.stale:
                return None, None
            return self.T.copy(), self.aux

    def close(self):
        self.stop = True
        self.thread.join(1.0)
        self.driver.close()


class Clutch:
    """클러치 판단. grip: 그리퍼 값이 on 아래로 내려가면 ON, off 위로 올라가면 OFF (히스테리시스). key: c 토글."""

    def __init__(self, source: str, keys: Keys, on: float = 0.2, off: float = 0.3):
        self.source, self.keys, self.on, self.off = source, keys, on, off
        self.state = False

    def update(self, aux: float | None) -> bool:
        if self.source == 'key':
            if self.keys.take('c'):
                self.state = not self.state
        elif aux is None:
            self.state = False                     # 리더가 끊기면 놓은 것으로 본다
        elif self.state:
            self.state = aux < self.off
        else:
            self.state = aux < self.on
        return self.state


def build_leader(args):
    from weld_teleop.kinematics import ChainKinematics, se3_from_xyz_rpy
    from weld_teleop.leader_drivers import FeetechDriver, MockDriver
    from weld_teleop.teleop_loop import ros_params
    st = ros_params(args.leader_config, 'leader_state_node')
    fk = ros_params(args.leader_config, 'leader_fk_node')
    kin = ChainKinematics(args.leader_urdf, list(fk['joint_names']), fk['ee_frame'],
                          se3_from_xyz_rpy(fk['tool_xyz'], fk['tool_rpy']))
    aux_id = int(st['aux_servo_id'])
    if args.mock:
        driver = MockDriver(len(st['joint_names']), amplitude=args.mock_amp, has_aux=aux_id >= 0)
    else:
        driver = FeetechDriver(args.port, int(st['baudrate']), list(st['servo_ids']), list(st['offsets']),
                               list(st['signs']), aux_id=aux_id if aux_id >= 0 else None,
                               aux_range=tuple(st['aux_range']))
    return Leader(driver, kin)


def human_from_args(args) -> HumanCorrection:
    return HumanCorrection(pos_scale=args.pos_scale, ang_scale=args.ang_scale, gain=args.gain,
                           limit=args.limit_mm[:2] + args.limit_deg[:2], reset_time=args.reset_time,
                           leader_to_seam_rpy=args.leader_to_seam_rpy)


def summarize_interventions(out: str) -> dict:
    rows = []
    for name in sorted(os.listdir(out)):
        if not name.endswith('.npz'):
            continue
        rec, _, meta = load_episode(os.path.join(out, name))
        if 'human_c' not in rec:
            continue
        rows.append({'file': name, 'result': meta.get('result'), 'policy_tag': meta.get('policy_tag'),
                     **intervention_stats(rec)})
    res = {'n': len(rows), 'episodes': rows}
    with open(os.path.join(out, 'interventions.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', required=True, help="기본 정책 policy.pt 경로 (또는 규칙 기준선 'rule')")
    ap.add_argument('--base-tag', help='고정 기본 정책 이름 (예: base-v1). 주면 sha256이 learning/base_policies.json과 '
                                       '같아야 시작한다')
    ap.add_argument('--scenarios', help='시나리오 *.json 또는 에피소드 *.npz 폴더')
    ap.add_argument('--split', default='test')
    ap.add_argument('--only')
    ap.add_argument('--per-family', type=int)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--run', help='결과 폴더 이름 (기본: --base-tag 또는 정책 폴더 이름)')
    ap.add_argument('--out', help='출력 폴더 (기본: --exp면 <실험>/runs/<run>/shared, 아니면 outputs/scratch/shared/<run>)')
    ap.add_argument('--port', help='리더 시리얼 포트 (예: COM3)')
    ap.add_argument('--mock', action='store_true', help='리더 없이 사인파 입력')
    ap.add_argument('--mock-amp', type=float, default=0.05)
    ap.add_argument('--leader-urdf', default=os.path.join(PKG, 'urdf', 'so102.urdf'))
    ap.add_argument('--leader-config', default=os.path.join(PKG, 'config', 'leader_so101_7axis.yaml'))
    ap.add_argument('--clutch', choices=('grip', 'key'), default='grip',
                    help='grip = 리더 그리퍼를 쥐는 동안 ON, key = c 키로 켜고 끔 (--mock이면 key)')
    ap.add_argument('--grip-on', type=float, default=0.2, help='그리퍼 값(0 닫힘~1 열림)이 이 아래면 클러치 ON')
    ap.add_argument('--grip-off', type=float, default=0.3, help='클러치 ON에서 이 위로 올라가면 OFF')
    ap.add_argument('--pos-scale', type=float, default=0.25, help='리더 이동 → 좌우·높이 보정 축척')
    ap.add_argument('--ang-scale', type=float, default=0.5, help='리더 회전 → 작업각·진행각 보정 축척')
    ap.add_argument('--gain', type=lambda s: floats(s, 4), default=[1.0] * 4,
                    help='축별 사람 권한 좌우,높이,작업각,진행각 (0~1, 값 하나면 모두)')
    ap.add_argument('--limit-mm', type=lambda s: floats(s, 2), default=[5.0, 5.0], help='좌우,높이 한도 [mm]')
    ap.add_argument('--limit-deg', type=lambda s: floats(s, 2), default=[10.0, 10.0], help='작업각,진행각 한도 [deg]')
    ap.add_argument('--reset-time', type=float, default=1.0, help='오프셋 초기화에 걸리는 시간 [s]')
    ap.add_argument('--leader-to-seam-rpy', type=lambda s: floats(s, 3), default=[0.0, 0.0, 0.0],
                    help='리더 베이스 → 이음매 좌표 회전 [rad] (x 진행, y 좌우, z 높이)')
    ap.add_argument('--follower-urdf', required=True, help='opti3.urdf 경로')
    ap.add_argument('--follower-config', default=os.path.join(PKG, 'config', 'follower_opti3.yaml'))
    ap.add_argument('--home-q', default=HOME_Q)
    ap.add_argument('--send', default='127.0.0.1:9870')
    ap.add_argument('--listen', default='127.0.0.1:9871')
    ap.add_argument('--max-time', type=float, default=900.0, help='에피소드 하나의 최대 실제 시간 [s]')
    paths.add_exp_arg(ap)
    args = ap.parse_args(argv)
    if not args.mock and not args.port:
        ap.error('--port 또는 --mock 중 하나가 필요합니다')
    if not args.scenarios:
        ap.error('--scenarios가 필요합니다')
    if args.mock:
        args.clutch = 'key'
    try:
        tag, digest = identify(args.policy, args.base_tag)
    except ValueError as e:
        sys.exit(f'기본 정책 확인 실패: {e}')
    if not tag and os.path.isfile(args.policy):
        print(f'주의: {args.policy}는 고정 기본 정책 목록에 없음 (sha256 {digest[:12]}…). 기록에는 해시를 남긴다.')
    run = args.run or tag or os.path.basename(os.path.dirname(os.path.abspath(args.policy))) or 'policy'
    out = args.out or str(paths.exp_dir(args.exp, 'runs') / run / 'shared' if args.exp
                          else paths.scratch_dir('shared', run))
    os.makedirs(out, exist_ok=True)
    todo = select(args.scenarios, args.split, args.only, args.per_family, args.limit)
    print(f'시나리오 {len(todo)}개 (나눔 {args.split}) → {os.path.abspath(out)}')
    human_cfg = human_from_args(args).config()
    paths.record_stage(args.exp, 'shared_to_sim', policy=args.policy, policy_sha256=digest, policy_tag=tag,
                       scenarios=args.scenarios, split=args.split, only=args.only, per_family=args.per_family,
                       limit=args.limit, out=out, leader='mock' if args.mock else args.port, clutch=args.clutch,
                       human=human_cfg, follower_urdf=args.follower_urdf,
                       follower_config_sha256=paths.file_sha256(args.follower_config), n=len(todo))
    if not todo:
        return 0

    from expert_to_sim import SimLink, build_stepper, run_episode
    policy = load_policy(args.policy)
    stepper = build_stepper(args.follower_config, args.follower_urdf)
    link = SimLink(args.listen, args.send, stepper.kin.nq)
    keys = Keys()
    leader = build_leader(args)
    clutch = Clutch(args.clutch, keys, args.grip_on, args.grip_off)
    home = np.array([float(x) for x in args.home_q.split(',')])
    base_id = int(time.time()) % 100000 * 1000
    scn_dir = os.path.join(out, 'scenarios')
    os.makedirs(scn_dir, exist_ok=True)
    quit_all = False
    try:
        for k, (name, scn) in enumerate(todo, 1):
            scn_path = os.path.join(scn_dir, name + '.json')
            save_scenario(scn, scn_path)
            seed = sensor_seed(name)
            human = human_from_args(args)
            state = {'skip': False}
            print(f'\n[{k}/{len(todo)}] {name}: 시작 자세로 옮기는 중…')

            def wait_start(state=state):
                if keys.take('q'):
                    state['quit'] = True
                    return 'stop'
                if keys.take('n'):
                    state['skip'] = True
                    return 'stop'
                if not state.get('prompted'):
                    keys.clear()
                    print('  준비됨: s = 시작, n = 건너뜀, q = 종료', flush=True)
                    state['prompted'] = True
                return keys.take('s')

            def human_input():
                T, aux = leader.latest()
                return T, clutch.update(aux), keys.take('r'), aux

            def log(msg, **kw):
                if msg.startswith('\r') and kw.get('end') == '':
                    h = human.applied
                    msg = (msg[:msg.rfind('|')] + f'| 클러치 {"ON " if clutch.state else "OFF"} 사람 좌우 {h[0]:+4.1f} '
                           f'높이 {h[1]:+4.1f} mm 각 {h[2]:+4.1f}/{h[3]:+4.1f}°' + (' 초기화' if human.resetting else ''))
                print(msg, **kw)

            t1 = time.time()
            rec, scn_out, result = run_episode(
                link, stepper, scn, scn_path, home_q=home, sensor_seed=seed, reset_id=base_id + k,
                max_time=args.max_time, log=log, wait_start=wait_start,
                make_driver=lambda m, e, seed=seed, human=human: SharedDriver(
                    policy, m, e, human, human_input, sensor_seed=seed, stop_requested=lambda: keys.take('q')))
            if state.get('quit'):
                quit_all = True
            print(f'\n  결과 {result}, {len(rec)} 주기, {time.time() - t1:.0f} s')
            if len(rec) and result not in ('reach', 'place'):
                stamp = time.strftime('%Y%m%d-%H%M%S')
                path = os.path.join(out, f'{name}__{stamp}' + ('.npz' if result == 'done' else '.failed.npz'))
                rec.save(path, scn_out, result=result, runner='isaac_shared', policy=args.policy,
                         policy_sha256=digest, policy_tag=tag, residual=None, sensor_seed=seed,
                         leader='mock' if args.mock else args.port, clutch_source=args.clutch, human=human_cfg)
                st = intervention_stats(rec.rows)
                print(f'  기록 {path}: 클러치 {st["n_clutch"]}회 (시간 {st["clutch_frac"] * 100:.0f}%), '
                      f'초기화 {st["n_reset"]}회, 사람 보정 구간 {st["human_frac"] * 100:.0f}%')
            if quit_all:
                break
    except KeyboardInterrupt:
        print('\n중단')
    finally:
        link.close()
        leader.close()

    print_report(build_report(out, args.policy))
    res = summarize_interventions(out)
    print(f'개입 요약: {os.path.join(os.path.abspath(out), "interventions.json")} ({res["n"]}개)')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
