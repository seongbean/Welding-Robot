"""ROS 없이 한 프로세스에서 도는 텔레옵 1주기 (리더 관절 → 팔로워 관절 명령)

ROS 노드 4개(leader_state → leader_fk → ee_mapping → follower_ik)가 토픽으로 나눠 하던 일을
그대로 한 함수에서 한다. Windows에서 Isaac Sim과 UDP로 연결할 때 사용 (sim_bridge/ 참고).

- 리더 FK → EeMapper(클러치·상대 매핑) → FollowerStepper(IK·안전 정지)
- 팔로워 실측 관절은 on_measured()로 넣는다. 실측이 끊기면 FollowerStepper가 명령을 멈춘다.
- 클러치 OFF 또는 리더 읽기 실패 시 매퍼를 재기준 상태로 두고 명령을 내지 않는다.
- 도달 범위 경계·특이점(IK 근사 해 또는 실패): 팔로워는 갈 수 있는 가장 가까운 자세까지만 가고
  매퍼를 그 자리에서 재기준한다. 범위 밖으로 더 나가는 리더 움직임은 버려지고, 안쪽으로
  돌아오는 움직임은 클러치를 다시 누르지 않아도 바로 따라간다 (status: out_of_reach).
"""
from __future__ import annotations

import json

import numpy as np
import yaml

from .follower_control import FollowerStepper
from .kinematics import ChainKinematics, se3_from_xyz_rpy
from .mapping import EeMapper


class TeleopLoop:
    def __init__(self, leader_kin: ChainKinematics, follower_kin: ChainKinematics,
                 mapper: EeMapper, stepper: FollowerStepper):
        self.leader_kin = leader_kin
        self.follower_kin = follower_kin
        self.mapper = mapper
        self.stepper = stepper
        self.out_of_reach = False

    def on_measured(self, q, t: float) -> None:
        self.stepper.on_measured(q, t)

    def tick(self, t: float, q_leader, clutch: bool) -> np.ndarray | None:
        """보낼 팔로워 관절 명령, 또는 None (사유는 self.status)."""
        if not clutch or q_leader is None or self.stepper.q_meas is None:
            self.mapper.reset()
            self.out_of_reach = False
            self.stepper.status = ('clutch_off' if not clutch else
                                   'leader_lost' if q_leader is None else 'waiting')
            return None
        fresh = False
        if t - self.stepper.t_meas > self.stepper.timeout:
            self.mapper.reset()   # 실측 복귀 시 그 자세에서 재기준
        else:
            T = self.mapper.step(self.leader_kin.fk(q_leader),
                                 self.follower_kin.fk(self.stepper.q_meas))
            if T is not None:
                self.stepper.on_target(T, t)
                fresh = True
        q = self.stepper.step(t)
        if self.stepper.status in ('ik_failed', 'approx'):
            self.out_of_reach = True
            if fresh:                    # 다음 주기에 현재 리더·팔로워 자세로 재기준
                self.mapper.reset()      # (재기준 주기에는 근사 목표 쪽으로 계속 움직임)
            if self.stepper.status == 'ik_failed':
                self.stepper.target = None   # 도달 못 한 목표는 버림
        elif self.stepper.status == 'ok':
            self.out_of_reach = False
        return q

    @property
    def target(self):
        """현재 팔로워 툴 목표 포즈 (pin.SE3, 베이스 기준) 또는 None."""
        return self.stepper.target

    @property
    def status(self) -> str:
        s = self.stepper
        if self.out_of_reach and s.status in ('ik_failed', 'approx', 'waiting'):
            return 'out_of_reach'
        return f'{s.status} {s.detail}' if s.status in ('tracking_error', 'ik_failed', 'out_of_limits') else s.status


# ----------------------------------------------------------------------- config
def ros_params(path: str, node: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)[node]['ros__parameters']


def build_loop(leader_yaml: str, mapping_yaml: str, follower_yaml: str,
               leader_urdf: str, follower_urdf: str) -> TeleopLoop:
    """ROS 노드와 같은 config 파일로 TeleopLoop 구성."""
    fk = ros_params(leader_yaml, 'leader_fk_node')
    leader = ChainKinematics(leader_urdf, list(fk['joint_names']), fk['ee_frame'],
                             se3_from_xyz_rpy(fk['tool_xyz'], fk['tool_rpy']))
    m = ros_params(mapping_yaml, 'ee_mapping_node')
    mapper = EeMapper(position_scale=m['position_scale'],
                      leader_to_follower_rpy=m['leader_to_follower_rpy'],
                      orientation_mode=m['orientation_mode'],
                      max_linear_step=m['max_linear_step'],
                      max_angular_step=m['max_angular_step'],
                      workspace_min=m['workspace_min'], workspace_max=m['workspace_max'])
    fo = ros_params(follower_yaml, 'follower_ik_node')
    follower = ChainKinematics(follower_urdf, list(fo['joint_names']), fo['ee_frame'],
                               se3_from_xyz_rpy(fo['tool_xyz'], fo['tool_rpy']))
    stepper = FollowerStepper(follower, max_joint_step=fo['max_joint_step'],
                              ik_tolerance=fo['ik_tolerance'],
                              max_track_error=fo['max_track_error'],
                              ik_accept_error=fo.get('ik_accept_error'),
                              input_timeout=fo['input_timeout'])
    return TeleopLoop(leader, follower, mapper, stepper)


# ----------------------------------------------------------------------- UDP 메시지
def encode(q, **extra) -> bytes:
    return json.dumps({'q': [float(x) for x in q], **extra}).encode()


def decode(data: bytes, n: int) -> dict | None:
    """길이가 n인 관절 벡터가 든 메시지만 통과 (깨진 패킷은 None)."""
    try:
        msg = json.loads(data.decode())
        q = np.asarray(msg['q'], dtype=float)
    except (ValueError, KeyError, TypeError):
        return None
    if q.shape != (n,) or not np.all(np.isfinite(q)):
        return None
    msg['q'] = q
    return msg
