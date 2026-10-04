"""리더 EE 포즈 → 팔로워 TCP 목표 포즈 (상대 매핑 + 클러치)

클러치 ON 순간의 리더 포즈 L0, 팔로워 포즈 F0를 기준점으로 저장하고
  p_target = p_F0 + s * R_bl (p_L - p_L0)
  R_target = (R_bl dR_L R_bl^T) R_F0,   dR_L = R_L R_L0^T
로 계산한다. R_bl: 리더 베이스 → 팔로워 베이스 회전(축 정렬용).

이후 1주기당 최대 이동량(병진/회전)을 제한하며 목표 쪽으로 따라가고,
팔로워 베이스 기준 작업공간 박스로 위치를 자른다.

orientation_mode
- full          : 자세 6D 전부 반영 (팔 6관절 리더 = 개조형 7축 기본값)
- position_only : 위치만 반영, 자세는 F0 유지 (팔 5관절 리더 또는 초기 검증용)
"""
import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float64

from .kinematics import se3_to_xyzquat, xyzquat_to_se3
from .leader_fk_node import pose_msg


def msg_to_se3(m: PoseStamped) -> pin.SE3:
    p, o = m.pose.position, m.pose.orientation
    return xyzquat_to_se3([p.x, p.y, p.z, o.x, o.y, o.z, o.w])


class EeMappingNode(Node):
    def __init__(self):
        super().__init__('ee_mapping_node')
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('position_scale', 1.0)
        self.declare_parameter('leader_to_follower_rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('orientation_mode', 'full')
        self.declare_parameter('max_linear_step', 0.002)    # m / cycle
        self.declare_parameter('max_angular_step', 0.02)    # rad / cycle
        self.declare_parameter('workspace_min', [-1.0, -1.0, 0.0])
        self.declare_parameter('workspace_max', [1.0, 1.0, 1.2])
        self.declare_parameter('auto_clutch', False)         # 테스트용: 시작 즉시 클러치 ON
        self.declare_parameter('follower_frame', 'link0')

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.s = float(p('position_scale'))
        self.R_bl = pin.rpy.rpyToMatrix(*p('leader_to_follower_rpy'))
        self.mode = p('orientation_mode')
        self.max_lin = float(p('max_linear_step'))
        self.max_ang = float(p('max_angular_step'))
        self.ws_min = np.array(p('workspace_min'), dtype=float)
        self.ws_max = np.array(p('workspace_max'), dtype=float)
        self.frame = p('follower_frame')

        self.leader: pin.SE3 | None = None
        self.follower: pin.SE3 | None = None
        self.clutch = bool(p('auto_clutch'))
        self.anchor = None          # (L0, F0)
        self.cmd: pin.SE3 | None = None

        self.create_subscription(PoseStamped, '/leader/ee_pose', self._on_leader, 10)
        self.create_subscription(PoseStamped, '/follower/ee_pose', self._on_follower, 10)
        self.create_subscription(Bool, '/teleop/clutch', self._on_clutch, 10)
        self.create_subscription(Float64, '/leader/aux', self._on_aux, 10)
        self.pub = self.create_publisher(PoseStamped, '/follower/ee_target', 10)
        self.pub_aux = self.create_publisher(Float64, '/follower/aux_command', 10)
        self.create_timer(1.0 / float(p('rate_hz')), self._tick)

    # --- callbacks ---
    def _on_leader(self, m):
        self.leader = msg_to_se3(m)

    def _on_follower(self, m):
        self.follower = msg_to_se3(m)

    def _on_clutch(self, m: Bool):
        if m.data != self.clutch:
            self.get_logger().info(f'clutch {"ON" if m.data else "OFF"}')
        self.clutch = m.data
        if not m.data:
            self.anchor = None

    def _on_aux(self, m: Float64):
        # TODO: 그리퍼 축 용도 확정 후 해석 (예: 임계값 넘으면 아크 ON 트리거)
        self.pub_aux.publish(m)

    # --- main loop ---
    def _tick(self):
        if not self.clutch or self.leader is None or self.follower is None:
            return
        if self.anchor is None:  # 클러치 ON 직후 재기준
            self.anchor = (pin.SE3(self.leader), pin.SE3(self.follower))
            self.cmd = pin.SE3(self.follower)
            return

        L0, F0 = self.anchor
        L = self.leader
        p = F0.translation + self.s * self.R_bl @ (L.translation - L0.translation)
        p = np.clip(p, self.ws_min, self.ws_max)
        if self.mode == 'full':
            dR = self.R_bl @ (L.rotation @ L0.rotation.T) @ self.R_bl.T
            R = dR @ F0.rotation
        else:
            R = F0.rotation
        raw = pin.SE3(R, p)

        # 1주기 이동량 제한: cmd → raw 방향 twist를 잘라서 적용
        xi = pin.log6(self.cmd.actInv(raw)).vector
        lin, ang = xi[:3], xi[3:]
        nl, na = np.linalg.norm(lin), np.linalg.norm(ang)
        if nl > self.max_lin:
            lin = lin * self.max_lin / nl
        if na > self.max_ang:
            ang = ang * self.max_ang / na
        self.cmd = self.cmd * pin.exp6(np.concatenate([lin, ang]))

        self.pub.publish(pose_msg(se3_to_xyzquat(self.cmd), self.frame,
                                  self.get_clock().now().to_msg()))


def main():
    rclpy.init()
    node = EeMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
