"""리더 EE 포즈 → 팔로워 TCP 목표 포즈 (상대 매핑 + 클러치)

계산식은 mapping.EeMapper 참고. 이 노드는 토픽 입출력과 클러치만 담당한다.

orientation_mode
- full          : 자세 6D 전부 반영 (팔 6관절 리더 = 개조형 7축 기본값)
- position_only : 위치만 반영, 자세는 F0 유지 (팔 5관절 리더 또는 초기 검증용)
"""
import pinocchio as pin
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float64

from .kinematics import se3_to_xyzquat, xyzquat_to_se3
from .leader_fk_node import pose_msg
from .mapping import EeMapper


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
        self.mapper = EeMapper(
            position_scale=p('position_scale'),
            leader_to_follower_rpy=p('leader_to_follower_rpy'),
            orientation_mode=p('orientation_mode'),
            max_linear_step=p('max_linear_step'),
            max_angular_step=p('max_angular_step'),
            workspace_min=p('workspace_min'),
            workspace_max=p('workspace_max'))
        self.frame = p('follower_frame')

        self.leader: pin.SE3 | None = None
        self.follower: pin.SE3 | None = None
        self.clutch = bool(p('auto_clutch'))

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
            self.mapper.reset()

    def _on_aux(self, m: Float64):
        # TODO: 그리퍼 축 용도 확정 후 해석 (예: 임계값 넘으면 아크 ON 트리거)
        self.pub_aux.publish(m)

    # --- main loop ---
    def _tick(self):
        if not self.clutch or self.leader is None or self.follower is None:
            return
        cmd = self.mapper.step(self.leader, self.follower)
        if cmd is None:
            return
        self.pub.publish(pose_msg(se3_to_xyzquat(cmd), self.frame,
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
