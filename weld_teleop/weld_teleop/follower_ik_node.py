"""팔로워(opti3): TCP 목표 포즈 → IK → 관절 명령

- /follower/joint_states(파라미터로 토픽 지정) 수신 → FK → /follower/ee_pose 발행
- /follower/ee_target 수신 → DLS IK → 관절 명령
- dry_run=true(기본): 실제 명령 대신 /follower/ik_preview(JointState)만 발행 → RViz 확인용

TODO(확인 필요): indy-ros2 실기 bringup에서 활성화되는 컨트롤러 이름과 명령 토픽.
  `ros2 control list_controllers`, `ros2 topic list`로 확인 후 command_topic/output_type 설정.
  대안: indy_moveit의 servo_mode(MoveIt Servo)에 포즈 명령을 보내는 방식도 가능.
"""
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .ee_mapping_node import msg_to_se3
from .kinematics import ChainKinematics, se3_from_xyz_rpy, se3_to_xyzquat
from .leader_fk_node import pose_msg


class FollowerIkNode(Node):
    def __init__(self):
        super().__init__('follower_ik_node')
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('joint_names', [''])
        self.declare_parameter('ee_frame', 'tcp')
        self.declare_parameter('tool_xyz', [0.0, 0.0, 0.0])   # 토치 팁 오프셋
        self.declare_parameter('tool_rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('base_frame', 'link0')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('max_joint_step', 0.01)         # rad / cycle
        self.declare_parameter('ik_tolerance', 1e-3)            # log6 error norm
        self.declare_parameter('dry_run', True)
        self.declare_parameter('output_type', 'joint_trajectory')  # | float64_array
        self.declare_parameter('command_topic',
                               '/joint_trajectory_controller/joint_trajectory')
        # loopback=true: 로봇 없이 테스트. 명령값을 곧 측정값으로 간주(initial_q에서 시작)
        self.declare_parameter('loopback', False)
        self.declare_parameter('initial_q', [0.0, 0.0, -1.57, 0.0, -1.57, 0.0])

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.kin = ChainKinematics(
            p('urdf_path'), list(p('joint_names')), p('ee_frame'),
            se3_from_xyz_rpy(p('tool_xyz'), p('tool_rpy')))
        self.base_frame = p('base_frame')
        self.dt = 1.0 / float(p('rate_hz'))
        self.max_step = float(p('max_joint_step'))
        self.tol = float(p('ik_tolerance'))
        self.dry_run = bool(p('dry_run'))
        self.output_type = p('output_type')

        self.q_meas: np.ndarray | None = None
        self.q_cmd: np.ndarray | None = None
        self.target = None
        self.loopback = bool(p('loopback'))
        if self.loopback:
            self.q_cmd = np.array(p('initial_q'), dtype=float)
            self.dry_run = True  # loopback에서는 실기 명령 금지
        else:
            self.create_subscription(JointState, p('joint_state_topic'), self._on_js, 10)
        self.create_subscription(PoseStamped, '/follower/ee_target', self._on_target, 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/follower/ee_pose', 10)
        self.pub_preview = self.create_publisher(JointState, '/follower/ik_preview', 10)
        if not self.dry_run:
            mtype = JointTrajectory if self.output_type == 'joint_trajectory' \
                else Float64MultiArray
            self.pub_cmd = self.create_publisher(mtype, p('command_topic'), 10)
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(f'dry_run={self.dry_run}, joints={self.kin.joint_names}')

    def _on_js(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            q = np.array([msg.position[idx[n]] for n in self.kin.joint_names])
        except KeyError:
            return
        self.q_meas = q
        if self.q_cmd is None:
            self.q_cmd = q.copy()
        self.pub_pose.publish(pose_msg(se3_to_xyzquat(self.kin.fk(q)),
                                       self.base_frame, msg.header.stamp))

    def _on_target(self, msg: PoseStamped):
        self.target = msg_to_se3(msg)

    def _tick(self):
        if self.loopback:
            self.pub_pose.publish(pose_msg(se3_to_xyzquat(self.kin.fk(self.q_cmd)),
                                           self.base_frame, self.get_clock().now().to_msg()))
        if self.target is None or self.q_cmd is None:
            return
        q_sol, ok, err = self.kin.ik(self.target, self.q_cmd)
        if not ok and err > self.tol:
            self.get_logger().warn(f'IK 미수렴 (err={err:.4f}) — 명령 보류',
                                   throttle_duration_sec=1.0)
            return
        dq = np.clip(q_sol - self.q_cmd, -self.max_step, self.max_step)
        self.q_cmd = self.q_cmd + dq
        self._publish(self.q_cmd)

    def _publish(self, q: np.ndarray):
        now = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = now
        js.name = self.kin.joint_names
        js.position = q.tolist()
        self.pub_preview.publish(js)
        if self.dry_run:
            return
        if self.output_type == 'joint_trajectory':
            tr = JointTrajectory()
            tr.joint_names = self.kin.joint_names
            pt = JointTrajectoryPoint()
            pt.positions = q.tolist()
            ns = int(self.dt * 2 * 1e9)
            pt.time_from_start = Duration(sec=ns // 10**9, nanosec=ns % 10**9)
            tr.points = [pt]
            self.pub_cmd.publish(tr)
        else:
            self.pub_cmd.publish(Float64MultiArray(data=q.tolist()))


def main():
    rclpy.init()
    node = FollowerIkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
