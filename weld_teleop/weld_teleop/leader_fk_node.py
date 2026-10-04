"""/leader/joint_states → /leader/ee_pose (리더 베이스 기준 PoseStamped)

6축/7축 전환 시 이 노드의 코드는 바뀌지 않는다. config의 urdf/joint_names/ee_frame만 바꾼다.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .kinematics import ChainKinematics, se3_from_xyz_rpy, se3_to_xyzquat


def pose_msg(v, frame_id, stamp) -> PoseStamped:
    m = PoseStamped()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    (m.pose.position.x, m.pose.position.y, m.pose.position.z,
     m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z,
     m.pose.orientation.w) = map(float, v)
    return m


class LeaderFkNode(Node):
    def __init__(self):
        super().__init__('leader_fk_node')
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('joint_names', [''])
        self.declare_parameter('ee_frame', '')
        self.declare_parameter('tool_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('tool_rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('base_frame', 'leader_base')

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.kin = ChainKinematics(
            p('urdf_path'), list(p('joint_names')), p('ee_frame'),
            se3_from_xyz_rpy(p('tool_xyz'), p('tool_rpy')))
        self.base_frame = p('base_frame')
        self.pub = self.create_publisher(PoseStamped, '/leader/ee_pose', 10)
        self.create_subscription(JointState, '/leader/joint_states', self._cb, 10)
        self.get_logger().info(f'leader chain: {self.kin.nq} joints → {p("ee_frame")}')

    def _cb(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            q = np.array([msg.position[idx[n]] for n in self.kin.joint_names])
        except KeyError as e:
            self.get_logger().error(f'joint_states에 {e} 없음', throttle_duration_sec=2.0)
            return
        v = se3_to_xyzquat(self.kin.fk(q))
        self.pub.publish(pose_msg(v, self.base_frame, msg.header.stamp))


def main():
    rclpy.init()
    node = LeaderFkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
