"""리더 서보 → /leader/joint_states (+ /leader/aux)

aux(현 그리퍼 축)는 용도 미정이므로 0~1로 정규화한 값만 그대로 내보낸다.
용도가 정해지면 ee_mapping_node에서 해석한다.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from .leader_drivers import FeetechDriver, MockDriver


class LeaderStateNode(Node):
    def __init__(self):
        super().__init__('leader_state_node')
        self.declare_parameter('driver', 'mock')            # mock | feetech
        self.declare_parameter('joint_names', [''])
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 1000000)
        self.declare_parameter('servo_ids', [0])
        self.declare_parameter('offsets', [0])
        self.declare_parameter('signs', [1])
        self.declare_parameter('aux_servo_id', -1)           # -1 = 사용 안 함
        self.declare_parameter('aux_range', [0, 4095])
        # driver=lerobot 전용
        self.declare_parameter('lerobot_id', 'my_leader')        # lerobot-calibrate 때 쓴 --teleop.id
        self.declare_parameter('lerobot_keys', [''])              # lerobot 모터 이름 (URDF joint 순서대로)
        self.declare_parameter('signs_f', [1.0])
        self.declare_parameter('offsets_rad', [0.0])

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.joint_names = list(p('joint_names'))
        aux_id = int(p('aux_servo_id'))

        if p('driver') == 'lerobot':
            from .leader_drivers import LerobotDriver
            self.driver = LerobotDriver(p('port'), p('lerobot_id'), list(p('lerobot_keys')),
                                        list(p('signs_f')), list(p('offsets_rad')))
        elif p('driver') == 'feetech':
            self.driver = FeetechDriver(
                p('port'), int(p('baudrate')), list(p('servo_ids')),
                list(p('offsets')), list(p('signs')),
                aux_id=aux_id if aux_id >= 0 else None,
                aux_range=tuple(p('aux_range')))
        else:
            self.driver = MockDriver(len(self.joint_names), has_aux=aux_id >= 0)

        self.pub_js = self.create_publisher(JointState, '/leader/joint_states', 10)
        self.pub_aux = self.create_publisher(Float64, '/leader/aux', 10)
        self.create_timer(1.0 / float(p('rate_hz')), self._tick)
        self.get_logger().info(
            f"driver={p('driver')}, joints({len(self.joint_names)})={self.joint_names}")

    def _tick(self):
        try:
            q, aux = self.driver.read()
        except (IOError, KeyError) as e:
            self.get_logger().warn(str(e), throttle_duration_sec=1.0)
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = q.tolist()
        self.pub_js.publish(msg)
        if aux is not None:
            self.pub_aux.publish(Float64(data=aux))

    def destroy_node(self):
        self.driver.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = LeaderStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
