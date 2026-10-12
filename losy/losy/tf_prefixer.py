#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


class TfPrefixer(Node):
    def __init__(self):
        super().__init__('tf_prefixer')
        self.declare_parameter('prefix', 'robot_name')
        self.declare_parameter('input_tf_topic', '/robot_name/tf_raw')
        self.declare_parameter('input_tf_static_topic', '/robot_name/tf_static_raw')

        prefix = self.get_parameter('prefix').value
        self.prefix = prefix.rstrip('/') + '/'

        in_tf = self.get_parameter('input_tf_topic').value
        in_tf_static = self.get_parameter('input_tf_static_topic').value

        self.pub_tf = self.create_publisher(TFMessage, '/tf', 10)
        static_qos = QoSProfile(depth=10)
        static_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        static_qos.reliability = ReliabilityPolicy.RELIABLE
        self.pub_tf_static = self.create_publisher(TFMessage, '/tf_static', static_qos)

        self.create_subscription(TFMessage, in_tf, self.cb_tf, 10)
        self.create_subscription(TFMessage, in_tf_static, self.cb_tf_static, static_qos)

    def _prefix_msg(self, msg: TFMessage) -> TFMessage:
        for t in msg.transforms:
            if not t.header.frame_id.startswith(self.prefix):
                t.header.frame_id = self.prefix + t.header.frame_id
            if not t.child_frame_id.startswith(self.prefix):
                t.child_frame_id = self.prefix + t.child_frame_id
        return msg

    def cb_tf(self, msg):
        self.pub_tf.publish(self._prefix_msg(msg))

    def cb_tf_static(self, msg):
        self.pub_tf_static.publish(self._prefix_msg(msg))


def main():
    rclpy.init()
    node = TfPrefixer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()