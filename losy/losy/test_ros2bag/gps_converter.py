#!/usr/bin/env python3
"""
ROS2 Node : GPS Fix -> XYZ (ENU local)
pip install pymap3d
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PointStamped
import pymap3d as pm


class GpsToXyz(Node):
    def __init__(self):
        super().__init__('gps_to_xyz')
        self.origin = None  # (lat0, lon0, alt0) — fixé au 1er fix

        self.sub = self.create_subscription(NavSatFix, '/gps/fix', self.callback, 10)
        self.pub = self.create_publisher(PointStamped, '/gps/xyz', 10)
        self.get_logger().info('Prêt — en attente du premier fix GPS...')

    def callback(self, msg: NavSatFix):
        if msg.status.status < 0:
            return  # fix invalide

        lat, lon, alt = msg.latitude, msg.longitude, msg.altitude

        if self.origin is None:
            self.origin = (lat, lon, alt)
            self.get_logger().info(f'Origine : lat={lat:.6f} lon={lon:.6f} alt={alt:.1f}m')

        # Une seule ligne de conversion !
        e, n, u = pm.geodetic2enu(lat, lon, alt, *self.origin)

        out = PointStamped()
        out.header = msg.header
        out.point.x, out.point.y, out.point.z = e, n, u
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GpsToXyz()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()