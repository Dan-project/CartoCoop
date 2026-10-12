import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np

class CheckRingTime(Node):
    def __init__(self):
        super().__init__('check_ring_time')
        self.sub = self.create_subscription(
            PointCloud2, '/velodyne', self.callback, 1)

    def callback(self, msg):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        n = msg.width
        
        rings = []
        times = []
        for i in range(n):
            # ring à offset 16, uint16 = 2 bytes
            ring_offset = i * 22 + 16
            ring = np.frombuffer(raw[ring_offset:ring_offset+2], dtype=np.uint16)[0]
            rings.append(ring)
            
            # time à offset 18, float32 = 4 bytes
            time_offset = i * 22 + 18
            time = np.frombuffer(raw[time_offset:time_offset+4], dtype=np.float32)[0]
            times.append(time)
        
        rings = np.array(rings)
        times = np.array(times)

        print("=== RING ===")
        print(f"  min:            {rings.min()}")
        print(f"  max:            {rings.max()}")
        print(f"  valeurs uniques: {np.unique(rings)}")

        print("=== TIME ===")
        print(f"  min:  {times.min():.6f} s")
        print(f"  max:  {times.max():.6f} s")
        print(f"  mean: {times.mean():.6f} s")

 
def main():
    rclpy.init()
    node = CheckRingTime()
    rclpy.spin(node)

if __name__ == '__main__':
    main()