import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

class ImuInterpolator(Node):
    def __init__(self):
        super().__init__('imu_interpolator')

        self.sub = self.create_subscription(Imu, '/kitti/imu', self.callback, 10)
        self.pub = self.create_publisher(Imu, '/imu_raw_f', 10)

        self.target_hz  = 200
        self.source_hz  = 10
        self.n_steps    = self.target_hz // self.source_hz  # 20

        # Two consecutive KITTI messages used for interpolation
        self.prev_msg = None
        self.curr_msg = None

        # Which interpolation step we are currently at (0..n_steps-1)
        self.step = 0

        # Timer fires at target_hz (every 5 ms) and publishes ONE message
        period_sec = 1.0 / self.target_hz
        self.timer = self.create_timer(period_sec, self.timer_callback)

    # ------------------------------------------------------------------
    # Called every ~100 ms when a new KITTI IMU message arrives
    # ------------------------------------------------------------------
    def callback(self, msg):
        if self.prev_msg is None:
            self.prev_msg = msg
            return

        # New pair available: reset interpolation window
        self.prev_msg = self.curr_msg if self.curr_msg is not None else msg
        self.curr_msg = msg
        self.step = 0

    # ------------------------------------------------------------------
    # Called every 5 ms by the timer → publishes exactly ONE message
    # ------------------------------------------------------------------
    def timer_callback(self):
        # Not enough data yet
        if self.prev_msg is None or self.curr_msg is None:
            return

        # If we have consumed all steps for this window, stay on last step
        # (avoids publishing stale data; LIO-SAM will just get repeated last frame)
        step = min(self.step, self.n_steps - 1)
        alpha = step / self.n_steps  # 0.0 → 0.95

        t0 = self.to_sec(self.prev_msg.header.stamp)
        t1 = self.to_sec(self.curr_msg.header.stamp)

        interp_msg = Imu()
        interp_msg.header.frame_id = self.curr_msg.header.frame_id

        # Interpolated timestamp
        t_interp = t0 + alpha * (t1 - t0)
        interp_msg.header.stamp.sec    = int(t_interp)
        interp_msg.header.stamp.nanosec = int((t_interp % 1) * 1e9)

        # Linear interpolation – acceleration
        interp_msg.linear_acceleration.x = self.lerp(
            self.prev_msg.linear_acceleration.x, self.curr_msg.linear_acceleration.x, alpha)
        interp_msg.linear_acceleration.y = self.lerp(
            self.prev_msg.linear_acceleration.y, self.curr_msg.linear_acceleration.y, alpha)
        interp_msg.linear_acceleration.z = self.lerp(
            self.prev_msg.linear_acceleration.z, self.curr_msg.linear_acceleration.z, alpha)

        # Linear interpolation – angular velocity
        interp_msg.angular_velocity.x = self.lerp(
            self.prev_msg.angular_velocity.x, self.curr_msg.angular_velocity.x, alpha)
        interp_msg.angular_velocity.y = self.lerp(
            self.prev_msg.angular_velocity.y, self.curr_msg.angular_velocity.y, alpha)
        interp_msg.angular_velocity.z = self.lerp(
            self.prev_msg.angular_velocity.z, self.curr_msg.angular_velocity.z, alpha)

        # SLERP – orientation (quaternion)
        q0 = [self.prev_msg.orientation.x, self.prev_msg.orientation.y,
              self.prev_msg.orientation.z, self.prev_msg.orientation.w]
        q1 = [self.curr_msg.orientation.x, self.curr_msg.orientation.y,
              self.curr_msg.orientation.z, self.curr_msg.orientation.w]

        rotations = Rotation.from_quat([q0, q1])
        slerp     = Slerp([0, 1], rotations)
        q_interp  = slerp(alpha).as_quat()

        interp_msg.orientation.x = q_interp[0]
        interp_msg.orientation.y = q_interp[1]
        interp_msg.orientation.z = q_interp[2]
        interp_msg.orientation.w = q_interp[3]

        # Covariances (non-zero diagonal required by LIO-SAM / GTSAM)
        interp_msg.orientation_covariance[0]         = 0.001
        interp_msg.orientation_covariance[4]         = 0.001
        interp_msg.orientation_covariance[8]         = 0.001
        interp_msg.angular_velocity_covariance[0]    = 0.001
        interp_msg.angular_velocity_covariance[4]    = 0.001
        interp_msg.angular_velocity_covariance[8]    = 0.001
        interp_msg.linear_acceleration_covariance[0] = 0.001
        interp_msg.linear_acceleration_covariance[4] = 0.001
        interp_msg.linear_acceleration_covariance[8] = 0.001

        self.pub.publish(interp_msg)

        # Advance step counter only while inside the current window
        if self.step < self.n_steps - 1:
            self.step += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def lerp(self, a, b, alpha):
        return a + alpha * (b - a)

    def to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9


def main():
    rclpy.init()
    node = ImuInterpolator()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
