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
        
        # keep the last message for interpolation
        self.prev_msg = None
        self.target_hz = 200
        self.source_hz = 10

    def callback(self, msg):
        if self.prev_msg is None: # for first message
            self.prev_msg = msg
            return

        # Nb messages to interpolate
        n_steps = self.target_hz // self.source_hz  # 20 messages

        t0 = self.to_sec(self.prev_msg.header.stamp)
        t1 = self.to_sec(msg.header.stamp) # current time, on header

        for i in range(n_steps):
            alpha = i / n_steps  # 0.0 → 0.95

            interp_msg = Imu()
            interp_msg.header.frame_id = msg.header.frame_id

            # Timestamp interpolated
            t_interp = t0 + alpha * (t1 - t0)
            interp_msg.header.stamp.sec = int(t_interp)
            interp_msg.header.stamp.nanosec = int((t_interp % 1) * 1e9)

            # Interpolation of linear acceleration
            interp_msg.linear_acceleration.x = self.lerp(
                self.prev_msg.linear_acceleration.x, msg.linear_acceleration.x, alpha)
            interp_msg.linear_acceleration.y = self.lerp(
                self.prev_msg.linear_acceleration.y, msg.linear_acceleration.y, alpha)
            interp_msg.linear_acceleration.z = self.lerp(
                self.prev_msg.linear_acceleration.z, msg.linear_acceleration.z, alpha)

            # Interpolation  of linear angular velocity
            interp_msg.angular_velocity.x = self.lerp(
                self.prev_msg.angular_velocity.x, msg.angular_velocity.x, alpha)
            interp_msg.angular_velocity.y = self.lerp(
                self.prev_msg.angular_velocity.y, msg.angular_velocity.y, alpha)
            interp_msg.angular_velocity.z = self.lerp(
                self.prev_msg.angular_velocity.z, msg.angular_velocity.z, alpha)

            # SLERP for the oriantation (because quaternion) find this  from scipy and it works well 
            q0 = [self.prev_msg.orientation.x, self.prev_msg.orientation.y,
                  self.prev_msg.orientation.z, self.prev_msg.orientation.w]
            q1 = [msg.orientation.x, msg.orientation.y,
                  msg.orientation.z, msg.orientation.w]
            
            rotations = Rotation.from_quat([q0, q1])
            slerp = Slerp([0, 1], rotations)
            q_interp = slerp(alpha).as_quat()

            interp_msg.orientation.x = q_interp[0]
            interp_msg.orientation.y = q_interp[1]
            interp_msg.orientation.z = q_interp[2]
            interp_msg.orientation.w = q_interp[3]

            # Covariances , non zero for lio sam 
            interp_msg.orientation_covariance[0] = 0.001
            interp_msg.orientation_covariance[4] = 0.001
            interp_msg.orientation_covariance[8] = 0.001
            interp_msg.angular_velocity_covariance[0] = 0.001
            interp_msg.angular_velocity_covariance[4] = 0.001
            interp_msg.angular_velocity_covariance[8] = 0.001
            interp_msg.linear_acceleration_covariance[0] = 0.001
            interp_msg.linear_acceleration_covariance[4] = 0.001
            interp_msg.linear_acceleration_covariance[8] = 0.001

            self.pub.publish(interp_msg) 
            # francais à mettre en anglais 
            #   attenntion ! ici cette boucle for se fait quasiment instantannement donc les 20 messages d'interpolation vont être quasiment envoyé en meme temps 
            # puis le callback est rappelé toute les 100ms (car kitti 10 hz, callbak appelé tout les 100ms alors) comme avant, 
            # donc 20 messages par 10ms aun lieu de 1 message par 100ms, ce qui nous donne une frequence d'envoie ( ros2 topic hz) d'environ 200.
            #  Cependant le header stamp est bien crée de la bonne facon cahque message un header stamp qui évolue normalement comme si on publié réellement à 200hz, 
            # et c'ets ceci que LIo sam va lire, pas la frequence réelle des envoies des topics.
            # Donc normalement cela marche bien , on remarque aussi un léger déphasage entre les données kitti et ceux interpolé causé par l'interpolation,
            #  peut etre essayé également un autre type d'interpolation cela peut etre interressant 

        self.prev_msg = msg

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
