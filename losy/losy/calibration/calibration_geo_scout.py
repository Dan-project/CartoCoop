import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import NavSatFix, MagneticField
import math

from pyproj import Transformer


class GPSGeorefSubscriber(Node):
    def __init__(self):
        super().__init__('gps_georef_node')

        self.subscription = self.create_subscription(
            NavSatFix, '/fix', self.gps_callback, 10
        )

        # Abonnement au magnétomètre pour récupérer le cap (yaw)
        self.mag_subscription = self.create_subscription(
            MagneticField, '/imu/mag', self.mag_callback, 10
        )

        # coordonnées min pour normalisation, depuis la carte -> à modifier
        self.X_min = 1.0
        self.Y_min = 1.0

        # yaw courant, mis à jour en continu par mag_callback,
        # None tant qu'aucune donnée magnétomètre n'est reçue
        self.yaw = None

        # Déclaration des paramètres qui contiendront la pose géoréférencée,
        # valeurs par défaut à 0.0 tant qu'aucun fix GPS valide n'est reçu.
        self.declare_parameter('robot_x', 0.0)
        self.declare_parameter('robot_y', 0.0)
        self.declare_parameter('robot_yaw', 0.0)

        self.transformer = Transformer.from_crs('EPSG:4326', 'EPSG:2154', always_xy=True)

        self.already_computed = False  # pour ne calculer/écrire qu'une seule fois

    def mag_callback(self, msg: MagneticField):
        # Calcul du cap (yaw) à partir du champ magnétique dans le plan XY
        # atan2(y, x) donne l'angle par rapport à l'axe X du repère du capteur
        mx = msg.magnetic_field.x
        my = msg.magnetic_field.y
        mz = msg.magnetic_field.z
        self.yaw = math.atan2(-my, mx)

    def gps_callback(self, msg: NavSatFix):
        if self.already_computed:
            return  # on ignore les fix suivants, on ne fait le calcul qu'une fois

        if msg.status.status < 0:
            return

        if self.yaw is None:
            self.get_logger().warn("Pas encore de cap magnétomètre reçu, en attente...")
            return  # on attend d'avoir au moins une mesure de cap avant de calculer

        # Lat/Lon -> lambert 93
        x_lambert93, y_lambert93 = self.transformer.transform(msg.longitude, msg.latitude)

        # Dans le repère de ta carte (0,0)
        x_carte = x_lambert93 - self.X_min
        y_carte = y_lambert93 - self.Y_min

        self.get_logger().info(
            f"Pose Robot Carte -> X: {x_carte} m | Y: {y_carte} m | Yaw: {self.yaw} rad"
        )

        # Écriture des paramètres avec les valeurs calculées
        self.set_parameters([
            Parameter('robot_x', Parameter.Type.DOUBLE, x_carte),
            Parameter('robot_y', Parameter.Type.DOUBLE, y_carte),
            Parameter('robot_yaw', Parameter.Type.DOUBLE, self.yaw),
        ])

        self.get_logger().info("Paramètres robot_x / robot_y / robot_yaw mis à jour.")

        self.already_computed = True


def main(args=None):
    rclpy.init(args=args)
    node = GPSGeorefSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()