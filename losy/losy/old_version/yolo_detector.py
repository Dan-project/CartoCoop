

import cv2
import rclpy
from rclpy.node import Node
# sensor_msgs.msg.Image
from ultralytics import YOLO
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge

class YoloDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')
        
        self.subscription = self.create_subscription(
            Image,
            '/kitti/camera_color_left/image_rect',
            self.listener_callback,
            10)
        
        self.bridge = CvBridge()
        self.get_logger().info("Init")

        self.get_logger().info("Model YOLOv8-Segmentation :")
        self.model = YOLO("yolo26n-sem.pt") 


    def listener_callback(self, msg):
        self.get_logger().info("Get it")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
        except Exception as e:
            self.get_logger().error(f"Erreur de conversion OpenCV : {e}")

        results = self.model(cv_image, stream=True)
        annotated_frame = cv_image
        for r in results:
            annotated_frame = r.plot()

        cv2.imshow("Camera left", annotated_frame)
        cv2.waitKey(1)


    def classYolo(sels,r,coord):
        """
        r from results
        coord : u and v coordonnées d'un point lidar projeté dans le repère de la camera en 2D
        """
        u,v=coord[0],coord[1]

        if r.masks is not None:
            
            semantic_map = r.masks.data[0].cpu().numpy().astype(int)
            
            classe_du_pixel = semantic_map[v, u] 
            
        if classe_du_pixel == 8:
            print("C'est un arbre !")


def main(args=None):
    rclpy.init(args=args)
    detector = YoloDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()