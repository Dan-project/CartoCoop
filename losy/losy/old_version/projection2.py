import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
from ultralytics import YOLO

# pour get trasnform function test
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener

class LidarImageFusionNode(Node):

    def __init__(self):
        super().__init__('lidar_image_fusion_node')

        # Initialisation OpenCV
        self.bridge = CvBridge()
        # Initialisation du modèle YOLO 26 nano semantique
        self.get_logger().info("Model YOLOv8-Segmentation :")
        self.model = YOLO("yolo26n-sem.pt") 
        
        # Subsribers caméra gauche et velodyne (On remplace le timer par des "subscribers passifs "on s'abonne aux topics sans callback direct pour pouvoir les synchroniser)
        self.image_sub = message_filters.Subscriber(self, Image, '/kitti/camera_color_left/image_raw') 
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, '/points_raw')
        
        # Temps d'aacquisitions syncrhonisé (le synchroniseur remplace le timer, il faut qu'ils arrivent avec un déalage max de 0.05 ici test à faire)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub], 
            queue_size=10, 
            slop=0.05
        )   
        self.ts.registerCallback(self.fusion_callback)
        self.get_logger().info('Synchronisation temporelle')

        # Matrices de calibration
        # calibration intrinsèque de la caméra gauche de couleur
        self.P2= np.array([
                    [7.18856000e+02, 0.00000000e+00, 6.07192800e+02,  4.53822500e+01],
                    [0.00000000e+00, 7.18856000e+02, 1.85215700e+02, -1.13088700e-01],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00,  3.77976100e-03]
                ])
        # calibration extrinsèque velodyne à caméra
        self.Tr = np.array([
            [4.27680239e-04, -9.99967248e-01, -8.08449168e-03, -1.19845993e-02],
            [-7.21062651e-03, 8.08119847e-03, -9.99941316e-01, -5.40398473e-02],
            [9.99973865e-01, 4.85948581e-04, -7.20693369e-03, -2.92196865e-01],
            [0,0,0,1]
                ])

        self.Tr_inverse = np.linalg.inv(self.Tr)
        self.P2_inverse = np.linalg.pinv(self.P2)

        self.W, self.H = 1242, 375  # taille image KITTI , A VERIf 1242  375
        self.only_lidar = False
        self.dict_labels={}
        self.point_index = 0 
        with open('semantique.txt','w') as f :
            f.write("Carte Sémantique")
    

        # pour get trasnform function test

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def fusion_callback(self, image_msg, lidar_msg):
        """ la fusion temporelle se fait naturellement grâce à l'initialisation passif des subscribers, puis on fait la fusion dans 'eapce on projettte 
            le lidar 3D vers le repère 2D de la caméra pour comparer les points du lidar avec les pixels de la caméra"""
    
        # self.get_logger().info('Callback actif')
        # Acquisitions des points  
        # points_lidar = pc2.read_points_numpy(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True) pose pb avec le nouveau bag qui a le ring dans le velodyne
        gen = pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points_lidar = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)
        
        # Mask 1 : réductions des points à ceux qui sont devant x>0
        mask = points_lidar[:, 0] > 0 # on garde dès que x est positif (car X pointe vers l'avant à verifier quand meme)
        points_front = points_lidar[mask]
        ones = np.ones((len(points_front), 1)) # homogénisation taille 4
        points_front_1 = np.hstack((points_front, ones)) # horizontal stack

        # Projection matricielle
        pts_cam = self.Tr @ points_front_1.T          # (4, N) dans le repère caméra,  points_front_1 : (N, 4) → on transpose pour avoir (4, N)
        pts_img = self.P2 @ pts_cam                   # (3, N) coordonnées homogènes image (pts_img = [u·w, v·w, w]), on est donc en coordonnées par pixels 

        # Mask 2 : réductions des points, on ne garde que les points devant la caméra profondeur w > 0 (pareil que Mask 1 mais certains point ne sont plus valides du à la distance ui séparer la caméra et le lidar, on garde quand même Mask 1 pour alléger les calculs)
        depth = pts_img[2, :] # w = profondeur dans le repère caméra
        valid_depth = depth > 0

        # Calculs des coordonnées 2D
        u = pts_img[0, valid_depth] / depth[valid_depth]
        v = pts_img[1, valid_depth] / depth[valid_depth]
        d = depth[valid_depth] # profondeur associée à chaque point projeté (donnée par le lidar toujours)

        # Mask 3 : réduction des points car ils ne se trouvent pas dans le champs de vision de la caméra
        in_frame = (u >= 0) & (u < self.W-1) & (v >= 0) & (v < self.H-1)
        # Nouvelles coordonnées
        u_valid = u[in_frame].astype(np.int32)
        v_valid = v[in_frame].astype(np.int32)
        d_valid = d[in_frame]
        
        # Image conversion ROS to OpenCV
        try:
            if self.only_lidar:
                cv_image = np.zeros((self.H, self.W, 3), dtype=np.uint8)  # fond noir
            else :
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Erreur de conversion OpenCV : {e}")
            return

        # Plot les points lidar valide sur la frame image reçu par la caméra
        # Coloriage des points selon la profondeur (proche=rouge, loin=bleu), on normalise la profondeur entre 0 et 255 sur une plage [0, 50] mètres
        d_norm = np.clip(d_valid / 50.0, 0.0, 1.0)            # 0.0 (proche)  1.0 (loin)
        d_uint8 = (d_norm * 255).astype(np.uint8)             # passage en uint8 pour colormap
        colors = cv2.applyColorMap(d_uint8.reshape(-1, 1), cv2.COLORMAP_JET)  # (N,1,3)
        colors = colors.reshape(-1, 3)                         # (N, 3) BGR
        for i in range(len(u_valid)):
            color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
            cv2.circle(cv_image, (u_valid[i], v_valid[i]), radius=1, color=color, thickness=-1)

        cv2.imshow("LiDAR projeté sur caméra gauche", cv_image)
        cv2.waitKey(1)

        # Yolo
    
        results = self.model(cv_image)
        result = results[0]
        # for i in range(len(u_valid)): # ecris dans un dictionnaire et sur le txt les labels et coordnne 2D des points lidar 
        #     name= self.classYolo(result ,u_valid[i], v_valid[i])
        #     c = [u_valid[i],v_valid[i]]
        #     self.dict_labels[name] = c
        #     self.write(name, u_valid[i], v_valid[i])
       
        pts_lidar_cb = self.comeback3D(u_valid,v_valid,d_valid) # repere du ldiar
        self.get_logger().info(f"len pt ldiar {len(pts_lidar_cb)}")
        pts_monde = self.get_transform_world(pts_lidar_cb)
        
        if pts_monde is None: # sécurité
            self.get_logger().info("on perd une frame !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return
        else :
            self.get_logger().info(f"len pt lidars {len(pts_monde)}")
        
        names = self.classYolo(result, u_valid, v_valid)
        for i in range(len(u_valid)): # ici pareil mais on note les coordonnée dan sle repère monde, facile car le point en 2d a l'index i correspond au point en 3D repère ponde à l'index i également
            # name = self.classYolo(result, u_valid[i], v_valid[i])
            x_m, y_m, z_m = pts_monde[i]
            # self.dict_labels[name] = [x_m, y_m, z_m]
            # self.write(name, [x_m, y_m, z_m])
            self.dict_labels[self.point_index] = [names[i], x_m, y_m, z_m]
            self.point_index += 1
            

    # def classYolo(self,r,u,v):
    #     """
    #     r from results
    #     coord : u and v coordonnées d'un point lidar projeté dans le repère de la camera en 2D
    #     """
    #     carte = r.semantic_mask.data
    #     id = carte[v,u].item()
    #     name = r.names[int(id)]

    #     return name

    def classYolo(self, r, u_valid, v_valid):
        """
        r : result YOLO
        u_valid, v_valid : tableaux numpy de coordonnées de tous les points lidar projetés
        retourne : tableau numpy de noms de labels (N,)
        """
        carte = r.semantic_mask.data
        ids = carte[v_valid, u_valid]
        names = np.array([r.names[int(i)] for i in ids])
        return names

    # def write(self, label, list):
    #     with open('semantique.txt', 'a') as f:
    #         f.write(f"\n{label},{list[0]},{list[1]},{list[2]}")

    # def write_all(self):
    #     self.get_logger().info("writing")
    #     with open('semantique.txt', 'a') as f:
    #         for name, coords in self.dict_labels.items():
    #             f.write(f"\n{name},{coords[0]},{coords[1]},{coords[2]}")

    def write_all(self):
        with open('semantique.txt', 'w') as f:
            f.write("Carte Sémantique")
            for idx, data in self.dict_labels.items():
                f.write(f"\n{data[0]},{data[1]},{data[2]},{data[3]}")


    def destroy_node(self):
        self.write_all()  # une seule écriture, quand on fait Ctrl+C
        super().destroy_node()


    # def comeback3D(self,u_img,v_img,d_img):
    #     """
    #     coord lidar repère image 
    #     """

    #     d = d_img
    #     u = u_img*d
    #     v = v_img*d

    #     coord_img_new = np.array([u, v, d])  
    #     # profondeur associée à chaque point projeté (donnée par le lidar toujours)


    #     ones = np.ones((len(coord_img_new), 1)) # homogénisation taille 4
    #     coord_img_h = np.hstack((coord_img_new, ones)) # horizontal stack
    #     pts_cam = self.P2_inverse @ coord_img_h
    #     pts_lidar = self.Tr_inverse @ pts_cam

    #     return pts_lidar

    def comeback3D(self, u_img, v_img, d_img):
        """
        Reconstruit les points 3D dans le repère LiDAR à partir des pixels 2D et de la profondeur.
        """
        u = u_img * d_img
        v = v_img * d_img
        d = d_img

        # Création d'une matrice (3, N)
        coord_img_new = np.vstack((u, v, d))  

        # Passage au repère Caméra (4, N) via la pseudo-inverse de P2
        pts_cam = self.P2_inverse @ coord_img_new
        
        # Passage au repère LiDAR (4, N) via l'inverse de Tr
        pts_lidar_h = self.Tr_inverse @ pts_cam

        # On ignore la ligne homogène la 4ème et on transpose pour obtenir une forme (N, 3)
        pts_lidar = pts_lidar_h[:3, :].T

        return pts_lidar


    def get_transform_world(self, pts_lidar_3d: np.ndarray) -> np.ndarray | None:
        """
        Transforme des points 3D du repère LiDAR vers le repère monde via TF LIO-SAM.
        pts_lidar_3d : (N, 3)
        retourne : (N, 3) dans le repère monde, ou None si TF indisponible
        """
        self.get_logger().info("Generation coordnnées 3D repère Monde :")
        try:
            t = self.tf_buffer.lookup_transform(
                'world',
                'velodyne',
                rclpy.time.Time()
            )

            q = t.transform.rotation
            tr = t.transform.translation
            rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

            T_world_lidar = np.eye(4)
            T_world_lidar[:3, :3] = rot
            T_world_lidar[:3, 3] = [tr.x, tr.y, tr.z]

            ones = np.ones((len(pts_lidar_3d), 1))
            pts_h = np.hstack((pts_lidar_3d, ones))   # (N, 4)
            pts_monde = (T_world_lidar @ pts_h.T).T    # (N, 4)

            return pts_monde[:, :3]                     # (N, 3)

        except Exception as e:
            self.get_logger().warn(f"TF indisponible : {e}")
            return None
        

        

# def main(args=None):
#     rclpy.init(args=args)
#     node = LidarImageFusionNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()



def main(args=None):
    rclpy.init(args=args)
    node = LidarImageFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Fermeture : écriture du fichier...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()