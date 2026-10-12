# Dépendances
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.duration import Duration
import cv2 
import json
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.spatial import cKDTree
import message_filters
from tf2_ros import Buffer, TransformListener
from collections import deque
from types import SimpleNamespace
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # pour plot
from mpl_toolkits.mplot3d import Axes3D 
from shapely.geometry import Point, Polygon, box
# Types de messages
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import String, Header
import time



class PerceptionNode(Node):

# --------------------------------- INIT ----------------------------------

    def __init__(self):
        super().__init__('perception_node')
        # ----- Déclaration des paramètres -----

        # choix du robot
        self.declare_parameter('robot','scout') # scout ou ranger (j'ai a duex acquisitions dispo, donc le choix du robot selectionne les bon paramètres dans la section "paramètres à initialiser".
        self.robot = self.get_parameter('robot').value

        # topic output
        self.declare_parameter('output_topic', 'semantic_points')

        # position robot + param de normalisation de la map
        self.declare_parameter('xinit_robot_scout',709809.507192942)
        self.declare_parameter('yinit_robot_scout',7056662.075542638)
        self.declare_parameter('xinit_robot_ranger',709735.5419107936 )
        self.declare_parameter('yinit_robot_ranger',7056805.517686295)

        self.declare_parameter('xmin_map',709433.5004293)
        self.xmin_map = self.get_parameter('xmin_map').value
        self.declare_parameter('ymin_map',7056437.95734587)
        self.ymin_map = self.get_parameter('ymin_map').value

        # paramètre d'acquisition et fusion de donnée
        self.declare_parameter('acquisition_hz', 2.0)
        self.acquisition_hz = self.get_parameter('acquisition_hz').value
        self.min_interval = 1.0 / self.acquisition_hz   # ex: 0.5s si 2Hz
        self.last_processed_time = None     
        self.declare_parameter('range',20) # range du lidar en mètres
        self.max_range = self.get_parameter('range').value

        # paramètre de fermeture de boucle
        self.declare_parameter('loop', False) # activer ou désactiver la loop closure (seulement ici liosam/liorf est indépendant, si la loop closure de liosam est activé elle n'aura aucune influence sur les données de notre map. Cependant, si on veut activer la loop closure, il faut l'activer ici est dans liosam/liorf)
        self.loop_closure = self.get_parameter('loop').value

        # topic lidar et camera
        topic_image = f'/{self.robot}/camera/color/image_raw'
        topic_lidar = f'/{self.robot}/ouster/points'

        if self.robot == "scout" :
            # coordonnées et oriantation du robot ()
            self.x_init_robot = self.get_parameter('xinit_robot_scout').value# valeurs données par le gps (environ 6 mètres de covariance)
            self.y_init_robot = self.get_parameter('yinit_robot_scout').value # valeurs données par le gps (environ 6 mètres de covariance)
            self.angle_init_robot = -0.09988048237230192 + np.pi/4 # valeurs données par l'imu mag (totalement faux, rajout pour être correcte et diffère selon les acquisitions donc ne pas si fier). Trouver une méthode correcte pour connaitre l'orientation du robot à l'initialisation
            # matrice de calibration intrinsèque de la caméra (topic camear info)
            self.P2 = np.array([
                [912.8180541992188, 0.0,               631.4204711914062, 0.0],
                [0.0,               912.3511962890625, 358.36956787109375, 0.0],
                [0.0,               0.0,               1.0,                0.0]
            ])
            # matrice de calibration extrinsèque lidar à caméra (voir la transformation entre ces deux frames, exemple os_lidar et camera_optical_frame)
            self.Tr = np.array([
                [ 0.000,  1.000,  0.000,  0.032],
                [ 0.000,  0.000, -1.000, -0.251],
                [-1.000,  0.000,  0.000, -0.262],
                [ 0.000,  0.000,  0.000,  1.000]
            ])
            # résolution caméra (topic camera info)
            self.W, self.H = 1280,720 
        
        elif self.robot == "ranger" :
            # coordonnées et oriantation du robot ()
            self.x_init_robot = self.get_parameter('xinit_robot_ranger').value # valeurs données par le gps (environ 6 mètres de covariance)
            self.y_init_robot = self.get_parameter('yinit_robot_ranger').value # valeurs données par le gps (environ 6 mètres de covariance)
            self.angle_init_robot = -0.447613997675286 - np.pi/2 # valeurs données par l'imu mag (totalement faux, rajout de pi/2 pour être correcte et diffère selon les acquisitions donc ne pas si fier). Trouver une méthode correcte pour connaitre l'orientation du robot à l'initialisation
            # matrice de calibration intrinsèque de la caméra (topic camear info)
            self.P2 = np.array([
                [905.9884643554688, 0.0,               648.2288818359375, 0.0],
                [0.0,               905.703857421875,  367.8875427246094, 0.0],
                [0.0,               0.0,               1.0,               0.0]
            ])
            # matrice de calibration extrinsèque lidar à caméra (voir la transformation entre ces deux frames, exemple os_lidar et camera_optical_frame)
            self.Tr = np.array([
                [ 0.004,  1.000,  0.006,  0.035],
                [ 0.006,  0.006, -1.000, -0.390],
                [-1.000,  0.004, -0.006, -0.224],
                [ 0.000,  0.000,  0.000,  1.000]
            ])
            # résolution caméra (topic camera info)
            self.W, self.H = 1280,720 

        else:
            raise ValueError(f"Robot inconnu: '{self.robot}' (attendu: 'scout' ou 'ranger')")

        # normalisation par rapport a l'origine de la carte a priori
        self.x_init_robot_norm = self.x_init_robot - self.xmin_map #  le node fusion n'a rien a recalculé , tout sera envoyé dans le repère final
        self.y_init_robot_norm = self.y_init_robot - self.ymin_map
        # Matrice de rotation initiale à appliquer au repère odom
        c, s = np.cos(self.angle_init_robot), np.sin(self.angle_init_robot)
        self.R_init = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])


        # ----- Initialisation des modèles -----

        # Initialisation de OpenCV
        self.bridge = CvBridge()
        # Initialisation du modèle YOLO 26 nano semantique
        self.get_logger().info(f"[{self.robot}] Chargement du modele YOLO.")
        self.model = YOLO("yolo26n-sem.pt") 
        


        # ----- Subscribers -----  
         
        # subscribers passifs
        self.image_sub = message_filters.Subscriber(self, Image, topic_image) 
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, topic_lidar)
        # subsriber odometry
        topic_name = f'/{self.robot}/liorf/mapping/odometry'
        self.odom_sub = self.create_subscription(Odometry, topic_name, self.odom_callback, 10)
        # Stockage des odometry recu par liorf/liosam
        self.odom_history = deque(maxlen=200)  # (t_sec, msg)


        # ----- Fusion -----

        # Temps d'acquisitions syncrhonisé (le synchroniseur remplace le timer, il faut qu'ils arrivent avec un décalage max de 0.05 ici test à faire)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub], 
            queue_size=10, 
            slop=0.05
        )  
        # Appelle du callback principale fusion tout les ts 
        self.ts.registerCallback(self.fusion_callback)
        self.get_logger().info('Synchronisation temporelle : Fusion Callback')
        # Stockage des frames en attentes
        self.pending_frames = deque(maxlen=30)  # taille limite pour pas saturer



        # ----- Loop Closure -----

        # rajout de la partie detection de loop closure 
        # if self.loop_closure :
        #     self.loop_closure_detected = False
        #     self.loop_sub = self.create_subscription(MarkerArray, f'/{self.robot}liorf/mapping/loop_closure_constraints', self.loop_callback, 10)
        #     self.marker_history = 0 

        #     self.path_sub = self.create_subscription(Path, f'/{self.robot}liorf/mapping/path', self.path_callback, 10) # contient toutes les positions, on récupère seulement quand il y a un loop closure 

        #     self.pending_frames_until_loop = []

        #     self.loop_closure_display_until = 0.0 # pour afficher sur open cv

        #     self.corrected_path = []

        #     self.new_data_dict = {feature: [] for feature in self.features} #  {'id': [], 'center_x': [], 'center_y': [], 'label': [], 'time': []}

        if self.loop_closure:
            self.loop_closure_detected = False
            self.loop_sub = self.create_subscription(MarkerArray, f'/{self.robot}liorf/mapping/loop_closure_constraints', self.loop_callback, 10)
            self.marker_history = 0 

            self.path_sub = self.create_subscription(Path, f'/{self.robot}/liorf/mapping/path', self.path_callback, 10)

            self.all_frames = deque()  # historique COMPLET, non borné, depuis le début
            self.corrected_path = []
            self.loop_closure_display_until = 0.0
            # self.recomputing = False
            # self.buffer_during_recompute = deque() # pour ne pas perdre els frames pendant le calcul du loop closure 

            # publisher de contrôle pour signaler un reset à map_fusion_node
            self.control_pub = self.create_publisher(String, f"/{self.robot}/semantic_points_control", 10)

        # ----- Publishers -----
        
        # out_topic = self.get_parameter('output_topic').value
        out_topic = f"/{self.robot}/{self.get_parameter('output_topic').value}"

        self.points_pub = self.create_publisher(PointCloud2, out_topic, 10)

        latched_qos = QoSProfile(depth=1)
        latched_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        latched_qos.reliability = QoSReliabilityPolicy.RELIABLE
        self.labels_pub = self.create_publisher(String, out_topic + '_labels', latched_qos)
        self._labels_published = False
 
        self.get_logger().info(
            f"[{self.robot}] Perception node pret, publication sur '{out_topic}'")
 
# ------------------------------- CALLBACK -------------------------------

    def odom_callback(self, msg):
        t = self._stamp_to_sec(msg.header.stamp)
        self.odom_history.append((t, msg))
        # debug
        # self.get_logger().info(f"odom reçu t={t:.3f}, historique={len(self.odom_history)}")

    def loop_callback(self,msg):
        # loop closure detection test
        # debug 
        # for i, m in enumerate(msg.markers):
        #     self.get_logger().info(f"marker {i}: ns={m.ns}, type={m.type}, n_points={len(m.points)}")

        if len(msg.markers) < 2: # si il y a rien , car markers  : marker 0 (nodes) et markers 1 (edges)
            return

        edges_marker = msg.markers[1]  # ns=loop_edges, type=5 (LINE_LIST)
        n_constraints = len(edges_marker.points) / 2 # /2 car 1 pts départ et 1 arrivé, donc 1 contraintes = 2 pts nouveaux

        if n_constraints > self.marker_history:
            self.get_logger().info(f"Loop Closure détecté ({n_constraints} : contraintes)")
            self.loop_closure_detected = True
            self.loop_closure_display_until = time.time() + 1.0  # affiché x s sur l'image
        self.marker_history = n_constraints

    def path_callback(self, msg):
        """On ne traite le Path que si un loop closure vient d'être détecté,
        sinon on ignore (le topic path publie en continu, indépendamment des loop closures)."""
        if not self.loop_closure_detected or not msg.poses:
            return
        
        # debug
        # self.get_logger().info(f"Path reçu")
        self.corrected_path = msg.poses # il faut verif le time stamp pour la syncrho des pts 
        self.loop_closure_detected = False

        self.recompute_all_points()  # à définir, 

    def fusion_callback(self, image_msg, lidar_msg):
        """ la fusion temporelle se fait naturellement grâce à l'initialisation passif des subscribers, puis on fait la fusion dans l'espace on projette 
            le lidar 3D vers le repère 2D de la caméra pour comparer les points du lidar avec les pixels de la caméra"""

        # Bloc suivant permets de changer la fréquence du callback, exemple avec kitti camera et lidar = 10hz 
        # ==> trop rapide car yolo et le reste n'a pas le temps de tout calculer donc grace a ce bloc on peut faire attendre le callback avec un min_intervale qui vient de la fréquence chosis dans les paramètres
        t_img = self._stamp_to_sec(image_msg.header.stamp)
        if self.last_processed_time is not None:
            if (t_img - self.last_processed_time) < self.min_interval:
                return
        self.last_processed_time = t_img

        # Lecture des points Lidar 
        gen = pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points_lidar = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)

        # Création de masques pour ne garder que les points lidar utilisable pour les comparer avec l'image caméra 
        # Mask 1 : devant x>0
        # mask = points_lidar[:, 0] > 0
        # points_front = points_lidar[mask]
        # ones = np.ones((len(points_front), 1))
        # points_front_1 = np.hstack((points_front, ones))

        ones = np.ones((len(points_lidar), 1))
        points_front_1 = np.hstack((points_lidar, ones))

        # Projection matricielle --> points lidar dans le repère de la caméra pts_image
        pts_cam = self.Tr @ points_front_1.T
        pts_img = self.P2 @ pts_cam

        # Mask 2 : profondeur positive (normalement c'est un doublon car on le fait avec x>0 avant (enfaite ca dépend si la caméra et devant le Lidar)
        depth = pts_img[2, :]
        valid_depth = depth > 0

        u = pts_img[0, valid_depth] / depth[valid_depth]
        v = pts_img[1, valid_depth] / depth[valid_depth]
        d = depth[valid_depth]

        # points_mask2 = points_front[valid_depth] # on applique le mask 2 aussi à points_front_1 pour pouvoir retrouver les points non transformé (pour pas devoir faire une retransformation inverse)
        points_mask2 = points_lidar[valid_depth] # on applique le mask 2 aussi à points_front_1 pour pouvoir retrouver les points non transformé (pour pas devoir faire une retransformation inverse)

        # Mask 3 : dans le champ de vision de la caméra
        in_frame = (u >= 0) & (u < self.W-1) & (v >= 0) & (v < self.H-1)
        u_valid = u[in_frame].astype(np.int32)
        v_valid = v[in_frame].astype(np.int32)
        d_valid = d[in_frame]

        points_mask3 = points_mask2[in_frame] # pareil avec mask 3 

        # Mask 4 : distance max
        range_mask = d_valid < self.max_range
        u_valid = u_valid[range_mask]
        v_valid = v_valid[range_mask]
        d_valid = d_valid[range_mask]

        points_mask4 = points_mask3[range_mask] # pareil avec mask 4
        
        # Image conversion ROS to OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"[{self.robot}] Erreur de conversion OpenCV : {e}")
            return
        
        # self.get_logger().info(
        #     f"points_lidar={len(points_lidar)}, points_front(x>0)={len(points_front)}, "
        #     f"valid_depth={valid_depth.sum()}, in_frame={in_frame.sum()}, "
        #     f"u_valid final={len(u_valid)}"
        # )

        # YOLO 
        results = self.model(cv_image)
        result = results[0]

        # publie la table label_id -> nom une seule fois (fixe pour un modele , permet de mettre ca dans le message point cloud )
        if not self._labels_published:
            self.labels_pub.publish(String(data=json.dumps(result.names)))
            self._labels_published = True
            self.get_logger().info(f"[{self.robot}] Table de labels publiee.")
 
        label_ids = self.classYolo(result, u_valid, v_valid)
 

        
        # Appelle de la fonction pour afficher sur opencv les pts lidar et les mask yolo
        self.show_opencv(u_valid,v_valid,d_valid,cv_image,result)

       
        # Création d'une queue de frames pour 
        self.pending_frames.append({
            'points': points_mask4,
            'label_ids': label_ids,  
            't_img': t_img,
            'stamp': lidar_msg.header.stamp,
            'range' : d_valid,
        })

        if self.loop_closure:
            self.all_frames.append({
                'points': points_mask4,
                'label_ids': label_ids,
                't_img': t_img,
                'stamp': lidar_msg.header.stamp,
                'range': d_valid,
            })
        # Appelle de la fonction qui gèere la queue
        self.try_process_pending()

# ---------------------------- USEFUL FUNCTION ---------------------------

    def _stamp_to_sec(self, stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    def show_opencv(self, u_valid, v_valid, d_valid, cv_image, result):
        # Afficher les images avec les mask yolo sur open cv en temps réel
        cv_image_mask, palette = self.semantic_mask(cv_image, result)

        if len(u_valid) > 0:  # sécurité 
            
            # Afficher les points Lidar sur le open cv en temps réel
            d_norm = np.clip(d_valid / 50.0, 0.0, 1.0)
            d_uint8 = (d_norm * 255).astype(np.uint8)
            colors = cv2.applyColorMap(d_uint8.reshape(-1, 1), cv2.COLORMAP_JET)
            colors = colors.reshape(-1, 3)
            for i in range(len(u_valid)):
                color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
                cv2.circle(cv_image_mask, (u_valid[i], v_valid[i]), radius=1, color=color, thickness=-1)

        else:
            self.get_logger().warn("Aucun point Lidar valide dans le champ de vision caméra.")

        # Légende des classes sémantiques
        cv_image_mask = self.draw_legend(cv_image_mask, result, palette)

        # Affichage temporaire du message loop closure
        if self.loop_closure:
            if time.time() < self.loop_closure_display_until:
                cv2.putText(cv_image_mask, "LOOP CLOSURE DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Masque sémantique + lidar projeté", cv_image_mask)
        cv2.waitKey(1)

    def semantic_mask(self, cv_image, result, alpha=0.5):
        carte = result.semantic_mask.data
        if hasattr(carte, 'cpu'):  # si c'est un tensor torch
            carte = carte.cpu().numpy()
        carte = np.asarray(carte).astype(np.int32)
        carte = np.squeeze(carte)
        num_classes = max(len(result.names), int(carte.max()) + 1)
        cmap = (np.linspace(0, 255, num_classes)).astype(np.uint8)
        palette = cv2.applyColorMap(cmap.reshape(-1, 1), cv2.COLORMAP_HSV).reshape(-1, 3)

        carte_clipped = np.clip(carte, 0, num_classes - 1)
        mask_overlay = palette[carte_clipped]

        if mask_overlay.shape[:2] != cv_image.shape[:2]:
            mask_overlay = cv2.resize(mask_overlay, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask_overlay = mask_overlay.astype(np.uint8)
        blended = cv2.addWeighted(cv_image, 1 - alpha, mask_overlay, alpha, 0)

        return blended, palette  # on retourne aussi la palette de color pour la légende

    def draw_legend(self, cv_image, result, palette):
        """
        Dessine une légende (couleur + nom de classe) en haut à droite de l'image.
        Utilise la même palette que semantic_mask pour que les couleurs correspondent.
        """
        names = result.names  # dict {id: nom}

        box_size = 18       # taille du carré de couleur
        padding = 6          # espace entre les entrées
        x_start = cv_image.shape[1] - 160   # position à droite, ajuster selon largeur d'image
        y_start = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        for i, (class_id, name) in enumerate(names.items()):
            y = y_start + i * (box_size + padding)

            # éviter de sortir de l'image si trop de classes
            if y + box_size > cv_image.shape[0]:
                break

            color = tuple(int(c) for c in palette[int(class_id)])

            # carré de couleur
            cv2.rectangle(cv_image, (x_start, y), (x_start + box_size, y + box_size), color, thickness=-1)
            cv2.rectangle(cv_image, (x_start, y), (x_start + box_size, y + box_size), (255, 255, 255), thickness=1)

            # texte du label
            cv2.putText(cv_image, name, (x_start + box_size + 6, y + box_size - 4),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return cv_image

    def classYolo(self, r, u_valid, v_valid):
        """
        r : result YOLO
        u_valid, v_valid : tableaux numpy de coordonnées de tout les points lidar projetés
        retourne : tableau numpy de label
        """
        carte = r.semantic_mask.data
        ids = carte[v_valid, u_valid]
        names = np.array([r.names[int(i)] for i in ids])
        return ids


# ----------------------- TRANSFORMATION/PROCESSIING ----------------------

    def apply_transform(self, pts_lidar_3d, odom_msg):
        q = odom_msg.pose.pose.orientation
        tr = odom_msg.pose.pose.position
        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        T_world_lidar = np.eye(4)
        T_world_lidar[:3, :3] = rot
        T_world_lidar[:3, 3] = [tr.x, tr.y, tr.z]

        ones = np.ones((len(pts_lidar_3d), 1))
        pts_h = np.hstack((pts_lidar_3d, ones))
        pts_monde = (T_world_lidar @ pts_h.T).T
        pts_monde = pts_monde[:, :3]          

        pts_monde = pts_monde @ self.R_init.T  

        return pts_monde

    def try_process_pending(self):
        # parcours toute les frames jusqu'à trouvé la paire de timestamp lidar odom compatible
        while self.pending_frames:
            frame = self.pending_frames[0]
            t_lidar = self._stamp_to_sec(frame['stamp'])
            odom_msg = self.interpolatation_odom(t_lidar)
            if odom_msg is None:
                break  # odometrie pas encore assez recente
 
            pts_monde = self.apply_transform(frame['points'], odom_msg)
            self.get_logger().info(f"[{self.robot}] Transformation de la frame/scan dans le repère monde")

            pts_monde[:, 0] += self.x_init_robot_norm
            pts_monde[:, 1] += self.y_init_robot_norm
 
            conf = 100.0 - frame['range'] * 100.0 / self.max_range
 
            self.publish_cloud(pts_monde, frame['label_ids'], conf, frame['stamp'])
            self.pending_frames.popleft()

    def recompute_all_points(self):    
        """Reconstruit entièrement new_data_dict en réappliquant, à chaque frame stockée
        depuis le début (points lidar bruts + timestamp), la pose interpolée depuis
        la trajectoire corrigée self.corrected_path."""
        if not self.corrected_path or not self.all_frames:
            return

        self.get_logger().info(f"[{self.robot}] Recalcul de {len(self.all_frames)} frames apres loop closure.")

        self.control_pub.publish(String(data=json.dumps({'cmd': 'reset', 'robot': self.robot})))
        time.sleep(0.1)

        for frame in self.all_frames:
            t_lidar = self._stamp_to_sec(frame['stamp'])
            pose = self.interpolate_pose_from_path(t_lidar, self.corrected_path)
            if pose is None:
                continue

            pts_monde = self.apply_transform(frame['points'], pose)
            pts_monde[:, 0] += self.x_init_robot_norm
            pts_monde[:, 1] += self.y_init_robot_norm
            conf = 100.0 - frame['range'] * 100.0 / self.max_range

            self.publish_cloud(pts_monde, frame['label_ids'], conf, frame['stamp'])

        self.get_logger().info(f"[{self.robot}] Recalcul termine.")


# ----------------------------- INTERPOLATIONS ------------------------------

    def interpolatation_odom(self, t_target):
        """
        Retourne un objet compatible avec apply_transform (pose.pose.position/.orientation),
        interpolé (lerp position + slerp rotation) entre les deux odoms encadrant t_target.
        None si l'historique ne couvre pas encore t_target.
        """
        if len(self.odom_history) == 0:
            return None

        t_last = self.odom_history[-1][0]
        if t_target > t_last:
            return None  # pas encore reçu 

        t_first = self.odom_history[0][0]
        if t_target <= t_first or len(self.odom_history) < 2: # scan lidar plus recent que le dernier odom 
            return self.odom_history[0][1] # return le plus proche 

        for i in range(len(self.odom_history) - 1): # on cherche les deux bornes , où target tombe entre deux odom
            t0, msg0 = self.odom_history[i]
            t1, msg1 = self.odom_history[i + 1]
            if t0 <= t_target <= t1:
                alpha = 0.0 if t1 == t0 else (t_target - t0) / (t1 - t0) # facteurs d'interpolations

                p0, p1 = msg0.pose.pose.position, msg1.pose.pose.position
                pos = SimpleNamespace( # interpolation position
                    x=p0.x + alpha * (p1.x - p0.x),
                    y=p0.y + alpha * (p1.y - p0.y),
                    z=p0.z + alpha * (p1.z - p0.z),
                )

                q0, q1 = msg0.pose.pose.orientation, msg1.pose.pose.orientation
                rots = R.from_quat([[q0.x, q0.y, q0.z, q0.w],
                                    [q1.x, q1.y, q1.z, q1.w]])
                slerp = Slerp([0, 1], rots) # interpolation rotation
                qx, qy, qz, qw = slerp([alpha])[0].as_quat()
                quat = SimpleNamespace(x=qx, y=qy, z=qz, w=qw)

                return SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(position=pos, orientation=quat)))

        return self.odom_history[-1][1]
  
    def interpolate_pose_from_path(self, t_target, poses):
        if not poses:
            return None

        times = [self._stamp_to_sec(p.header.stamp) for p in poses]
        t_target = min(max(t_target, times[0]), times[-1])

        i = 0
        while i < len(times) - 2 and times[i + 1] < t_target:
            i += 1

        t0, t1 = times[i], times[i + 1]
        alpha = 0.0 if t1 == t0 else (t_target - t0) / (t1 - t0)
        alpha = min(max(alpha, 0.0), 1.0)

        p0, p1 = poses[i].pose.position, poses[i + 1].pose.position
        pos = SimpleNamespace(
            x=p0.x + alpha * (p1.x - p0.x),
            y=p0.y + alpha * (p1.y - p0.y),
            z=p0.z + alpha * (p1.z - p0.z),
        )

        q0, q1 = poses[i].pose.orientation, poses[i + 1].pose.orientation
        rots = R.from_quat([[q0.x, q0.y, q0.z, q0.w],
                            [q1.x, q1.y, q1.z, q1.w]])
        qx, qy, qz, qw = Slerp([0, 1], rots)([alpha])[0].as_quat()
        quat = SimpleNamespace(x=qx, y=qy, z=qz, w=qw)

        return SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(position=pos, orientation=quat)))

    def publish_cloud(self, pts_monde, label_ids, conf, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = 'map'
 
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='label_id', offset=12, datatype=PointField.INT32, count=1),
            PointField(name='conf', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='stamp', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
 
        cloud_data = np.zeros(len(pts_monde), dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('label_id', np.int32), ('conf', np.float32), ('stamp', np.float32),
        ])
        cloud_data['x'] = pts_monde[:, 0]
        cloud_data['y'] = pts_monde[:, 1]
        cloud_data['z'] = pts_monde[:, 2]
        cloud_data['label_id'] = label_ids
        cloud_data['conf'] = conf
        cloud_data['stamp'] = self._stamp_to_sec(stamp)
 
        msg = pc2.create_cloud(header, fields, cloud_data.tolist())
        self.points_pub.publish(msg)
        self.get_logger().info("message publieeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
 

    # def publish_loop_cloud(self, pts_monde, label_ids, conf, stamp):
    #     header = Header()
    #     header.stamp = stamp
    #     header.frame_id = 'map'
 
    #     fields = [
    #         PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='label_id', offset=12, datatype=PointField.INT32, count=1),
    #         PointField(name='conf', offset=16, datatype=PointField.FLOAT32, count=1),
    #     ]
 
    #     cloud_data = np.zeros(len(pts_monde), dtype=[
    #         ('x', np.float32), ('y', np.float32), ('z', np.float32),
    #         ('label_id', np.int32), ('conf', np.float32),
    #     ])
    #     cloud_data['x'] = pts_monde[:, 0]
    #     cloud_data['y'] = pts_monde[:, 1]
    #     cloud_data['z'] = pts_monde[:, 2]
    #     cloud_data['label_id'] = label_ids
    #     cloud_data['conf'] = conf
 
    #     msg = pc2.create_cloud(header, fields, cloud_data.tolist())
    #     self.points_pub.publish(msg)
    #     self.get_logger().info("message publieeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
 

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
 
