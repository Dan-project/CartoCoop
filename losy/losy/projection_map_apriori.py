# Dépendances
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2 
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
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import time

class LidarImageFusionNode(Node):


# --------------------------------- INIT ----------------------------------

    def __init__(self):
        super().__init__('lidar_image_fusion_node')
        

        # ----- Déclaration des paramètres -----

        # choix du robot
        self.declare_parameter('robot','scout') # scout ou ranger (j'ai a duex acquisitions dispo, donc le choix du robot selectionne les bon paramètres dans la section "paramètres à initialiser".
        self.robot = self.get_parameter('robot').value
        # paramètres de création de map
        self.declare_parameter('dim', '2d') # dimension 2d ou 3d
        self.dim = self.get_parameter('dim').value
        self.declare_parameter('cell_size',1) # taille de la grille (x y)
        self.cell_size = self.get_parameter('cell_size').value
        self.declare_parameter('cell_size_z',1) # taille de la grille (z) --> voxel 
        self.cell_size_z = self.get_parameter('cell_size_z').value
        # position robot + param de normalisation de la map
        self.declare_parameter('xmin_robot_scout',709809.507192942)
        self.xmin_robot_scout = self.get_parameter('xmin_robot_scout').value
        self.declare_parameter('ymin_robot_scout',7056662.075542638)
        self.ymin_robot_scout = self.get_parameter('ymin_robot_scout').value
        self.declare_parameter('xmin_robot_ranger',709735.5419107936 )
        self.xmin_robot_ranger = self.get_parameter('xmin_robot_ranger').value
        self.declare_parameter('ymin_robot_ranger',7056805.517686295)
        self.ymin_robot_ranger = self.get_parameter('ymin_robot_ranger').value
        self.declare_parameter('xmin_map',709433.5004293)
        self.xmin_map = self.get_parameter('xmin_map').value
        self.declare_parameter('ymin_map',7056437.95734587)
        self.ymin_map = self.get_parameter('ymin_map').value
        self.declare_parameter('xmax_map', 709891.84367255)
        self.xmax_map = self.get_parameter('xmax_map').value
        self.declare_parameter('ymax_map', 7057059.64260302)
        self.ymax_map = self.get_parameter('ymax_map').value
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



        # ----- Paramètres à intialiser -----
        
        if self.robot == "scout" :
            topic_image = '/camera/camera/color/image_raw'
            topic_lidar = '/ouster/points'
            # coordonnées et oriantation du robot ()
            self.x_init_robot = self.xmin_robot_scout# valeurs données par le gps (environ 6 mètres de covariance)
            self.y_init_robot = self.ymin_robot_scout# valeurs données par le gps (environ 6 mètres de covariance)
            self.angle_init_robot = -0.09988048237230192 + np.pi/4  # valeurs données par l'imu mag (totalement faux, rajout pour être correcte et diffère selon les acquisitions donc ne pas si fier). Trouver une méthode correcte pour connaitre l'orientation du robot à l'initialisation
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
            topic_image = '/ugv1/camera/color/image_raw'
            topic_lidar = '/ugv1/ouster/points'
            # coordonnées et oriantation du robot ()
            self.x_init_robot = self.xmin_robot_ranger # valeurs données par le gps (environ 6 mètres de covariance)
            self.y_init_robot = self.ymin_robot_ranger # valeurs données par le gps (environ 6 mètres de covariance)
            self.angle_init_robot = -0.447613997675286 -np.pi/2 # valeurs données par l'imu mag (totalement faux, rajout de pi/2 pour être correcte et diffère selon les acquisitions donc ne pas si fier). Trouver une méthode correcte pour connaitre l'orientation du robot à l'initialisation
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



        # ----- Map a-priori -----

        # initialisation des paramètres
        self.prior_map_dict = {} 
        self.x_init_robot_norm = self.x_init_robot - self.xmin_map
        self.y_init_robot_norm = self.y_init_robot - self.ymin_map
        # Matrice de rotation initiale à appliquer au repère odom
        c, s = np.cos(self.angle_init_robot), np.sin(self.angle_init_robot)
        self.R_init = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        # Chargement de la carte a priori 
        default_path = '/home/multirobots/multirobots_ws/src/losy/losy/priormap/export_grid.geojson'
        if default_path:
            self.get_logger().info(f"Chargement de la carte a priori : {default_path}")
            self.prior_map_dict = self.geojson_to_dict(default_path)
            self.get_logger().info(f"{len(self.prior_map_dict)} entités chargées depuis le GeoJSON.")
        else:
            self.get_logger().warn(f"Aucun {default_path} fourni, carte a priori vide.")
        # convertir la carte en geo data frame
        self.prior_map_gdf = self.dict_to_gdf(self.prior_map_dict)



        # ----- Initialisation des modèles -----

        # Initialisation de OpenCV
        self.bridge = CvBridge()
        # Initialisation du modèle YOLO 26 nano semantique
        self.get_logger().info("Model YOLOv8-Segmentation :")
        self.model = YOLO("yolo26n-sem.pt") 
        


        # ----- Subscribers -----  
         
        # subscribers passifs
        self.image_sub = message_filters.Subscriber(self, Image, topic_image) 
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, topic_lidar)
        # subsriber odometry
        self.odom_sub = self.create_subscription(Odometry, '/liorf/mapping/odometry', self.odom_callback, 10)
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



        # ----- Initialisation des dictionnaires de stockage -----

        self.features = ['x','y','z','label','time','conf_r'] # pour data dict slope
        self.data_dict = {feature: [] for feature in self.features} #  {'id': [], 'center_x': [], 'center_y': [], 'label': [], 'time': []}
        # dictionnaires pour la carte de traversabilité à continuer
        self.dict_trav = {
            "road" : 100.0 # a ameliorer ici
        }



        # ----- Loop Closure -----

        # rajout de la partie detection de loop closure (à partir de liorf/liosam pas des données directement)
        if self.loop_closure :
            self.loop_closure_detected = False
            self.loop_sub = self.create_subscription(MarkerArray, 'liorf/mapping/loop_closure_constraints', self.loop_callback, 10)
            self.marker_history = 0 

            self.path_sub = self.create_subscription(Path, 'liorf/mapping/path', self.path_callback, 10) # contient toutes les positions, on réxupère seulement quand il y a un loop closure 

            self.pending_frames_until_loop = []

            self.loop_closure_display_until = 0.0 # pour afficher sur open cv

            self.corrected_path = []

            self.new_data_dict = {feature: [] for feature in self.features} #  {'id': [], 'center_x': [], 'center_y': [], 'label': [], 'time': []}

# ------------------------------- PRE-ONLINE -------------------------------

    def geojson_to_dict(self, geojson_path):
        """
        Lit le GeoJSON (FeatureCollection de Polygons) et le retransforme
        en dictionnaire { id : [x, y, label, time] }
        """
        # Charger le GeoJSON
        gdf = gpd.read_file(geojson_path)
        
        prior_map_dict = {}
        
        # Parcourir chaque entité
        for idx, row in gdf.iterrows():

            # Récupération de x et y depuis les propriétés center_x/center_y 
            x = row['center_x'] if 'center_x' in row and not pd.isna(row['center_x']) else row.geometry.centroid.x
            y = row['center_y'] if 'center_y' in row and not pd.isna(row['center_y']) else row.geometry.centroid.y
            
            label = row['label']
            time_val = row['time'] if 'time' in row and not pd.isna(row['time']) else 0.0
            
            # Construction du dictionnaire au format exact
            prior_map_dict[idx] = [x, y, label, time_val]
            
        return prior_map_dict
   
    def dict_to_gdf(self, prior_map_dict, crs='EPSG:2154'):
        df = pd.DataFrame.from_dict(
            prior_map_dict, orient='index', columns=['x', 'y', 'label', 'time']
        )
        geometry = [
            box(x - self.cell_size/2, y - self.cell_size/2, x + self.cell_size/2, y + self.cell_size/2)
            for x, y in zip(df['x'], df['y'])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
        return gdf

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
            self.get_logger().error(f"Erreur de conversion OpenCV : {e}")
            return
        
        # self.get_logger().info(
        #     f"points_lidar={len(points_lidar)}, points_front(x>0)={len(points_front)}, "
        #     f"valid_depth={valid_depth.sum()}, in_frame={in_frame.sum()}, "
        #     f"u_valid final={len(u_valid)}"
        # )
        # YOLO 
        results = self.model(cv_image)
        result = results[0]
        names = self.classYolo(result, u_valid, v_valid)

        # Appelle de la fonction pour afficher sur opencv les pts lidar et les mask yolo
        self.show_opencv(u_valid,v_valid,d_valid,cv_image,result)

       
        # Création d'une queue de frames pour 
        self.pending_frames.append({
            'points': points_mask4,
            'names': names,  
            't_img': t_img,
            'stamp': lidar_msg.header.stamp,
            'range' : d_valid,
        })

        if self.loop_closure :
            self.pending_frames_until_loop.append({
                'points': points_mask4,
                'names': names,  
                't_img': t_img,
                'stamp': lidar_msg.header.stamp,
                'range' : d_valid,

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

# --------------------------------- YOLO ----------------------------------
       
    def classYolo(self, r, u_valid, v_valid):
        """
        r : result YOLO
        u_valid, v_valid : tableaux numpy de coordonnées de tout les points lidar projetés
        retourne : tableau numpy de label
        """
        carte = r.semantic_mask.data
        ids = carte[v_valid, u_valid]
        names = np.array([r.names[int(i)] for i in ids])
        return names

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
                break  # odométrie pas encore assez récente


            pts_monde = self.apply_transform(frame['points'], odom_msg)
            self.get_logger().info("Transformation de la frame/scan dans le repère monde")

            # slopes = self.compute_slope(pts_monde,frame['names'])              # angle en degrés, un par point

            for i in range(len(pts_monde)):
                x_m, y_m, z_m = pts_monde[i]
                self.data_dict['label'].append(frame['names'][i])
                # self.data_dict['x'].append(x_m)
                # self.data_dict['y'].append(y_m) faire modif des trois pour le robot monde amap a priori
                # self.data_dict['z'].append(z_m)

                self.data_dict['x'].append(x_m + self.x_init_robot_norm) # + coordonnes du trobot normalisé par rap a la carte a priori
                self.data_dict['y'].append(y_m + self.y_init_robot_norm) 
                self.data_dict['z'].append(z_m)

                self.data_dict['time'].append(frame['t_img'])
                taux =  100 - frame['range'][i] *100 / self.max_range
                self.data_dict['conf_r'].append(taux) # on mets ici pour que chaque pts est sa conf pour garder le max dans le voxel plus tard
                # self.get_logger().info(f"{taux}")

                # self.data_dict['slope'].append(slopes[i])
                    
            self.pending_frames.popleft()

    def recompute_all_points(self):
        """Reconstruit entièrement new_data_dict en réappliquant, à chaque frame stockée
        depuis le début (points lidar bruts + timestamp), la pose interpolée depuis
        la trajectoire corrigée self.corrected_path."""
        if not self.corrected_path or not self.pending_frames_until_loop:
            return

        self.get_logger().info(f"Recalcul de {len(self.pending_frames_until_loop)} frames/scans après loop closure.")
        self.new_data_dict = {feature: [] for feature in self.features}

        for frame in self.pending_frames_until_loop:
            t_lidar = self._stamp_to_sec(frame['stamp'])
            pose = self.interpolate_pose_from_path(t_lidar, self.corrected_path)
            if pose is None:
                continue

            pts_monde = self.apply_transform(frame['points'], pose)

            # slopes = self.compute_slope(pts_monde,frame['names'])              # angle en degrés, un par point

            for i in range(len(pts_monde)):
                x_m, y_m, z_m = pts_monde[i]
                self.new_data_dict['label'].append(frame['names'][i])
                self.new_data_dict['x'].append(x_m + self.x_init_robot_norm)
                self.new_data_dict['y'].append(y_m + self.y_init_robot_norm)
                self.new_data_dict['z'].append(z_m)
                self.new_data_dict['time'].append(frame['t_img'])
                taux =  100 - frame['range'][i] *100 / self.max_range
                self.new_data_dict['conf_r'].append(taux)
                # self.get_logger().info(f"{taux}")

                # self.data_dict['slope'].append(slopes[i]) la pente

        self.get_logger().info("Recalcul terminé.")

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

# ------------------ OFFLINE : PARSEURS/PLOT/ÉCRITURE ----------------------

    def destroy_node(self):
        """
        Permets de réaliser des actions après avoir mis fin au programme, comme écrire dans un fichier
        Cependant, cela ne sera pas utilisable dnas un usage long terme sans arret comme en condition réel, donc à modiifer, par exemple avec un timer qui active tout les x secondes le programmes de parseurs (sans les plots etc)
        """
        # self.write_all()  # une seule écriture, quand on fait Ctrl+C
        if self.loop_closure:
            # self.write_all()  # une seule écriture, quand on fait Ctrl+C
            self.plot_path()
        # debug 
        if self.dim=='3d':
            self.parseur_3d()
        elif self.dim=='2d':
            self.parseur()

        super().destroy_node()

    def parseur(self):
        """
        self.dict = {
            1 : [x,y,label,time],
            2 : [x,y,label,time], 
            # ... i
        }

        Permets de créer la  map  discrétisé sous forme de geojson en 2D 
        
        """

        if self.loop_closure :

            df = pd.DataFrame(self.new_data_dict) # convertit le dict en df
        else :
            df = pd.DataFrame(self.data_dict) # convertit le dict en df

        geometry = [Point(x,y) for x,y in zip(df['x'], df['y'])]
        gdf_dict = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:2154')
        
        xmin = 0
        xmax = self.xmax_map- self.xmin_map
        ymin= 0 
        ymax = self.ymax_map -self.ymin_map

        step=self.cell_size
        cols=np.arange(xmin,xmax,step)
        rows=np.arange(ymin,ymax,step)
    
        # filling grid
        squares =[]
        for x in cols:
            for y in rows:
                squares.append(Polygon([(x,y),(x+step,y),(x+step,y+step),(x,y+step)]))

        grid = gpd.GeoDataFrame({'geometry':squares}, crs=gdf_dict.crs)

        # grid_filtered = gpd.sjoin(grid,gdf_dict, how='inner', predicate='intersects') # left garde aussi les cases vides , inner non 
        # grid_filtered.to_file('grid_map.geojson', driver='GeoJSON')
        show=True

        grid_filtered = gpd.sjoin(grid, gdf_dict, how='inner', predicate='intersects')

        # Label dominant + géométrie par case
        grid_agg = grid_filtered.groupby(level=0).agg({
            'label': lambda x: x.mode()[0],
            'geometry': 'first'
        })
        grid_agg = gpd.GeoDataFrame(grid_agg, geometry='geometry', crs=grid.crs)

        # Nombre de points du label dominant, et nombre total de points dans la case
        counts = grid_filtered.groupby(level=0)['label'].value_counts()
        nb_dominant = counts.groupby(level=0).max()       # le compte max = celui du label dominant
        nb_total = grid_filtered.groupby(level=0).size()  # nb total de points dans la case

        grid_agg['pct_dominant'] = (nb_dominant / nb_total * 100).round(1)

        path1 = f'/home/multirobots/multirobots_ws/src/losy/losy/geojson_folder/discretized2D_semantic_map_step_{step}.png'


        if show:
            colors = {
                'road' : 'black',
                'sidewalk' : 'grey',
                'terrain': 'grey',
                'building': 'red',
                'wall' : 'orange',
                'fence' : 'orange',
                'pole' : 'orange',
                'traffic_light':'orange',
                'traffic_sign':'orange',
                'person' : 'blue',
                'rider': 'blue',
                'car' : 'purple',
                'motorcycle': 'purple',
                'bus': 'purple',
                'truck': 'purple',
                'train': 'purple',
                'vegetation': 'forestgreen',
                'sky':'lightblue'
            }
            fig, ax = plt.subplots(figsize=(15, 8))
             # plot la map a priori en fond ne gris clair pour se rendre comtpe 

            self.prior_map_gdf.plot(
            ax=ax,
            color='whitesmoke',
            edgecolor='gray',
            linewidth=0.4,
            zorder=1
        )
            grid_agg.plot(
                ax=ax,
                color=[colors.get(l, 'grey') for l in grid_agg['label']],
                edgecolor='white',
                linewidth=0.1
            )
            plt.savefig(path1, dpi=150, bbox_inches='tight')
            plt.show()
        
        # Export GeoJSON : centre de chaque case + label dominant + pourcentage
        grid_centers = grid_agg.copy()
        grid_centers['geometry'] = grid_centers.geometry.centroid
        path2= f'/home/multirobots/multirobots_ws/src/losy/losy/geojson_folder/discretized2D_semantic_map_step_{step}.geojson'
        grid_centers.to_file(path2, driver='GeoJSON')

    def parseur_3d(self):

        """
        Permets de créer la  map  discrétisé sous forme de geojson en 3D
        change de méthode car les fonctione telles que sjoin, le Polygon dans geopadndas ne marche pas nativemetn en 3D, 
        """

        # dict en dataframe pandas
        if self.loop_closure :

            df = pd.DataFrame(self.new_data_dict) # convertit le dict en df
        else :
            df = pd.DataFrame(self.data_dict) # convertit le dict en df

        # init des variables
        xmin = 0
        xmax = self.xmax_map  - self.xmin_map
        ymin= 0 
        ymax = self.ymax_map  - self.ymin_map
        zmin, zmax = -5, 20
        stepz = self.cell_size_z
        step = self.cell_size

        # calcul de l'indice de voxel pour chaque point
        df['ix'] = ((df['x'] - xmin) // step).astype(int)
        df['iy'] = ((df['y'] - ymin) // step).astype(int)
        df['iz'] = ((df['z'] - zmin) // stepz).astype(int)

        # systeme de layer 
        # liste des tranches en z bornes
        self.get_logger().info(f"z min={df['z'].min():.2f}, max={df['z'].max():.2f}, mean={df['z'].mean():.2f}")
        mask_layer1 = (df['z'] >= zmin) & (df['z'] < zmax)
        # mask_layer2 = (df['z'] >= 1) & (df['z'] < 3)
        # mask_layer3 = (df['z'] >= 3) & (df['z'] < 20)

        # in_any_layer = mask_layer1 | mask_layer2 | mask_layer3
        df = df[mask_layer1]

        #  on filtre les points hors grille, important au final? on peut utiliser ca pour entre les layers?  
        nx = int((xmax - xmin) // step)
        ny = int((ymax - ymin) // step)
        # nz = int((zmax - zmin) // stepz) sert a rien deja un mask sur z ducoup
        mask = (df['ix'].between(0, nx-1) & df['iy'].between(0, ny-1))
        df = df[mask]

        # agrégation par voxel
        grouped = df.groupby(['ix', 'iy', 'iz']) # on regroupe tout les poitns qui sont dans le meme voxels
        label_dominant = grouped['label'].agg(lambda x: x.mode()[0]) # pour chaque groupe de voxel on regarde le lables dominant
        nb_total = grouped.size() # nb total de pts/labels dans le groupe 
        nb_dominant = grouped['label'].agg(lambda x: x.value_counts().max()) # nb de label dominant dans le groupe
        pct_dominant = (nb_dominant / nb_total * 100).round(1) # pourcentage de label dominant dans le groupe/ dans le voxel


        df['is_dominant'] = df['label'] == df.groupby(['ix','iy','iz'])['label'].transform(lambda x: x.mode()[0])# tout les points d'un voxel avec le label diminant

        # On groupe les points dominants une seule fois
        dominant_grouped = df[df['is_dominant']].groupby(['ix', 'iy', 'iz'])

        # Extraction des métriques du label dominant (Séries Pandas)
        tmin_dominant = dominant_grouped['time'].min()
        tmax_dominant = dominant_grouped['time'].max()
        conf_max_dominant = dominant_grouped['conf_r'].max().round(1)

        # .fillna(0.0) attribue 0.0 si le label n'est pas dans self.tab_trav
        trav_dominant = label_dominant.map(self.dict_trav).fillna(0.0)

        # slope_mean_dominant = dominant_grouped['slope'].mean().round(1)

        # on crée un dataframe de la grille 3D labelisé, fais le matching tout seul car ils sont créer avec groupby et ils ont tous un multiindex ix, iy iz qui repsente un voxel
        voxel_agg = pd.DataFrame({
            'label': label_dominant,
            'nb_total': nb_total,
            'pct_dominant': pct_dominant,
            'tmin' : tmin_dominant,
            'tmax' : tmax_dominant,
            'conf_range' : conf_max_dominant,
            'trav' : trav_dominant,
            # 'slope' : slope_mean_dominant,
        }).reset_index()

        # coordonnées du centre de chaque voxel (utile pour export/visualisation)
        voxel_agg['x_center'] = xmin + (voxel_agg['ix'] + 0.5) * step # coord , incide +0.5 pour se mettre au milleu fois step pour la distance réelle 
        voxel_agg['y_center'] = ymin + (voxel_agg['iy'] + 0.5) * step
        voxel_agg['z_center'] = zmin + (voxel_agg['iz'] + 0.5) * stepz

        # plot

        
        show=True
        path1 = f'/home/multirobots/multirobots_ws/src/losy/losy/geojson_folder/discretized3D_semantic_map_stepXY_{step}_stepZ_{stepz}.png'

        if show:
                
            # Calcul des étendues réelles
            x_range = voxel_agg['x_center'].max() - voxel_agg['x_center'].min()
            y_range = voxel_agg['y_center'].max() - voxel_agg['y_center'].min()
            z_range = voxel_agg['z_center'].max() - voxel_agg['z_center'].min()


            # colors = {
            #     'building': 'red',
            #     'road': 'black',
            #     'vegetation': 'forestgreen',
            #     'grass': 'lightgreen',
            #     'water': 'blue',
            #     'ground': 'grey',
            # }
            # fig = plt.figure(figsize=(12, 8))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_box_aspect((x_range, y_range, z_range))  # aspect proportionnel aux données réelles
            # ax.scatter(voxel_agg['x_center'], voxel_agg['y_center'], voxel_agg['z_center'],
            #         c=[colors.get(l, 'grey') for l in voxel_agg['label']], s=0.2, alpha=0.6)
            # plt.savefig(path1, dpi=150)
            # plt.show()

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect((x_range, y_range, z_range))

            # Définition des catégories uniques
            categories = pd.Categorical(voxel_agg['label'])

            scatter = ax.scatter(
                voxel_agg['x_center'], 
                voxel_agg['y_center'], 
                voxel_agg['z_center'],
                c=categories.codes,
                cmap='tab20',
                s=0.2, 
                alpha=0.6
            )

            # Génération automatique de la légende
            handles, _ = scatter.legend_elements()
            ax.legend(
                handles, 
                categories.categories, 
                title="Classes", 
                loc="upper left", 
                bbox_to_anchor=(1.05, 1)  # Place la légende à l'extérieur du graphique
            )

            plt.savefig(path1, dpi=150, bbox_inches='tight')
            plt.show()


        path2 = f'/home/multirobots/multirobots_ws/src/losy/losy/geojson_folder/discretized3D_semantic_map_stepXY_{step}_stepZ_{stepz}.geojson'
        geometry = [Point(x, y, z) for x, y, z in zip(voxel_agg['x_center'], voxel_agg['y_center'], voxel_agg['z_center'])]

        voxel_gdf = gpd.GeoDataFrame(voxel_agg.drop(columns=['x_center', 'y_center', 'z_center']), geometry=geometry, crs='EPSG:2154') # on drop les cx y z center dcar ils sont deja dans geometries sinon on les aura ausis dans les properties

        voxel_gdf.to_file(path2, driver='GeoJSON')














        # labels_sol = list(self.dict_trav.keys())  # ex: ['road', 'grass', 'sand', ...]

        # voxel_agg_sol = voxel_agg[voxel_agg['label'].isin(labels_sol)]

        # if voxel_agg_sol.empty:
        #     self.get_logger().warn("Aucun voxel de sol trouvé, rien à plot.")
        #     return






        # Plot traversabilité : projection 2D , en fonciton des labels juste et prends toutes les cases en compte applatit la 3D probleme
        # pour chaque colonne (ix, iy), on garde le trav minimal (le plus restrictif)
        # trav_2d = voxel_agg.groupby(['ix', 'iy']).agg({ # pour chaque ix iy , on garde la plus petite traversabilité et les premier x et y, car c 'est tjrs les memes pour un ix iy 
        #     'trav': 'min',
        #     'x_center': 'first',
        #     'y_center': 'first'
        # }).reset_index()

        # fig, ax = plt.subplots(figsize=(15, 8))
        # sc = ax.scatter(
        #     trav_2d['x_center'], trav_2d['y_center'],
        #     c=trav_2d['trav'], cmap='Greens',
        #     vmin=0, vmax=100, s=(step*100), marker='s'  # marker carré pour évoquer une grille
        # )

        # plt.colorbar(sc, label='Traversabilité (%)')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_title(f'Traversabilité (step={step}m)')
        # ax.axis('equal')

        # path3 = f'/home/multirobots/multirobots_ws/src/losy/losy/geojson_folder/traversability_step_{step}.png'
        # plt.savefig(path3, dpi=150, bbox_inches='tight')






        # --- Plot de la pente : projection 2D top-down, uniquement sur les voxels de sol ---
        # show_slope = True
        # if show_slope:
        #     # on garde le max de pente par colonne (ix,iy), le plus restrictif/prudent
        #     slope_2d = voxel_agg.groupby(['ix', 'iy']).agg({
        #         'slope': 'max',
        #         'x_center': 'first',
        #         'y_center': 'first'
        #     }).reset_index()

        #     fig, ax = plt.subplots(figsize=(15, 8))
        #     sc = ax.scatter(
        #         slope_2d['x_center'], slope_2d['y_center'],
        #         c=slope_2d['slope'], cmap='inferno_r',   # sombre = raide, clair = plat (inversé volontairement)
        #         vmin=-10, vmax=10,                          # borne max ajustable selon ton terrain
        #         s=(step * 100), marker='s'
        #     )
        #     plt.colorbar(sc, label='Pente (degrés)')
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('y')
        #     ax.set_title(f'Carte de pente (step={step}m)')
        #     ax.axis('equal')

        #     path4 = f'/home/multirobots/multirobots_ws/src/losy/losy/geojson_folder/slope_step_{step}.png'
        #     plt.savefig(path4, dpi=150, bbox_inches='tight')
        #     plt.show()

    def write_all(self):
        """
        modif comme ca on utilse que un dict
        """
        with open('/home/multirobots/multirobots_ws/src/losy/losy/plot_via_txt/semantic.txt', 'w') as f:
            f.write("Carte Sémantique")
            n = len(self.data_dict['x'])
            for i in range(n):
                f.write(f"\n{self.data_dict['label'][i]},{self.data_dict['x'][i]:.4f},{self.data_dict['y'][i]:.4f},{self.data_dict['z'][i]:.4f}")

        if self.loop_closure :

            with open('/home/multirobots/multirobots_ws/src/losy/losy/plot_via_txt/semantic_loop.txt', 'w') as f:
                f.write("Carte Sémantique")
                n = len(self.new_data_dict['x'])
                for i in range(n):
                    f.write(f"\n{self.new_data_dict['label'][i]},{self.new_data_dict['x'][i]:.4f},{self.new_data_dict['y'][i]:.4f},{self.new_data_dict['z'][i]:.4f}")

    def plot_path(self):
        """
        Plot 2D (x,y) comparant la trajectoire brute (odom_history, non corrigée)
        et la trajectoire corrigée (corrected_path, après loop closure).
        """
        if not self.corrected_path:
            self.get_logger().warn("corrected_path vide, rien à plot.")
            return
        if not self.odom_history:
            self.get_logger().warn("odom_history vide, rien à plot.")
            return

        # --- Trajectoire corrigée ---
        xs_c = [p.pose.position.x for p in self.corrected_path]
        ys_c = [p.pose.position.y for p in self.corrected_path]
        times_c = [self._stamp_to_sec(p.header.stamp) for p in self.corrected_path]

        # --- Trajectoire brute (non corrigée) ---
        xs_raw = [msg.pose.pose.position.x for (t, msg) in self.odom_history]
        ys_raw = [msg.pose.pose.position.y for (t, msg) in self.odom_history]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Trajectoire brute en gris pointillé
        ax.plot(xs_raw, ys_raw, '--', color='red', linewidth=1.0, alpha=0.6, label='Trajectoire brute (odom)')
        ax.scatter(xs_raw, ys_raw, color='red', s=8, alpha=0.5)

        # Trajectoire corrigée, colorée par timestamp comme avant
        sc = ax.scatter(xs_c, ys_c, c=times_c, cmap='viridis', s=30, label='Trajectoire corrigée (path)')
        ax.plot(xs_c, ys_c, '-', color='gray', linewidth=0.5, alpha=0.5)

        for i in range(0, len(xs_c), max(1, len(xs_c)//30)):
            ax.annotate(str(i), (xs_c[i], ys_c[i]), fontsize=7)

        plt.colorbar(sc, label='timestamp (s) — trajectoire corrigée')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Comparaison trajectoire brute vs corrigée ({len(self.corrected_path)} poses corrigées)')
        ax.axis('equal')
        plt.savefig('/home/multirobots/multirobots_ws/src/losy/losy/path/corrected_path.png', dpi=150)
        plt.close(fig)
        self.get_logger().info("Plot comparaison trajectoire brute/corrigée sauvegardé.") 
   



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