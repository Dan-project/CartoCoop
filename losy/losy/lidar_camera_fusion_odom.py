# Dépendances
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2 
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R, Slerp
import message_filters
from tf2_ros import Buffer, TransformListener
from collections import deque
from types import SimpleNamespace
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # pour plot
from mpl_toolkits.mplot3d import Axes3D 
from shapely.geometry import Point, Polygon
# Types de messages
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2


class LidarImageFusionNode(Node):
# --------------------------------- INIT ----------------------------------

    def __init__(self):
        super().__init__('lidar_image_fusion_node')

        # Initialisation OpenCV
        self.bridge = CvBridge()

        # Initialisation du modèle YOLO 26 nano semantique
        self.get_logger().info("Model YOLOv8-Segmentation :")
        self.model = YOLO("yolo26n-sem.pt") 
        
        # Subsribers Passifs caméra gauche et velodyne (On remplace le timer par des "subscribers passifs " (on s'abonne aux topics sans callback direct pour pouvoir les synchroniser)
        self.image_sub = message_filters.Subscriber(self, Image, '/kitti/camera_color_left/image_raw') 
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, '/points_raw')
        # Subsriber odometry
        self.odom_sub = self.create_subscription(Odometry, '/liorf/mapping/odometry', self.odom_callback, 10)
    

        # Temps d'acquisitions syncrhonisé (le synchroniseur remplace le timer, il faut qu'ils arrivent avec un décalage max de 0.05 ici test à faire)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub], 
            queue_size=10, 
            slop=0.05
        )  
        # Appelle du callback principale fusion tout les ts 
        self.ts.registerCallback(self.fusion_callback)
        self.get_logger().info('Synchronisation temporelle')


        # Paramètres
        self.declare_parameter('acquisition_hz', 2.0)
        self.acquisition_hz = self.get_parameter('acquisition_hz').value
        self.min_interval = 1.0 / self.acquisition_hz   # ex: 0.5s si 2Hz
        self.last_processed_time = None     

        self.declare_parameter('range',20)
        self.max_range = self.get_parameter('range').value

        self.declare_parameter('dim', '2d')
        self.dim = self.get_parameter('dim').value



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

        self.W, self.H = 1242,375  # taille image KITTI 

        # Stockage sous forme de dictionnaire
        self.features = ['x','y','z','label','time'] # pour data dict
        self.data_dict = {feature: [] for feature in self.features} #  {'id': [], 'center_x': [], 'center_y': [], 'label': [], 'time': []}

        # Stockage des frames en attentes
        self.pending_frames = deque(maxlen=30)  # taille limite pour pas saturer

        # Stockage des odometry recu par Liorf/Liosam
        self.odom_history = deque(maxlen=200)  # (t_sec, msg)

        # Pour la méthode avec récupération de tf au lieu de s'abonner au topic odom
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)

# ------------------------------- CALLBACK -------------------------------

    def odom_callback(self, msg):
        t = self._stamp_to_sec(msg.header.stamp)
        self.odom_history.append((t, msg))
        self.get_logger().info(f"odom reçu t={t:.3f}, historique={len(self.odom_history)}")

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
        mask = points_lidar[:, 0] > 0
        points_front = points_lidar[mask]
        ones = np.ones((len(points_front), 1))
        points_front_1 = np.hstack((points_front, ones))

        # Projection matricielle --> points lidar dans le repère de la caméra pts_image
        pts_cam = self.Tr @ points_front_1.T
        pts_img = self.P2 @ pts_cam

        # Mask 2 : profondeur positive (normalement c'est un doublon car on le fait avec x>0 avant (enfaite ca dépend si la caméra et devant le Lidar)
        depth = pts_img[2, :]
        valid_depth = depth > 0

        u = pts_img[0, valid_depth] / depth[valid_depth]
        v = pts_img[1, valid_depth] / depth[valid_depth]
        d = depth[valid_depth]

        points_mask2 = points_front[valid_depth] # on applique le mask 2 aussi à points_front_1 pour pouvoir retrouver les points non transformé (pour pas devoir faire une retransformation inverse)

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
        })
        # Appelle de la fonction qui gèere la queue
        self.try_process_pending()
       
# ----------------------------- MAIN FUNCTION ----------------------------

    def try_process_pending(self):
        # parcours toute les frames jusqu'à trouvé la paire de timestamp lidar odom compatible
        while self.pending_frames:
            frame = self.pending_frames[0]
            t_lidar = self._stamp_to_sec(frame['stamp'])

            odom_msg = self.interpolatation_odom(t_lidar)

            if odom_msg is None:
                break  # odométrie pas encore assez récente


            pts_monde = self.apply_transform(frame['points'], odom_msg)
            self.get_logger().info("on a la transformation")
            for i in range(len(pts_monde)):
                x_m, y_m, z_m = pts_monde[i]
                self.data_dict['label'].append(frame['names'][i])
                self.data_dict['x'].append(x_m)
                self.data_dict['y'].append(y_m)
                self.data_dict['z'].append(z_m)
                self.data_dict['time'].append(frame['t_img'])

            self.pending_frames.popleft()

# ---------------------------- USEFUL FUNCTION ---------------------------

    def _stamp_to_sec(self, stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    def show_opencv(self,u_valid,v_valid,d_valid,cv_image,result):
        # Afficher les images avec les mask yolo sur open cv en temps réel
        cv_image_mask = self.semantic_mask(cv_image, result)

        # Afficher les points Lidar sur le open cv en temps réel
        d_norm = np.clip(d_valid / 50.0, 0.0, 1.0)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        colors = cv2.applyColorMap(d_uint8.reshape(-1, 1), cv2.COLORMAP_JET)
        colors = colors.reshape(-1, 3)
        for i in range(len(u_valid)):
            color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
            cv2.circle(cv_image_mask, (u_valid[i], v_valid[i]), radius=1, color=color, thickness=-1)

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
        return cv2.addWeighted(cv_image, 1 - alpha, mask_overlay, alpha, 0)
    
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

# ----------------------- TRANSFORMATION/INTERPOLATION --------------------

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
        return pts_monde[:, :3]

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

# --------------------------------- DESTROY -------------------------------

    def destroy_node(self):
        """
        Permets de réaliser des actions après avoir mis fin au programme, comme écrire dans un fichier
        Cependant, cela ne sera pas utilisable dnas un usage long terme sans arret comme en condition réel, donc à modiifer, par exemple avec un timer qui active tout les x secondes le programmes de parseurs (sans les plots etc)
        """
        # self.write_all()  # une seule écriture, quand on fait Ctrl+C
        # debug 
        if len(self.data_dict['z']) > 0:
            z_arr = np.array(self.data_dict['z'])
            self.get_logger().info(
                f"DEBUG z: min={z_arr.min():.3f} max={z_arr.max():.3f} mean={z_arr.mean():.3f} n={len(z_arr)}"
            )

        if self.dim=="3d" :
            self.parseur_3d()
        else:
            self.parseur()
        super().destroy_node()

# ------------------ OFFLINE : PARSEURS/PLOT/ÉCRITURE ----------------------

    def write_all(self):
        path  = '/home/multirobots/multirobots_ws/src/losy/losy/plot_via_txt/semantic.txt'
        with open(path, 'w') as f:
            f.write("Carte Sémantique")
            n = len(self.data_dict['x'])
            for i in range(n):
                f.write(f"\n{self.data_dict['label'][i]},{self.data_dict['x'][i]:.4f},{self.data_dict['y'][i]:.4f},{self.data_dict['z'][i]:.4f}")

    def parseur(self):
        """
        self.dict = {
            1 : [x,y,label,time],
            2 : [x,y,label,time], 
            # ... i
        }

        Permets de créer la  map  discrétisé sous forme de geojson en 2D 
        
        """


        df = pd.DataFrame(self.data_dict) # convertit le dict en df
        geometry = [Point(x,y) for x,y in zip(df['x'], df['y'])]
        gdf_dict = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:2154')
        
        xmin,xmax = -400,400
        ymin,ymax = -200,200
        step=2
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
                'building': 'red',
                'road': 'black',
                'vegetation': 'forestgreen',
                'grass': 'lightgreen',
                'water': 'blue',
                'ground': 'grey',
            }
            fig, ax = plt.subplots(figsize=(15, 8))
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
        df = pd.DataFrame(self.data_dict)  # x, y, z, label, time

        # init des variables
        xmin, xmax = -400, 400
        ymin, ymax = -200, 200
        zmin, zmax = -5, 5
        stepz = 0.5
        step = 0.5

        # calcul de l'indice de voxel pour chaque point
        df['ix'] = ((df['x'] - xmin) // step).astype(int)
        df['iy'] = ((df['y'] - ymin) // step).astype(int)
        df['iz'] = ((df['z'] - zmin) // stepz).astype(int)

        # systeme de layer 
        # liste des tranches en z bornes

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

        # dominant_points = grouped[grouped['label'] == label_dominant] # tout les points d'un voxel avec le label diminant
        # tmin_dominant = dominant_points['time'].min() # prends le time d'aqcusiiton de la camera du label dominant, le premier vue , min
        # tmax_dominant = dominant_points['time'].max() # le dernier 

        df['is_dominant'] = df['label'] == df.groupby(['ix','iy','iz'])['label'].transform(lambda x: x.mode()[0])# tout les points d'un voxel avec le label diminant
        tmin_dominant = df[df['is_dominant']].groupby(['ix','iy','iz'])['time'].min()# prends le time d'aqcusiiton de la camera du label dominant, le premier vue , min
        tmax_dominant = df[df['is_dominant']].groupby(['ix','iy','iz'])['time'].max() # le dernier 
        
          
        # on crée un dataframe de la grille 3D labelisé, fais le matching tout seul car ils sont créer avec groupby et ils ont tous un multiindex ix, iy iz qui repsente un voxel
        voxel_agg = pd.DataFrame({
            'label': label_dominant,
            'nb_total': nb_total,
            'pct_dominant': pct_dominant,
            'tmin' : tmin_dominant,
            'tmax' : tmax_dominant
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

                colors = {
                    'building': 'red',
                    'road': 'black',
                    'vegetation': 'forestgreen',
                    'grass': 'lightgreen',
                    'water': 'blue',
                    'ground': 'grey',
                }
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_box_aspect((x_range, y_range, z_range))  # aspect proportionnel aux données réelles
                ax.scatter(voxel_agg['x_center'], voxel_agg['y_center'], voxel_agg['z_center'],
                        c=[colors.get(l, 'grey') for l in voxel_agg['label']], s=0.2, alpha=0.6)
                plt.savefig(path1, dpi=150)
                plt.show()


        path2 = f'/home/multirobots/multirobots_ws/src/losy/losy/geojson_folder/discretized3D_semantic_map_stepXY_{step}_stepZ_{stepz}.geojson'
        geometry = [Point(x, y, z) for x, y, z in zip(voxel_agg['x_center'], voxel_agg['y_center'], voxel_agg['z_center'])]
        voxel_gdf = gpd.GeoDataFrame(voxel_agg.drop(columns=['x_center', 'y_center', 'z_center']), geometry=geometry, crs='EPSG:2154') # on drop les cx y z center dcar ils sont deja dans geometries sinon on les aura ausis dans les properties
        voxel_gdf.to_file(path2, driver='GeoJSON')



    """
        NOTES : 
            remettre le time syncrohnizer de ros

            quetsion : 
            Lors du mapping semantique, les obstacles dynamqiues sont aussi labelisé a un endroite meme si ils sont en mouvement, 
            si on les eneleves ok mais defois ils sont statiques 

            sur le modele YOLO actuel j'ai pas de confiance 

            jai rajouté le pourcentage de presence dun label dans un voxel

            Pour le temps d'qcsuitions de la camera on mets lequel ? ca ron va voir plsueirus fois la meem chose 

            On a encore des discontinuité a 2hz 


            enelevr els matrices inverses et méthode a modifier

            voir les a stack des données rgb pronfdeur etc 
            Ce framework fusionne en temps réel les données d'un LiDAR et d'une caméra pour construire une carte sémantique de l'environnement traversé par un robot.
              Chaque scan LiDAR est projeté dans l'image caméra, où un modèle YOLO segmente la scène par classes (route, végétation, bâtiment, etc.), 
              ce qui permet d'attribuer une étiquette sémantique à chaque point 3D. Ces points étiquetés sont ensuite replacés dans le repère du monde grâce à l'odométrie du robot 
              (fournie par LIORF, avec interpolation entre les mesures), puis accumulés au fil du temps. Si une fermeture de boucle (loop closure) est détectée,
                la trajectoire corrigée est utilisée pour recalculer rétroactivement la position de tous les points déjà collectés, évitant ainsi les dérives accumulées.
                  Enfin, à l'arrêt du programme, l'ensemble des points est discrétisé en grille (2D) ou en voxels (3D) pour produire une carte finale exportée en GeoJSON,
                    avec une estimation de traversabilité par zone selon le type de terrain détecté.

    """


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