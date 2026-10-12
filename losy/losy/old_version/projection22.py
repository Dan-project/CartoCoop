import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
from ultralytics import YOLO




# LIDAR CAMERA FUSION AVEC LE COMEBACK MATRICIELLE AVEC MATRICES INVERSES, MARCHE MAIS LOURD EN CALCUL POUR RIEN,
# DE PLUS IL N'Y A PAS ENCORE LE SSYTEME DE QUEUE POUR LES FRAMES POUR AVOIR LA FRAME ET LA TF AVEC LE BON TIME STAMP LA ON METS LE TEMPS ACTUEL DONC PAS LA BONNE TRRANSFORMATION TF DE LIO SAM 





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



        self.declare_parameter('acquisition_hz', 2.0)
        self.acquisition_hz = self.get_parameter('acquisition_hz').value
        self.min_interval = 1.0 / self.acquisition_hz   # ex: 0.5s si 2Hz
        self.last_processed_time = None     

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

        self.Tr_inverse = np.linalg.inv(self.Tr)
        self.P2_inverse = np.linalg.pinv(self.P2)

        self.W, self.H = 1242,375  # taille image KITTI , A VERIf 1242  375
        self.only_lidar = False
        self.features = ['x','y','z','label','time'] # pour data dict
        self.data_dict = {feature: [] for feature in self.features} #  {'id': [], 'center_x': [], 'center_y': [], 'label': [], 'time': []}
    


        # pour get trasnform function test

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


        self.max_range=20

    def fusion_callback(self, image_msg, lidar_msg):
        """ la fusion temporelle se fait naturellement grâce à l'initialisation passif des subscribers, puis on fait la fusion dans 'eapce on projettte 
            le lidar 3D vers le repère 2D de la caméra pour comparer les points du lidar avec les pixels de la caméra"""


        t_img = self._stamp_to_sec(image_msg.header.stamp)

        # contrôle de fréquence : on ignore cette paire si trop proche de la précédente traitée
        if self.last_processed_time is not None:
            if (t_img - self.last_processed_time) < self.min_interval:
                return   # on skip, la paire est valide mais trop rapprochée

        self.last_processed_time = t_img


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

        # mask 4 distance max 
        range_mask = d_valid < self.max_range
        u_valid = u_valid[range_mask]
        v_valid = v_valid[range_mask]
        d_valid = d_valid[range_mask]
        
        # Image conversion ROS to OpenCV
        try:
            if self.only_lidar:
                cv_image = np.zeros((self.H, self.W, 3), dtype=np.uint8)  # fond noir
            else :
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Erreur de conversion OpenCV : {e}")
            return



        # Yolo avant pour afficher en fond le mask semantique
    
        results = self.model(cv_image)
        result = results[0]
        cv_image_mask = self.overlay_semantic_mask(cv_image, result)
 


        # Plot les points lidar valide sur la frame image reçu par la caméra
        # Coloriage des points selon la profondeur (proche=rouge, loin=bleu), on normalise la profondeur entre 0 et 255 sur une plage [0, 50] mètres
        d_norm = np.clip(d_valid / 50.0, 0.0, 1.0)            # 0.0 (proche)  1.0 (loin)
        d_uint8 = (d_norm * 255).astype(np.uint8)             # passage en uint8 pour colormap
        colors = cv2.applyColorMap(d_uint8.reshape(-1, 1), cv2.COLORMAP_JET)  # (N,1,3)
        colors = colors.reshape(-1, 3)                         # (N, 3) BGR
        for i in range(len(u_valid)):
            color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
            cv2.circle(cv_image_mask, (u_valid[i], v_valid[i]), radius=1, color=color, thickness=-1)


        cv2.imshow("Masque sémantique + lidar projeté", cv_image_mask)
        cv2.waitKey(1)


        # come back
        pts_lidar_cb = self.comeback3D(u_valid,v_valid,d_valid) # repere du ldiar
        self.get_logger().info(f"len pt ldiar {len(pts_lidar_cb)}")
        pts_monde = self.get_transform_world(pts_lidar_cb,lidar_msg.header.stamp)
        
        if pts_monde is None: # sécurité
            self.get_logger().info("on perd une frame !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return
        else :
            self.get_logger().info(f"len pt lidars {len(pts_monde)}")
        
        names = self.classYolo(result, u_valid, v_valid)
        for i in range(len(u_valid)): # ecris dans un dictionnaire

            x_m, y_m, z_m = pts_monde[i]
            self.data_dict['label'].append(names[i])
            self.data_dict['x'].append(x_m)
            self.data_dict['y'].append(y_m)
            self.data_dict['z'].append(z_m)
            self.data_dict['time'].append(t_img)


        # comparaison avant apres
        # self.get_logger().info(
        #     f"AVANT comeback3D — points originaux (dans le champ de vision) : "
        #     f"x[{points_front[in_frame][:,0].min():.2f}, {points_front[in_frame][:,0].max():.2f}] "
        #     f"y[{points_front[in_frame][:,1].min():.2f}, {points_front[in_frame][:,1].max():.2f}] "
        #     f"z[{points_front[in_frame][:,2].min():.2f}, {points_front[in_frame][:,2].max():.2f}]"
        # )
        # self.get_logger().info(
        #     f"APRES comeback3D — points reconstruits : "
        #     f"x[{pts_lidar_cb[:,0].min():.2f}, {pts_lidar_cb[:,0].max():.2f}] "
        #     f"y[{pts_lidar_cb[:,1].min():.2f}, {pts_lidar_cb[:,1].max():.2f}] "
        #     f"z[{pts_lidar_cb[:,2].min():.2f}, {pts_lidar_cb[:,2].max():.2f}]"
        # )
        
    def overlay_semantic_mask(self, cv_image, result, alpha=0.5):
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
    
    def _stamp_to_sec(self, stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9
          
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
 
    def write_all(self):
        """
        modif comme ca on utilse que un dict
        """
        with open('semantique.txt', 'w') as f:
            f.write("Carte Sémantique")
            n = len(self.data_dict['x'])
            for i in range(n):
                f.write(f"\n{self.data_dict['label'][i]},{self.data_dict['x'][i]:.4f},{self.data_dict['y'][i]:.4f},{self.data_dict['z'][i]:.4f}")

    def destroy_node(self):
        self.write_all()  # une seule écriture, quand on fait Ctrl+C
        if self.dim=="3d" :
            self.parseur_3d()
        else:
            self.parseur()
        super().destroy_node()

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

    def get_transform_world(self, pts_lidar_3d: np.ndarray,stamp) -> np.ndarray | None:
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
                stamp,# rclpy.time.Time()
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
                      
    def parseur(self):
        """

         vers geojson,a ameliorer l'init de la carte l'origine du robot

        self.data_dict ={
            'x' : [],
            'y' : [],
            'label' : [],
            'time' : []
        }

        # or 
        self.dict = {
            1 : [x,y,label,time],
            2 : [x,y,label,time], 
            # ... i

        }
        
        """

        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point, Polygon

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
        import matplotlib.pyplot as plt
              
        # if show:
        #     colors ={
        #         'building':'red',
        #         'road':'black',
        #         'forest':'forestgreen',
        #         'grass':'lightgreen',
        #         'water':'blue',
        #         'ground':'grey',
        #     }

        #     fig, ax = plt.subplots(figsize=(15, 8))
        #     grid_filtered.plot(
        #         ax=ax,
        #         color=[colors.get(l, 'grey') for l in grid_filtered['label']],
        #         edgecolor='white',
        #         linewidth=0.1
        #     )
        #     plt.show()            


        # grid_filtered = gpd.sjoin(grid, gdf_dict, how='inner', predicate='intersects')

        # #  Un label dominant par case test
        # grid_agg = (
        #     grid_filtered.groupby(grid_filtered.index)
        #     .agg(label=('label', lambda x: x.mode()[0]))
        # )
        # grid_agg = grid.loc[grid_agg.index].copy()
        # grid_agg['label'] = grid_agg.index.map(grid_filtered.groupby(level=0)['label'].agg(lambda x: x.mode()[0]))

        # if show:
        #     colors ={
        #         'building':'red',
        #         'road':'black',
        #         'forest':'forestgreen',
        #         'grass':'lightgreen',
        #         'water':'blue',
        #         'ground':'grey',
        #     }
  
        #     fig, ax = plt.subplots(figsize=(15, 8))
        #     grid_filtered.plot(
        #         ax=ax,
        #         color=[colors.get(l, 'grey') for l in grid_filtered['label']],
        #         edgecolor='white',
        #         linewidth=0.1
        #     )
        #     plt.savefig('grid_map.png', dpi=150, bbox_inches='tight')
        #     plt.show()

        
        #     # Un seul label (le plus fréquent) par case
        #     grid_agg = grid_filtered.groupby(level=0).agg({'label': lambda x: x.mode()[0]})

        #     # On recolle la géométrie carré originale (jointure sur l'index, rapide)
        #     grid_agg = grid.join(grid_agg, how='inner')

        #     # Centre de chaque case
        #     grid_agg['geometry'] = grid_agg.geometry.centroid

        #     grid_agg.to_file('grid_centers.geojson', driver='GeoJSON')


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
            plt.savefig('grid_map.png', dpi=150, bbox_inches='tight')
            plt.show()
        
        # Export GeoJSON : centre de chaque case + label dominant + pourcentage
        grid_centers = grid_agg.copy()
        grid_centers['geometry'] = grid_centers.geometry.centroid
        grid_centers.to_file('grid_centers.geojson', driver='GeoJSON')

    def parseur_3d(self):

        """
        change de méthode car les fonctione telles que sjoin, le Polygon dnas geopadndas ne marche pas nativemetn en 3D, 
        """
        import numpy as np
        import pandas as pd
        import geopandas as gpd # pour utiliser la fonction de gpd to_file 
        import matplotlib.pyplot as plt # pour plot
        from mpl_toolkits.mplot3d import Axes3D 
        from shapely.geometry import Point
        # dict en dataframe pandas
        df = pd.DataFrame(self.data_dict)  # x, y, z, label, time

        # init des variables
        xmin, xmax = -400, 400
        ymin, ymax = -200, 200
        zmin, zmax = 0, 10
        stepz = 1
        step = 2

        # calcul de l'indice de voxel pour chaque point
        df['ix'] = ((df['x'] - xmin) // step).astype(int)
        df['iy'] = ((df['y'] - ymin) // step).astype(int)
        df['iz'] = ((df['z'] - zmin) // stepz).astype(int)

        # systeme de layer 
        # liste des tranches en z bornes

        mask_layer1 = (df['z'] >= 0) & (df['z'] < 1)
        mask_layer2 = (df['z'] >= 1) & (df['z'] < 3)
        mask_layer3 = (df['z'] >= 3) & (df['z'] < 20)

        in_any_layer = mask_layer1 | mask_layer2 | mask_layer3
        df = df[in_any_layer]

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
        if show:
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
                ax.scatter(voxel_agg['x_center'], voxel_agg['y_center'], voxel_agg['z_center'],
                        c=[colors.get(l, 'grey') for l in voxel_agg['label']], s=1)
                plt.savefig('grid_map_3d.png', dpi=150)
                plt.show()



        geometry = [Point(x, y, z) for x, y, z in zip(voxel_agg['x_center'], voxel_agg['y_center'], voxel_agg['z_center'])]
        voxel_gdf = gpd.GeoDataFrame(voxel_agg.drop(columns=['x_center', 'y_center', 'z_center']), geometry=geometry, crs='EPSG:2154') # on drop les cx y z center dcar ils sont deja dans geometries sinon on les aura ausis dans les properties
        voxel_gdf.to_file(f'voxels_m{xmax-xmin}x{ymax-ymin}x{zmax-zmin}_s{step}.geojson', driver='GeoJSON')

"""
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