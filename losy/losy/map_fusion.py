# Dépendances
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, qos_profile_sensor_data
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



class MapFusionNode(Node):

# --------------------------------- INIT ----------------------------------

    def __init__(self):
        super().__init__('map_fusion_node')
 

        # ----- Déclaration des paramètres -----

        self.declare_parameter('robots', ['scout', 'ranger'])
        self.robots = list(self.get_parameter('robots').value)

        # paramètres de création de map
        self.declare_parameter('dim', '2d') # dimension 2d ou 3d
        self.dim = self.get_parameter('dim').value
        self.declare_parameter('cell_size',1) # taille de la grille (x y)
        self.cell_size = self.get_parameter('cell_size').value
        self.declare_parameter('cell_size_z',1) # taille de la grille (z) --> voxel 
        self.cell_size_z = self.get_parameter('cell_size_z').value
        # bornes de la zone
        self.declare_parameter('xmin_map', 709433.5004293)
        self.xmin_map = self.get_parameter('xmin_map').value
        self.declare_parameter('ymin_map', 7056437.95734587)
        self.ymin_map = self.get_parameter('ymin_map').value
        self.declare_parameter('xmax_map', 709891.84367255)
        self.declare_parameter('ymax_map', 7057059.64260302)
        self.xmax_map = self.get_parameter('xmax_map').value
        self.ymax_map = self.get_parameter('ymax_map').value
 

        self.declare_parameter('write_interval_sec', 30.0) # temps avant importation
        self.write_interval = self.get_parameter('write_interval_sec').value
 
        self.declare_parameter('output_dir', '/home/multirobots/multirobots_ws/src/losy/losy/multi')
        self.output_dir = self.get_parameter('output_dir').value

        self.declare_parameter('semantic_txt_path',
                                '/home/multirobots/multirobots_ws/src/losy/losy/multi/semantic.txt')
        self.semantic_txt_path = self.get_parameter('semantic_txt_path').value

        self.declare_parameter('semantic_txt_loop',
                                '/home/multirobots/multirobots_ws/src/losy/losy/multi/loop.txt')
        self.semantic_txt_loop = self.get_parameter('semantic_txt_loop').value


        self.declare_parameter('prior_map_path',
                                '/home/multirobots/multirobots_ws/src/losy/losy/priormap/export_grid.geojson')
        prior_path = self.get_parameter('prior_map_path').value

        # paramètre de fermeture de boucle
        self.declare_parameter('loop', False) # activer ou désactiver la loop closure (seulement ici liosam/liorf est indépendant, si la loop closure de liosam est activé elle n'aura aucune influence sur les données de notre map. Cependant, si on veut activer la loop closure, il faut l'activer ici est dans liosam/liorf)
        # self.loop_closure = self.get_parameter('loop').value


        # ----- carte a priori -----
        self.prior_map_gdf = None
        if prior_path:
            try:
                prior_dict = self.geojson_to_dict(prior_path)
                self.prior_map_gdf = self.dict_to_gdf(prior_dict)
                self.get_logger().info(f"{len(prior_dict)} entites chargees depuis la carte a priori.")
            except Exception as e:
                self.get_logger().warn(f"Carte a priori non chargee: {e}")
 
        # ----- stockage partage, commun a tous les robots -----
        self.features = ['x', 'y', 'z', 'label', 'time', 'conf_r']
        self.data_dict = {robot: {f: [] for f in self.features} for robot in self.robots}
        self.label_maps = {}  # robot -> {label_id: nom}
        self.dict_trav = {"road": 100.0}
 
        # ----- abonnements dynamiques, un jeu de topics par robot -----
        latched_qos = QoSProfile(depth=1)
        latched_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        latched_qos.reliability = QoSReliabilityPolicy.RELIABLE
 
        self._subs = [] # abonnement a tout les robots dispo
        for robot in self.robots:
            points_topic = f'/{robot}/semantic_points'
            labels_topic = f'/{robot}/semantic_points_labels'
            control_topic = f'/{robot}/semantic_points_control' # signalement loop closure
 
            self._subs.append(self.create_subscription(
                PointCloud2, points_topic,
                lambda msg, r=robot: self.points_callback(msg, r), 10))
            self._subs.append(self.create_subscription(
                String, labels_topic,
                lambda msg, r=robot: self.labels_callback(msg, r), latched_qos))
            
            self._subs.append(self.create_subscription(
                String, control_topic,
                lambda msg, r=robot: self.control_callback(msg, r), 10))
 
            self.get_logger().info(f"Abonnement fusion: {points_topic} / {labels_topic}")
 
        # export periodique
        # pas seulement a l'arret du noeud comme je faisais avant (pb potentiel, prendre trop de temps et raté les infos des frames)
        self.create_timer(self.write_interval, self.periodic_export)
 

# ------------------------------- CALLBACK -------------------------------

    def labels_callback(self, msg, robot):
        self.label_maps[robot] = {int(k): v for k, v in json.loads(msg.data).items()}
        self.get_logger().info(f"Table de labels recue pour '{robot}' ({len(self.label_maps[robot])} classes).")
    
    def points_callback(self, msg, robot):
        label_map = self.label_maps.get(robot)
        if label_map is None:
            self.get_logger().warn(f"Points de '{robot}' recus avant sa table de labels, ignores.")
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        d = self.data_dict[robot]
        n_before = len(d['x'])
        for x, y, z, label_id, conf in pc2.read_points(
                msg, field_names=('x', 'y', 'z', 'label_id', 'conf'), skip_nans=True):
            d['x'].append(float(x))
            d['y'].append(float(y))
            d['z'].append(float(z))
            d['label'].append(label_map.get(int(label_id), 'unknown'))
            d['time'].append(t)
            d['conf_r'].append(float(conf))
        n_added = len(d['x']) - n_before
        self.get_logger().info(f"[{robot}] +{n_added} points (total={len(d['x'])})")
  
    def control_callback(self, msg, robot):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if cmd.get('cmd') == 'reset':
            n_before = len(self.data_dict[robot]['x'])
            self.data_dict[robot] = {f: [] for f in self.features}
            self.get_logger().info(f"[{robot}] Reset apres loop closure ({n_before} points effaces, en attente de republication).")

# ----------------------------- CARTE A-PRIORI -----------------------------

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


# ----------------------------- EXPORTATION ------------------------------

    def periodic_export(self):
        total_points = sum(len(self.data_dict[r]['x']) for r in self.robots)
        if total_points == 0:
            self.get_logger().info("Aucun point recu pour l'instant, export ignore.")
            return
        self.get_logger().info(f"Export de {total_points} points au total.")

        # self.write_all()
        # self.get_logger().info("Ecriture des fichiers")
        try:
            if self.dim == '3d':
                self.parseur_3d()
            else:
                self.parseur()
                self.get_logger().info("Parseur")
        except Exception as e:
            self.get_logger().error(f"Erreur lors de l'export de la carte: {e}")

# -----------------------  PARSEURS/PLOT/ÉCRITURE --------------------------
    def combined_df(self):
        dfs = [pd.DataFrame(self.data_dict[r]) for r in self.robots if len(self.data_dict[r]['x']) > 0]
        if not dfs:
            return pd.DataFrame({f: [] for f in self.features})
        return pd.concat(dfs, ignore_index=True)

    def parseur(self):
        """
        self.dict = {
            1 : [x,y,label,time],
            2 : [x,y,label,time], 
            # ... i
        }

        Permets de créer la  map  discrétisé sous forme de geojson en 2D 
        
        """

    
        df = self.combined_df() # convertit le dict en df et assemble les df de tout les robots

        geometry = [Point(x,y) for x,y in zip(df['x'], df['y'])]
        gdf_dict = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:2154')
        
        xmin = 0
        xmax = 709891.84367255 - self.xmin_map
        ymin= 0 
        ymax = 7057059.64260302 -self.ymin_map

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

        path1 = f'/home/multirobots/multirobots_ws/src/losy/losy/multi/discretized2D_semantic_map_step_{step}.png'


        # plot la map a priori en fond ne gris clair pour se rendre comtpe 


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
        
        # Export GeoJSON : centre de chaque case + label dominant + pourcentage
        grid_centers = grid_agg.copy()
        grid_centers['geometry'] = grid_centers.geometry.centroid
        path2= f'/home/multirobots/multirobots_ws/src/losy/losy/multi/discretized2D_semantic_map_step_{step}.geojson'
        grid_centers.to_file(path2, driver='GeoJSON')

    def parseur_3d(self):

        """
        Permets de créer la  map  discrétisé sous forme de geojson en 3D
        change de méthode car les fonctione telles que sjoin, le Polygon dans geopadndas ne marche pas nativemetn en 3D, 
        """

        df = self.combined_df() # convertit le dict en df et assemble les df de tout les robots

        # init des variables
        xmin = 0
        xmax = 709891.84367255 - self.xmin_map
        ymin= 0 
        ymax = 7057059.64260302 -self.ymin_map
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
        path1 = f'/home/multirobots/multirobots_ws/src/losy/losy/multi/discretized3D_semantic_map_stepXY_{step}_stepZ_{stepz}.png'

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
        with open(self.semantic_txt_path, 'w') as f:
            f.write("Carte Semantique")
            for robot in self.robots:
                d = self.data_dict[robot]
                n = len(d['x'])
                for i in range(n):
                    f.write(f"\n{d['label'][i]},{d['x'][i]:.4f},{d['y'][i]:.4f},{d['z'][i]:.4f}")

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
    node = MapFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Fermeture : export final de la carte...")
        node.periodic_export()
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
