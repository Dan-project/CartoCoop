# LOSY - Guide développeur

Ce document décrit les fonctions de `projection_map_apriori.py`. Les autres noeuds reprennent les mêmes fonctions : `projection_kitti.py` sans la carte a priori, `perception.py` et `map_fusion.py` avec quelques fonctions en plus pour faire communiquer les deux noeuds.

## Repères et transformations

```
point LiDAR (repère os_lidar)
  -> Tr  : repère optique caméra          (projection, pour lire le label YOLO)
  -> P2  : plan image (u, v, profondeur)

point LiDAR (repère os_lidar)
  -> pose LIORF interpolée : repère odom/map de LIORF
  -> R_init                : rotation vers l'orientation de la carte a priori
  -> + (x_init_robot_norm, y_init_robot_norm) : repère local de la carte a priori
```

Le repère local de la carte a priori a pour origine (`xmin_map`, `ymin_map`) en Lambert 93. Toutes les coordonnées manipulées par le noeud sont exprimées dans ce repère pour éviter les très grandes valeurs.

La projection sert uniquement à lire le label : les points stockés sont les coordonnées LiDAR d'origine, filtrées par les mêmes masques que les points projetés. Il n'y a donc pas de transformation inverse image -> LiDAR.

## Synchronisation temporelle LiDAR / caméra

La caméra et le LiDAR ne publient ni au même instant ni à la même fréquence. Pour associer une image à un scan, le noeud n'utilise pas de timer mais des subscribers passifs :

```python
self.image_sub = message_filters.Subscriber(self, Image, topic_image)
self.lidar_sub = message_filters.Subscriber(self, PointCloud2, topic_lidar)

self.ts = message_filters.ApproximateTimeSynchronizer(
    [self.image_sub, self.lidar_sub], queue_size=10, slop=0.05)
self.ts.registerCallback(self.fusion_callback)
```

Un subscriber `message_filters` n'a pas de callback propre : il se contente de remplir une file. L'`ApproximateTimeSynchronizer` parcourt les deux files (10 messages chacune) et cherche une image et un scan dont les horodatages sont écartés de moins de `slop` (50 ms). Dès qu'une paire est trouvée, il appelle `fusion_callback(image_msg, lidar_msg)`. Les messages qui ne trouvent pas de partenaire sont abandonnés.

### Fréquence d'acquisition (`acquisition_hz`)

Les capteurs publient plus vite que la chaîne ne peut traiter (10 Hz sur KITTI) : YOLO, la projection et le stockage n'ont pas le temps de suivre, et un retard croissant s'accumulerait. Au début de `fusion_callback`, toute paire dont l'horodatage image est à moins de `1 / acquisition_hz` secondes de la dernière paire traitée est donc ignorée.

Le choix de la fréquence est un compromis :

- trop basse : moins de frames par seconde, donc des trous dans la carte quand le robot va vite ;
- trop haute : la chaîne ne suit plus, les files du synchroniseur débordent et des frames sont perdues de façon incontrôlée.

Valeurs utilisées : 4 Hz pour KITTI (`projection_kitti.py`), 2 Hz par défaut pour Scout et Ranger, avec un PC limité en calcul. À 2 Hz, des discontinuités restent visibles sur KITTI, où le véhicule dépasse 50 km/h ; pour des robots terrestres à vitesse de marche, 2 Hz suffit. Sur une machine plus puissante, la fréquence peut être augmentée.

Le filtrage se fait sur l'horodatage du message et non sur l'horloge système : il reste correct quand un bag est rejoué avec `--clock`.

### Deuxième synchronisation : frame / odométrie

C'est une synchronisation distincte, décrite dans `try_process_pending()` et `interpolatation_odom()` : une frame déjà étiquetée attend dans `pending_frames` que l'odométrie de LIORF couvre son horodatage, puis la pose est interpolée à cet instant exact.

## Structures de données

| Nom | Type | Contenu |
|---|---|---|
| `pending_frames` | `deque(maxlen=30)` | Frames étiquetées en attente d'une odométrie qui couvre leur horodatage. |
| `pending_frames_until_loop` | `list` | Toutes les frames depuis le début (seulement si `loop`), pour le recalcul après loop closure. Non bornée. |
| `odom_history` | `deque(maxlen=200)` | Couples `(t_sec, Odometry)` reçus de LIORF. |
| `data_dict` | `dict` de listes | Points accumulés : `x, y, z, label, time, conf_r`. |
| `new_data_dict` | `dict` de listes | Même format, reconstruit après chaque loop closure. Utilisé à la place de `data_dict` pour l'export si `loop` est actif. |
| `prior_map_gdf` | `GeoDataFrame` | Carte a priori, une cellule carrée par entité. |
| `dict_trav` | `dict` | Table label -> traversabilité (0 à 100). |

Une frame est un dictionnaire :

```python
{'points': ndarray (N, 3),   # coordonnées LiDAR d'origine, après filtrage
 'names':  ndarray (N,),     # label YOLO de chaque point
 't_img':  float,            # horodatage de l'image (s)
 'stamp':  Time,             # horodatage du scan LiDAR
 'range':  ndarray (N,)}     # profondeur caméra de chaque point (m)
```

Les poses interpolées sont renvoyées sous forme de `SimpleNamespace` qui imite la structure d'un message `Odometry` (`pose.pose.position` et `pose.pose.orientation`), pour que `apply_transform` accepte indifféremment un vrai message ou une pose interpolée.

## Initialisation

### `__init__()`

Déclare les paramètres ROS, puis selon `robot` (`scout` ou `ranger`) fixe les topics image et LiDAR, la position et l'orientation initiales, les matrices `P2` et `Tr` et la résolution `W, H`.

Calcule ensuite la position initiale normalisée (`x_init_robot - xmin_map`, `y_init_robot - ymin_map`) et la matrice de rotation `R_init` autour de z, charge la carte a priori, charge YOLO, crée les abonnements et le synchroniseur, initialise les structures de stockage et, si `loop` est actif, les abonnements de loop closure.

Pour ajouter un robot, c'est ici qu'il faut ajouter une branche `elif`.

### `geojson_to_dict(geojson_path)`

Lit le GeoJSON de la carte a priori et renvoie `{index: [x, y, label, time]}`. `x` et `y` viennent des propriétés `center_x` / `center_y` si elles existent, sinon du centroïde de la géométrie. `time` vaut 0 s'il est absent.

### `dict_to_gdf(prior_map_dict, crs='EPSG:2154')`

Reconstruit un `GeoDataFrame` à partir du dictionnaire, avec pour chaque entité un carré de côté `cell_size` centré sur `(x, y)`. 

Le carré utilise `cell_size`, la taille de cellule de la carte sémantique. Si la carte a priori a été discrétisée avec un autre pas, le fond affiché ne correspondra pas exactement.

## Callbacks

### `odom_callback(msg)`

Ajoute `(t, msg)` à `odom_history`.

### `loop_callback(msg)`

Reçoit le `MarkerArray` des contraintes de loop closure de LIORF. Le marker contient deux points par contrainte : le nombre de contraintes vaut donc `len(points) / 2`. Si ce nombre augmente, une nouvelle loop closure vient d'être intégrée et `loop_closure_detected` passe à `True`.

### `path_callback(msg)`

Reçoit la trajectoire complète publiée par LIORF. Elle est publiée en continu, donc ignorée tant qu'aucune loop closure n'a été détectée. Sinon elle est stockée dans `corrected_path`, le drapeau est remis à `False` et `recompute_all_points()` est appelé.

### `fusion_callback(image_msg, lidar_msg)`

Appelée par le synchroniseur pour chaque paire image/LiDAR (voir « Synchronisation temporelle LiDAR / caméra »).

1. Ignore la paire si elle arrive moins de `1 / acquisition_hz` secondes après la précédente.
2. Lit les points LiDAR, les passe en coordonnées homogènes, puis calcule `pts_cam = Tr @ p` et `pts_img = P2 @ pts_cam`.
3. Applique les masques : profondeur > 0, `(u, v)` dans l'image, profondeur < `range`. Les mêmes masques sont appliqués aux points LiDAR d'origine.
4. Convertit l'image en OpenCV, lance YOLO, lit le label de chaque point avec `classYolo`.
5. Affiche le contrôle visuel avec `show_opencv`.
6. Ajoute la frame à `pending_frames` (et à `pending_frames_until_loop` si `loop`), puis appelle `try_process_pending`.

Le masque « x > 0 » de KITTI est désactivé ici : sur Scout et Ranger, le filtrage par profondeur caméra suffit.

## Affichage

### `_stamp_to_sec(stamp)`

Convertit un horodatage ROS en secondes (float).

### `show_opencv(u_valid, v_valid, d_valid, cv_image, result)`

Superpose le masque YOLO à l'image, dessine les points LiDAR projetés colorés selon leur profondeur (colormap JET, normalisée sur 50 m), ajoute la légende et, pendant 1 s après une loop closure, le texte « LOOP CLOSURE DETECTED ». N'intervient pas dans le calcul de la carte.

### `semantic_mask(cv_image, result, alpha=0.5)`

Convertit le masque YOLO (tensor torch ou tableau) en image couleur avec une palette HSV d'une couleur par classe, le redimensionne à la taille de l'image si besoin, puis le mélange à l'image. Renvoie l'image mélangée et la palette.

### `draw_legend(cv_image, result, palette)`

Dessine en haut à droite un carré de couleur et le nom de chaque classe, avec la même palette que `semantic_mask`.

## YOLO

### `classYolo(r, u_valid, v_valid)`

Lit l'identifiant de classe dans le masque YOLO au pixel `(v, u)` de chaque point et renvoie le tableau des noms de labels.

Le masque est lu directement, sans redimensionnement : cela suppose que le masque YOLO a la même résolution que l'image caméra.

## Transformation et traitement

### `apply_transform(pts_lidar_3d, odom_msg)`

Construit la matrice 4x4 `T_world_lidar` à partir du quaternion et de la position de la pose, l'applique aux points, puis applique la rotation initiale `R_init`. Le décalage de position initiale n'est pas appliqué ici mais dans les fonctions appelantes.

### `try_process_pending()`

Traite `pending_frames` dans l'ordre d'arrivée. Pour la plus ancienne frame, cherche une pose avec `interpolatation_odom` ; si l'odométrie ne couvre pas encore son horodatage, s'arrête et réessaiera à la frame suivante. Sinon, transforme les points, ajoute le décalage initial, calcule la confiance `conf_r = 100 - range * 100 / max_range`, ajoute chaque point à `data_dict` et retire la frame de la file.

### `recompute_all_points()`

Appelée après une loop closure. Vide `new_data_dict`, puis pour chaque frame de `pending_frames_until_loop` : pose interpolée sur `corrected_path`, transformation, décalage initial, ajout des points. Son coût augmente avec la durée de la mission, puisque toutes les frames depuis le début sont recalculées.

## Interpolations

### `interpolatation_odom(t_target)`

Renvoie la pose à l'instant `t_target` à partir de `odom_history` :

- `None` si l'historique est vide ou si `t_target` est plus récent que la dernière odométrie (la frame reste en attente) ;
- la première odométrie si `t_target` est plus ancien ;
- sinon, interpolation entre les deux odométries qui encadrent `t_target` : linéaire pour la position, sphérique (`Slerp`) pour l'orientation.

### `interpolate_pose_from_path(t_target, poses)`

Même principe sur la trajectoire corrigée (`Path`). `t_target` est borné à l'intervalle de la trajectoire. Cette interpolation est importante car la trajectoire corrigée ne contient que les poses clés, bien moins nombreuses que les messages d'odométrie.

Contrairement à `projection_kitti.py`, cette version ne gère pas le cas d'une trajectoire à une seule pose.

## Export (à l'arrêt du noeud)

### `destroy_node()`

Appelée à l'arrêt (Ctrl+C). Trace la trajectoire si `loop` est actif, lance `parseur_3d()` ou `parseur()` selon `dim`, puis détruit le noeud.

### `parseur()`xs

Carte 2D :

1. Crée un `GeoDataFrame` de points à partir de `data_dict` (ou `new_data_dict`).
2. Construit une grille de carrés de côté `cell_size` sur toute l'emprise de la carte a priori.
3. Jointure spatiale (`sjoin`, `inner`) : seules les cellules contenant des points sont gardées.
4. Par cellule : label dominant (mode), `pct_dominant` = part des points portant ce label.
5. Trace la carte a priori en fond et les cellules colorées par label, enregistre le PNG.
6. Exporte en GeoJSON le centre de chaque cellule avec son label et `pct_dominant`.

La grille couvre toute la carte a priori : avec un petit `cell_size`, le nombre de polygones et le temps de calcul augmentent vite.

### `parseur_3d()`

Carte 3D. `sjoin` et les polygones ne fonctionnant pas en 3D, chaque point reçoit des indices entiers de voxel `(ix, iy, iz)` par division entière, puis les points sont groupés par voxel.

1. Garde les points entre `zmin = -5` et `zmax = 20` m et à l'intérieur de la grille.
2. Par voxel : label dominant, `nb_total`, `pct_dominant`.
3. Sur les seuls points du label dominant : `tmin`, `tmax` (première et dernière observation) et `conf_range` (confiance max).
4. `trav` : traversabilité lue dans `dict_trav` selon le label dominant (0 si absent).
5. Calcule le centre de chaque voxel, trace un nuage 3D coloré par classe, enregistre le PNG.
6. Exporte en GeoJSON un `Point` 3D par voxel avec toutes ces propriétés.

### `write_all()`

Écrit les points bruts (`label, x, y, z`) dans un fichier texte. N'est plus appelée, gardée pour le débogage.

### `plot_path()`

Compare la trajectoire brute (`odom_history`) et la trajectoire corrigée (`corrected_path`) en 2D et enregistre `path/corrected_path.png`.

`odom_history` étant limité à 200 messages, la trajectoire brute affichée ne couvre que la fin de la mission.