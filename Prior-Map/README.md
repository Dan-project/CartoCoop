# Carte a priori - prior_map.py

`prior_map.py` construit la carte a priori utilisée par LOSY. Il part d'un export OpenStreetMap, le découpe en grille de cellules carrées, donne un label à chaque cellule et exporte le résultat en GeoJSON.

## Fonctionnement

1. **Chargement** : lit `data/large_esprit.geojson`, exporté depuis [Overpass Turbo](https://overpass-turbo.eu/) (alentours du bâtiment ESPRIT).
2. **Projection** : convertit les coordonnées GPS (latitude, longitude) en mètres, en Lambert 93 (EPSG:2154).
3. **Changement d'origine** : décale toute la carte pour que son coin inférieur gauche soit en (0, 0).
4. **Discrétisation** : crée une grille de cellules de 1 m (`self.step`) et ne garde que les cellules qui touchent un élément OSM.
5. **Classification** : donne un label à chaque cellule à partir des tags OSM. Quand un élément a plusieurs tags, le premier de cette liste l'emporte :

   `building` > `road` > `railway` > `park` / `leisure` > `amenity` > `forest` > `water` > `grass` > `ground` (par défaut)

6. **Export** : enregistre pour chaque cellule sa géométrie, son centre (`center_x`, `center_y`), son label et l'identifiant OSM d'origine.

## Lancer le script

Les chemins sont relatifs : le script doit être lancé depuis son propre dossier.

```bash
python3 prior_map.py
```

## Fichiers produits

| Fichier | Contenu |
|---|---|
| `exported_data/export_grid.geojson` | La carte a priori, à copier dans `losy/priormap/` |
| `Images/map_step<step>m.png` | Aperçu de la carte classée |
| `Images/map_gazebo_building_<step>.png` | Bâtiments seuls, en noir, pour construire un monde Gazebo |
| `Images/map_gazebo_path_<step>.png` | Routes seules, en noir, pour construire un monde Gazebo |

## Utiliser une autre zone

### Exporter la zone depuis Overpass Turbo

Sélectionner la zone, exporter en GeoJSON, puis placer le fichier dans `data/` et modifier `path_to_data` dans le script.

### Choisir la taille des cellules

Modifier `self.step` (en mètres). Un pas plus fin donne une carte plus précise mais plus lourde à calculer.

### Reporter l'emprise dans LOSY

Au lancement, le script affiche les bornes de la zone en Lambert 93 (ligne `min max [xmin ymin xmax ymax]`). Ces quatre valeurs doivent être reportées dans les paramètres `xmin_map`, `ymin_map`, `xmax_map` et `ymax_map` des noeuds LOSY, sinon la carte sémantique ne sera pas alignée avec la carte a priori.

## Limites

- Pas d'altitude : la carte est uniquement en 2D.
- La classification ne garde que quelques grandes catégories ; les détails OSM sont regroupés.
