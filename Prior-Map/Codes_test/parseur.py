import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon # ou Polygon, Box, etc. test parseur dict to geojson file

# test exemple de dictionnaire 
data_dict = {
    'id': [1, 2, 3],
    'center_x': [110, 110.2, 150.8],
    'center_y': [111.1, 111.24, .2],
    # 'center_z' : [1, 1, 1],
    'label': ['building', 'road', 'forest'],
    'time': ['0.1', '0.2', '0.3'],# acquisition de la camera header 
    # faculatif , moyen de les mettre si seulement présent
    # 'confiance': ['0', '0', '0'],
    # 'traversabillité': ['0', '0.5', '1'], # 
    # 'obstacle statique': ['building', 'road', 'forest'],
    # 'obstacle dynamique': ['voiture', 'personne'],
}

# Convertir le dictionnaire en un DataFrame standard
df = pd.DataFrame(data_dict)

# Créer les géométries géographiques à partir des coordonnées
geometry = [Point(x, y) for x, y in zip(df['center_x'], df['center_y'])]

# Convertir en GeoDataFrame en spécifiant le CRS (ex: Lambert-93 EPSG:2154)
gdf_dict = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:2154")


# Créationd de la grille 800 par 400 pour couvrir toute la map du data set kitti
xmin,xmax = 0,800
ymin,ymax = 0,400
step = 10
import numpy as np
cols = np.arange(xmin, xmax, step)
rows = np.arange(ymin, ymax, step)

# filling grid
squares = []
for x in cols:
    for y in rows:
        squares.append(Polygon([(x, y), (x + step, y), (x + step, y + step), (x, y + step)]))

# creating, crs=self.gdf.crs as our reference system (GeoDataFrame) 
grid = gpd.GeoDataFrame({'geometry': squares}, crs=gdf_dict.crs)

# join ,
grid_filtered = gpd.sjoin(grid, gdf_dict, how="left", predicate="intersects")


show =True
import matplotlib.pyplot as plt
if show:
    colors = {
        'building': 'red',
        'road': 'black',
        'forest': 'forestgreen',
        'grass': 'lightgreen',
        'water': 'blue',
        'ground': 'grey'
    }
    
    grid_filtered.plot(column='label', categorical=True, legend=True, color=[colors.get(l, 'grey') for l in grid_filtered['label']],edgecolor='white',linewidth=0.1,figsize=(15, 8))
    plt.savefig(f'Images/map_step{step}m.png', bbox_inches='tight', pad_inches=0, dpi=300)
    grid_filtered.to_file('gridmap.geojson', driver="GeoJSON")
    plt.show()


# Exporter au format GeoJSON

output_path1 = './exported_data/dict_export.geojson'
output_path2 = './exported_data/grille_export.geojson'
gdf_dict.to_file(output_path1, driver='GeoJSON')
grid_filtered.to_file(output_path2, driver='GeoJSON')

print(f"Fichier sauvegardé avec succès dans : {output_path1}")


