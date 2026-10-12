import geopandas as gpd
import pandas as pd

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