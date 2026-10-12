import geopandas as gpd
import pandas as pd


def geojson_to_dict(geojson_path): 
    """
    Lit le GeoJSON (FeatureCollection de Polygons) et le retransforme
    en dictionnaire { id : [x, y, label, time] }
    """
    # Charger le GeoJSON
    gdf = gpd.read_file(geojson_path)
    
    prior_map_dict = {}
    
    # Parcourir chaque entité
    for idx, row in gdf.iterrows():
        # Utilise la propriété 'id' du GeoJSON s'il existe, sinon l'index de la ligne
        # feature_id = int(row['id']) if 'id' in row and not pd.isna(row['id']) else (idx + 1)
        
        # Récupération de x et y depuis les propriétés center_x/center_y (ou centroïde du Polygon en fallback)
        x = row['center_x'] if 'center_x' in row and not pd.isna(row['center_x']) else row.geometry.centroid.x
        y = row['center_y'] if 'center_y' in row and not pd.isna(row['center_y']) else row.geometry.centroid.y
        
        label = row['label']
        time_val = row['time'] if 'time' in row and not pd.isna(row['time']) else 0.0
        
        # Construction du dictionnaire au format exact
        prior_map_dict[idx] = [x, y, label, time_val]

        # debug 
        # for i, (k, v) in enumerate(prior_map_dict.items()):
        #     if i >= 10:
        #         break
        #     print(f"{k}, {v}")
                
    return prior_map_dict


def main():
    geojson_to_dict("./exported_data/export_grid.geojson")

main()