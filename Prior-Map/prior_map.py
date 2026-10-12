import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

class PriorMap:
    """
    author = Calamia Dan
    """

    def __init__(self):
        
        # GeoJson data
        path_to_data = "./data/large_esprit.geojson"
        self.gdf = gpd.read_file(path_to_data)
        # transformation latitude and longitude to coordinates in meters
        self.gdf = self.gdf.to_crs(epsg=2154) # Lambert-93 / EPSG:2154

        # state 
        self.x = np.array([])

        # Discretization and Map parameters
        self.step=1 # m

    def discrtization(self,show=False, save=False):
        # define boundary
        xmin,ymin,xmax,ymax = self.gdf.total_bounds
        print(f" min max {self.gdf.total_bounds}")

        # set offset to define origin on (xmin,ymin) --> (0,0)
        # move all geometries
        self.gdf['geometry'] = self.gdf.geometry.translate(xoff=-xmin, yoff=-ymin)
        
        # define new boundary
        xmin_new,ymin_new,xmax_new,ymax_new = self.gdf.total_bounds
        self.size = [xmin_new, ymin_new, xmax_new, ymax_new]
        print(self.gdf.total_bounds)
        print(self.size)

        # surface of map
        surface = (xmax_new-xmin_new)*(ymax_new-ymin_new)
        print(f'Surface : {surface/1e6} km2')
        
        # creating grid
        cols = np.arange(xmin_new, xmax_new, self.step)
        rows = np.arange(ymin_new, ymax_new, self.step)
        
        # filling grid
        squares = []
        for x in cols:
            for y in rows:
                squares.append(Polygon([(x, y), (x + self.step, y), (x + self.step, y + self.step), (x, y + self.step)]))

        # creating, crs=self.gdf.crs as our reference system (GeoDataFrame) 
        grid = gpd.GeoDataFrame({'geometry': squares}, crs=self.gdf.crs)

        # joint space between data (way, building etc) and squares created (GeoDataFrame)
        grid_filtered = gpd.sjoin(grid, self.gdf, how="inner", predicate="intersects")

        # plot
        # if show :
        #     grid_filtered.plot(edgecolor='white')
        #     plt.show()
        # if save :
        #     grid_filtered.plot(edgecolor='white')
        #     plt.savefig(f'Images/map_discretized_{self.step}m.svg',format = 'svg', bbox_inches='tight', pad_inches=0, dpi=300)

        return grid_filtered
    
    def classify_cell(self, row):

        """
        Use to clean the database or labeling OSM forms.
        Example :
        BEFORE --> building : yes , forest: no , road : no
        AFTER label : building

        labels free : building, highway, forest or ground if nothing

        """

        # Buildings 
        if pd.notnull(row.get('building')):
            return 'building'
        
        # paths
        if pd.notnull(row.get('highway')):
            return 'road'
        
        if pd.notnull(row.get('railway')):
            return 'railway'

        # others
        if pd.notnull(row.get('leisure')):
            if row.get('leisure') in ['park', 'garden', 'nature_reserve']:
                return 'park'
            return 'leisure' # Pour les 'pitch', 'stadium', etc.
        if pd.notnull(row.get('amenity')):
            return 'amenity' # Regroupe écoles, parkings, hôpitaux

        # forest
        forest_tags = ['forest', 'orchard', 'vineyard']
        wood_tags = ['wood', 'scrub', 'heath']
        if row.get('landuse') in forest_tags or row.get('natural') in wood_tags:
            return 'forest'
        
        # water 
        if pd.notnull(row.get('water')) or row.get('natural') == 'water' or row.get('waterway'):
            return 'water'

        # grass 
        if row.get('landuse') in ['grass', 'meadow', 'farmland']:
            return 'grass'
        # other
        return 'ground'
    
    def export_grid(self,grid,show=False):

        # get center of all sqaures
        grid['center_x'] = grid.geometry.centroid.x
        grid['center_y'] = grid.geometry.centroid.y

        # creating label 
        grid['label'] = grid.apply(self.classify_cell, axis=1) # axis 1 : on columns, apply the fonction classify_cell to all columns
        
        # Interested columns for the final GeoJson file
        final_columns = ['geometry', 'center_x', 'center_y', 'label','id']
        grid_export = grid[final_columns]
        
        # plot grid after labeling 
        # create dictionary from self.labels
        if show:
            colors = {
                'building': 'red',
                'road': 'black',
                'forest': 'forestgreen',
                'grass': 'lightgreen',
                'water': 'blue',
                'ground': 'grey'
            }

            grid_export.plot(column='label', categorical=True, legend=True, color=[colors.get(l, 'grey') for l in grid_export['label']],edgecolor='white',linewidth=0.1,figsize=(15, 8))
            plt.savefig(f'Images/map_step{self.step}m.png', bbox_inches='tight', pad_inches=0, dpi=300)
            plt.show()
        
        # export
        output_path = './exported_data/export_grid.geojson'
        grid_export.to_file(output_path, driver='GeoJSON')

        return grid_export
        
    def save_for_gazebo(self,grid_filtered):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # buildings : dark
        buildings = grid_filtered[grid_filtered['label'] == 'building']
        buildings.plot(ax=ax, color='black')

        ax.set_xlim(self.size[0], self.size[2])
        ax.set_ylim(self.size[1], self.size[3])
        ax.set_axis_off()
        ax.set_aspect('equal')
        
        plt.savefig(f'Images/map_gazebo_building_{self.step}.png', bbox_inches='tight', pad_inches=0, dpi=300)

        fig2, ax2 = plt.subplots(figsize=(10, 10))

        # paths : dark 
        paths = grid_filtered[grid_filtered['label'] == 'road']
        paths.plot(ax=ax2, color='black')

        ax2.set_xlim(self.size[0], self.size[2])
        ax2.set_ylim(self.size[1], self.size[3])
        ax2.set_axis_off()
        ax2.set_aspect('equal')

        plt.savefig(f'Images/map_gazebo_path_{self.step}.png', bbox_inches='tight', pad_inches=0, dpi=300)


        print("Uploaded Images for gz")

if __name__ == "__main__":
    method = PriorMap()
    g = method.discrtization(show=False)
    method.export_grid(g,show=True) # g GeoDataFrame is modified in export_grid function by classify cell, It contains all columns like older and labels, center etc. export_grid Geodataframe contains only the newest colums refer to the exported data.
    method.save_for_gazebo(g)
