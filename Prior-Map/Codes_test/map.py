import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

lidar_data = pd.read_csv('ton_fichier.dat', sep=',')

