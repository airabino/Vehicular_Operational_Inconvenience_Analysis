import os
import sys
import time
import numpy as np
import pandas as pd
import pickle as pkl
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

import h3
import h3pandas

from tobler.util import h3fy
from tobler.area_weighted import area_interpolate

import warnings
warnings.filterwarnings("ignore")

def MakeEmptyHex(gdf,resolution=8):
	#Converting to EPSG 4326 for compatibility with H3 Hexagons
	gdf=gdf.to_crs(epsg=4326)
	#Making the empty hex
	gdf_empty=h3fy(gdf,resolution=resolution)
	return gdf_empty

def DataToH3(data,resolution=4,lat_col='centroid_x',lng_col='centroid_y'):
	df_h3=data.h3.geo_to_h3_aggregate(resolution,lat_col=lat_col,lng_col=lng_col)
	gdf_h3=gpd.GeoDataFrame(df_h3).set_crs(epsg=4326).copy()
	return gdf_h3