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

data_keep=[
		'Owned', '1 Owned', '2 to 4 Owned', '5+ Owned',
		'Mobile Home Owned', 'Other Owned', 'Rented', '1 Rented',
		'2 to 4 Rented', '5+ Rented', 'Mobile Home Rented', 'Other Rented',
		'Total', 'White alone', 'Black or African American alone',
		'American Indian and Alaska Native alone', 'Asian alone',
		'Native Hawaiian and Other Pacific Islander alone',
		'Some other race alone', 'Two or more races',
		'Two races including Some other race',
		'Two races excluding Some other race and three or more races',
		'Median Income']

def MakeEmptyHex(gdf,resolution=8):
	#Converting to EPSG 4326 for compatibility with H3 Hexagons
	gdf=gdf.to_crs(epsg=4326)
	#Making the empty hex
	gdf_empty=h3fy(gdf,resolution=resolution)
	return gdf_empty

def DataToHex(gdf_tracts,gdf_hex,data_keys=data_keep):
	print(data_keys)
	gdf_tracts=gdf_tracts.to_crs(epsg=2163)
	gdf_hex=gdf_hex.to_crs(epsg=2163)
	gdf_out=area_interpolate(gdf_tracts,gdf_hex,extensive_variables=data_keys)
	gdf_out=gdf_out.to_crs(epsg=4326)
