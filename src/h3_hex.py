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
from scipy.spatial import cKDTree

import warnings
warnings.filterwarnings("ignore")

default_interpolation_keys=['Total_Housing','Owned', '1 Owned', '2 to 4 Owned',
		'5+ Owned', 'Mobile Home Owned',
		'Other Owned', 'Rented', '1 Rented', '2 to 4 Rented', '5+ Rented',
		'Mobile Home Rented', 'Other Rented', 'Total_Race',
		'White Alone', 'Black or African American Alone',
		'American Indian and Alaska Native Alone', 'Asian Alone',
		'Native Hawaiian and Other Pacific Islander Alone',
		'Some Other Race Alone', 'Two or More Races']

default_keep_closest_keys=['Median Income']

def ContainsCentroid(source_gdf,target_gdf):
	indices=np.zeros(target_gdf.shape[0])
	source_indices=np.arange(0,source_gdf.shape[0],1).astype(int)
	for idx in range(target_gdf.shape[0]):
		indices[idx]=source_indices[source_gdf.contains(target_gdf['geometry'][idx].centroid)]

	return indices


def MakeEmptyHex(gdf,resolution=8):
	#Converting to EPSG 4326 for compatibility with H3 Hexagons
	gdf=gdf.to_crs(epsg=4326)
	#Making the empty hex
	gdf_empty=h3fy(gdf,resolution=resolution)
	gdf_empty.reset_index(inplace=True)
	return gdf_empty

def KeepClosest(source_gdf,target_gdf,columns):

	# source_xy=np.stack([source_gdf.geometry.centroid.x,
	# 	source_gdf.geometry.centroid.y],-1)

	# target_xy=np.stack([target_gdf.geometry.centroid.x,
	# 	target_gdf.geometry.centroid.y],-1)

	# tree=cKDTree(source_xy)

	# distances,indices=tree.query(target_xy)

	indices=ContainsCentroid(source_gdf,target_gdf)

	new_data=source_gdf.loc[indices,columns].to_numpy()

	target_gdf[columns]=new_data

	return target_gdf

def DataToHex(gdf_tracts,gdf_hex,interpolation_keys=default_interpolation_keys,
	keep_closest_keys=default_keep_closest_keys):
	gdf_tracts=gdf_tracts.to_crs(epsg=2163)
	gdf_hex=gdf_hex.to_crs(epsg=2163)
	if len(interpolation_keys)>0:
		gdf_out=area_interpolate(gdf_tracts,gdf_hex,extensive_variables=interpolation_keys)
	if len(keep_closest_keys)>0:
		gdf_out=KeepClosest(gdf_tracts,gdf_out,keep_closest_keys)
	gdf_out=gdf_out.to_crs(epsg=4326)

	return gdf_out

def DataToTracts(gdf_hex,gdf_tracts,extensive_variables=[],intensive_variables=[],
	nearest_variables=[]):
	gdf_tracts=gdf_tracts.to_crs(epsg=2163)
	gdf_hex=gdf_hex.to_crs(epsg=2163)
	if len(extensive_variables)>0:
		gdf_out=area_interpolate(gdf_hex,gdf_tracts,extensive_variables=extensive_variables)
	if len(intensive_variables)>0:
		gdf_out=area_interpolate(gdf_hex,gdf_tracts,intensive_variables=intensive_variables)
	if len(nearest_variables)>0:
		gdf_out=KeepClosest(gdf_out,gdf_tracts,nearest_variables)
	gdf_out=gdf_out.to_crs(epsg=4326)

	return gdf_out
