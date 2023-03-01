import sys
import time
import json
import requests
import warnings
import numpy as np
import numpy.random as rand
import pandas as pd
import pickle as pkl
import reverse_geocoder as rg
import pandas as pd
import geopandas as gpd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from shapely.ops import cascaded_union
from itertools import combinations
from shapely.geometry import Polygon
from scipy.stats import t
from scipy.stats._continuous_distns import _distn_names
from .mapbox_router import RouteSummary

#Function for calculating distances between lon/lat pairs
def haversine(lon1,lat1,lon2,lat2):

	r=6372800 #[m]

	dLat=np.radians(lat2-lat1)
	dLon=np.radians(lon2-lon1)
	lat1=np.radians(lat1)
	lat2=np.radians(lat2)

	a=np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
	c=2*np.arcsin(np.sqrt(a))

	return c*r

def OutlierIndices(x):
	return ((x>x.mean()+x.std()*3)|(x<x.mean()-x.std()*3))

def OutliersFilter(resources_lons,resources_lats,resources):

	distances_from_mean=haversine(resources_lons,resources_lats,
		resources_lons.mean(),resources_lats.mean())
	outliers=OutlierIndices(distances_from_mean)

	return resources_lons[~outliers],resources_lats[~outliers],resources[~outliers]

def RemoveOutlierResources(resources_lons,resources_lats,resources):

	resources_lons=np.array(resources_lons)
	resources_lats=np.array(resources_lats)
	resources=np.array(resources)

	len_resources=resources.size

	for idx in range(10):

		resources_lons,resources_lats,resources=Filter(resources_lons,resources_lats,resources)

		if len_resources == resources.size:
			break
		else:
			len_resources=resources.size

	return resources_lons,resources_lats,resources

#Function for loading in charger data
def LoadAFDCData(filename='Data/AFDC/evse_stations.json',lon=None,lat=None,radius=None):

	data=json.load(open(filename))
	df_all=pd.DataFrame(data["fuel_stations"])

	if all([lon,lat,radius]):
		radii=haversine(lon,lat,df_all['longitude'].to_numpy(),df_all['latitude'].to_numpy())
		df_all=df_all[radii<=radius]
		df_all.reset_index(inplace=True,drop=True)

	return df_all

#Function for determining Destination Charger Likelihood (DCL)
def ComputeDCL(resources_lons,resources_lats,resources,centroid_indices,df_fs,charger_distance):
	resources_lons=np.array(resources_lons)
	resources_lats=np.array(resources_lats)
	resources=np.array(resources)
	unique_centroid_indices=np.unique(centroid_indices).astype(int)
	dcl=np.zeros(unique_centroid_indices.shape)
	dest_chargers=np.zeros(unique_centroid_indices.shape)
	destinations=np.zeros(unique_centroid_indices.shape)
	df_fs_lvl2=df_fs[df_fs['ev_level2_evse_num']>0]
	for idx,centroid_idx in enumerate(unique_centroid_indices):
		
		# print(resources_lons[centroid_indices==centroid_idx].shape)
		lon_resources_g,lon_chargers_g=np.meshgrid(resources_lons[centroid_indices==centroid_idx],
			df_fs_lvl2['longitude'].to_numpy(),indexing='ij')
		lat_resources_g,lat_chargers_g=np.meshgrid(resources_lats[centroid_indices==centroid_idx],
			df_fs_lvl2['latitude'].to_numpy(),indexing='ij')
		distances=haversine(lon_resources_g,lat_resources_g,lon_chargers_g,lat_chargers_g)
		# print(distances.shape,lon_resources_g.shape,centroid_idx,
		# 	((distances<=charger_distance).sum(axis=1)>0).sum())
		dcl[idx]=((distances<=charger_distance).sum(axis=1)>0).sum()/lon_resources_g.shape[0]
		dest_chargers[idx]=((distances<=charger_distance).sum(axis=1)>0).sum()
		destinations[idx]=lon_resources_g.shape[0]
		# print('{} {} {} {}'.format(idx,dcl[idx],dest_chargers[idx],destinations[idx]),end='\r')
		# break
	return dcl,dest_chargers,destinations

def ComputeDCFCP(gdf_cb,df_fs):
	centroids_x=gdf_cb['centroid_x']
	centroids_y=gdf_cb['centroid_y']
	# print(centroids_x.shape)
	distances=np.zeros(centroids_x.shape)
	durations=np.zeros(centroids_x.shape)
	dcfcp=np.zeros(centroids_x.shape)
	df_fs_dcfc=df_fs[df_fs['ev_dc_fast_num']>0]
	for idx in tqdm(range(len(dcfcp))):
		lon_chargers_g=df_fs_dcfc['longitude'].to_numpy()
		lat_chargers_g=df_fs_dcfc['latitude'].to_numpy()
		charger_distances=haversine(centroids_x[idx],centroids_y[idx],lon_chargers_g,lat_chargers_g)
		min_dist_idx=np.argmin(charger_distances)
		route=RouteSummary((centroids_x[idx],centroids_y[idx]),
			(lon_chargers_g[min_dist_idx],lat_chargers_g[min_dist_idx]))
		distances[idx]=route['distances']
		durations[idx]=route['durations']
		dcfcp[idx]=durations[idx]*2
	return dcfcp,distances,durations

def CalculateSIC_NHTS_HO(gdf,model_pickle,hc_col,wc_val,dcfcr_val,bc_val,dcl2=False,dcfcp2=False):
	model,df1,data,maxes,mins=pkl.load(open(model_pickle,'rb'))
	# print(maxes,mins)
	hc_idx,wc_idx,dcl_idx,bc_idx,dcfcr_idx,dcfcp_idx=(1,2,3,7,10,11)
	bc=np.ones(gdf.shape[0])*(bc_val-mins[bc_idx])/(maxes[bc_idx]-mins[bc_idx])
	# hc=np.ones(gdf.shape[0])*hc_val
	hc=gdf[hc_col]
	# print(hc,hc.min(),hc.max(),np.isnan(hc).sum())
	# hc=gdf['Housing__1']/gdf['Housing__1'].max()
	wc=np.ones(gdf.shape[0])*wc_val
	dcl=(gdf['DCL'].to_numpy()-mins[dcl_idx])/(maxes[dcl_idx]-mins[dcl_idx])
	# dcl[dcl<.5]=dcl.mean()
	# print(dcl.mean(),(dcl==0).sum())
	if dcl2:
		dcl[dcl<dcl.max()/2]=dcl.max()
		# dcl*=1
	# print(dcl.mean(),(dcl==0).sum())
	# dcl[:]=dcl.max()
	dcfcr=np.ones(gdf.shape[0])*(dcfcr_val-mins[dcfcr_idx])/(maxes[dcfcr_idx]-mins[dcfcr_idx])
	dcfcp=(gdf['DCFCP']*60-mins[dcfcp_idx])/(maxes[dcfcp_idx]-mins[dcfcp_idx])
	# dcfcp[dcfcp>.5]=dcfcp.mean()
	if dcfcp2:
		dcfcp[dcfcp>dcfcp.min()*2]=dcfcp.min()*2
	# dcfcp[:]=dcfcp.min()
	# print(dcfcp.to_numpy())
	df=pd.DataFrame(np.vstack((bc,hc,wc,dcl,dcfcr,dcfcp)).T,columns=['BC','HC','WC','DCL','DCFCR','DCFCP'])
	# print(df)
	sic=model.predict(df)
	# print(sic.mean())
	sic[sic<0]=0
	return sic,hc




#Function for down-selecting census blocks by centroid distance from point
def DownSlectBlocks(gdf,lon,lat,radius):
	radii=haversine(lon,lat,gdf['centroid_x'].to_numpy(),gdf['centroid_y'].to_numpy())
	gdf_new=gdf[radii<=radius].copy()
	gdf_new.reset_index(inplace=True,drop=True)
	return gdf_new

#Function for pulling resource locations for a series of block centroids
def PullDataCentroids(bmdp,lons,lats):
	resources_lons,resources_lats,resources=[],[],[]
	centroid_lons,centroid_lats,centroid_indices=[],[],[]
	for idx in tqdm(range(len(lons))):
	# for idx in tqdm(range(5)):
		try:
			rlon,rlat,r=bmdp.PullResources(lons[idx],lats[idx])
			# print(len(resources_lons),len(rlon))
			resources_lons.extend(rlon)
			resources_lats.extend(rlat)
			resources.extend(r)
			centroid_lons.extend([lons[idx]]*len(r))
			centroid_lats.extend([lats[idx]]*len(r))
			centroid_indices.extend([idx]*len(r))
		except:
			print(idx)
			pass
	return resources_lons,resources_lats,resources,centroid_lons,centroid_lats,centroid_indices

#Class for generating pull requests, pulling, and processing data from Bing Maps
class BingMapsDataPuller():
	def __init__(self,key):
		self.key=key

	def PullResources(self,lon,lat):
		pull_shop,pull_eat_drink,pull_see_do=self.GeneratePullRequests(lon,lat)
		# print(pull_shop,pull_eat_drink,pull_see_do)
		return self.Pull(pull_shop,pull_eat_drink,pull_see_do)

	def GeneratePullRequests(self,lon,lat):
		pull_shop='''
		https://dev.virtualearth.net/REST/v1/LocalSearch/?userCircularMapView=
		{:.6f},{:.6f},5000&type=Shop&maxResults=25&key={}
		'''.format(lat,lon,self.key)
		pull_shop="".join(line.strip() for line in pull_shop.splitlines())
		pull_eat_drink='''
		https://dev.virtualearth.net/REST/v1/LocalSearch/?userCircularMapView=
		{:.6f},{:.6f},5000&type=EatDrink&maxResults=25&key={}
		'''.format(lat,lon,self.key)
		pull_eat_drink="".join(line.strip() for line in pull_eat_drink.splitlines())
		pull_see_do='''
		https://dev.virtualearth.net/REST/v1/LocalSearch/?userCircularMapView=
		{:.6f},{:.6f},5000&type=SeeDo&maxResults=25&key={}
		'''.format(lat,lon,self.key)
		pull_see_do="".join(line.strip() for line in pull_see_do.splitlines())
		# print(pull_shop,pull_eat_drink,pull_see_do)
		return pull_shop,pull_eat_drink,pull_see_do

	def Pull(self,pull_shop,pull_eat_drink,pull_see_do):
		results_shop=requests.get(pull_shop).json()
		results_eat_drink=requests.get(pull_eat_drink).json()
		results_see_do=requests.get(pull_see_do).json()
		resources=np.array((
			results_shop['resourceSets'][0]['resources']+
			results_eat_drink['resourceSets'][0]['resources']+
			results_see_do['resourceSets'][0]['resources']))
		# print(len(resources))
		resources_lons,resources_lats=np.zeros(len(resources)),np.zeros(len(resources))
		success=[False]*(len(resources))
		for idx,resource in enumerate(resources):
			try:
				resources_lons[idx]=resource['point']['coordinates'][1]
				resources_lats[idx]=resource['point']['coordinates'][0]
				success[idx]=True
			except:
				pass
		return resources_lons[success],resources_lats[success],resources[success]