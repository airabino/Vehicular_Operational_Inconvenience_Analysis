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
from .mapbox_router import GetRouter,RouteSummary

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
def ComputeDCL(gdf,df_chargers,charger_distance_threshold=50):

	df_chargers_ac_level_2=df_chargers[df_chargers['ev_level2_evse_num']>0]

	dcl=np.zeros(gdf.shape[0])

	for idx in tqdm(range(gdf.shape[0])):
		try:
			resources_lons=np.array(gdf['resource_lons'][idx])
			resources_lats=np.array(gdf['resource_lats'][idx])
			resources=np.array(gdf['resources'][idx])

			# resources_lons,resources_lats,resources=RemoveOutlierResources(
			# 	resources_lons,resources_lats,resources)

			lon_resources_g,lon_chargers_g=np.meshgrid(resources_lons,
				df_chargers_ac_level_2['longitude'].to_numpy(),indexing='ij')
			lat_resources_g,lat_chargers_g=np.meshgrid(resources_lats,
				df_chargers_ac_level_2['latitude'].to_numpy(),indexing='ij')

			distances=haversine(lon_resources_g,lat_resources_g,lon_chargers_g,lat_chargers_g)

			dcl[idx]=(((distances<=charger_distance_threshold).sum(axis=1)>0).sum()/
				lon_resources_g.shape[0])
		except:
			pass

	gdf['DCL']=dcl

	return gdf

def ComputeDCFCP(gdf,df_chargers,key):

	router=GetRouter(key)

	centroids_x=gdf.centroid.x
	centroids_y=gdf.centroid.y

	durations=np.zeros(centroids_x.shape)
	dcfcp=np.zeros(centroids_x.shape)

	df_chargers_dc_level_1_2=df_chargers[df_chargers['ev_dc_fast_num']>0]
	lon_chargers_g=df_chargers_dc_level_1_2['longitude'].to_numpy()
	lat_chargers_g=df_chargers_dc_level_1_2['latitude'].to_numpy()

	for idx in tqdm(range(len(dcfcp))):
		
		charger_distances=haversine(centroids_x[idx],centroids_y[idx],
			lon_chargers_g,lat_chargers_g)
		min_dist_idx=np.argmin(charger_distances)

		route=RouteSummary(router,(centroids_x[idx],centroids_y[idx]),
			(lon_chargers_g[min_dist_idx],lat_chargers_g[min_dist_idx]))

		durations[idx]=route['durations']

		dcfcp[idx]=durations[idx]*2

	gdf['DCFCP']=dcfcp

	return gdf

def CalculateHomeChargingAvailability(gdf,file='Home_Charging_Availability.xlsx'):

	hca=pd.read_excel(file)

	hca_scenario_1=HomeChargingScenario(hca['Number of Respondents in the Sample'].to_numpy(),
		hca['Scenario 1: Discounted Existing Electrical Access'].to_numpy())
	gdf['HC Scenario 1']=hca_scenario_1.Calculate(gdf)

	hca_scenario_2=HomeChargingScenario(hca['Number of Respondents in the Sample'].to_numpy(),
		hca['Scenario 2: Existing Electrical Access'].to_numpy())
	gdf['HC Scenario 2']=hca_scenario_2.Calculate(gdf)

	hca_scenario_3=HomeChargingScenario(hca['Number of Respondents in the Sample'].to_numpy(),
		hca['Scenario 3: Existing Electrical Access (w/ parking behavior mod)'].to_numpy())
	gdf['HC Scenario 3']=hca_scenario_3.Calculate(gdf)

	hca_scenario_4=HomeChargingScenario(hca['Number of Respondents in the Sample'].to_numpy(),
		hca['Scenario 4: Enhanced Electrical Access'].to_numpy())
	gdf['HC Scenario 4']=hca_scenario_4.Calculate(gdf)

	hca_scenario_5=HomeChargingScenario(hca['Number of Respondents in the Sample'].to_numpy(),
		hca['Scenario 5: Enhanced Electrical Access (w/ parking behavior mod)'].to_numpy())
	gdf['HC Scenario 5']=hca_scenario_5.Calculate(gdf)

	hca_scenario_6=HomeChargingScenario(hca['Number of Respondents in the Sample'].to_numpy(),
		hca['Scenario 6: No Home Charging'].to_numpy())
	gdf['HC Scenario 6']=hca_scenario_6.Calculate(gdf)

	hca_scenario_7=HomeChargingScenario(hca['Number of Respondents in the Sample'].to_numpy(),
		hca['Scenario 7: Complete Home Charging'].to_numpy())
	gdf['HC Scenario 7']=hca_scenario_7.Calculate(gdf)

	return gdf

class HomeChargingScenario():
	def __init__(self,respondents,percentages):

		self.respondents=respondents
		self.percentages=percentages

		self.Populate()

	def WeightedSum(self,indices):
		return ((self.respondents[indices]*self.percentages[indices]).sum()/
			self.respondents[indices].sum())

	def Calculate(self,gdf):

		keys=(['1 Owned','2 to 4 Owned','5+ Owned','Mobile Home Owned','Other Owned',
			'1 Rented','2 to 4 Rented','5+ Rented','Mobile Home Rented','Other Rented'])

		out=np.zeros(gdf.shape[0])

		for key in keys:

			out+=gdf[key].to_numpy()*self.percentage_availability[key]

		out/=gdf['Total_Housing'].to_numpy()

		return out


	def Populate(self):

		self.percentage_availability={}
		self.percentage_availability['1 Owned']=self.WeightedSum([0,2])
		self.percentage_availability['2 to 4 Owned']=self.WeightedSum([7])
		self.percentage_availability['5+ Owned']=self.WeightedSum([7])
		self.percentage_availability['Mobile Home Owned']=self.WeightedSum([8])
		self.percentage_availability['Other Owned']=self.WeightedSum([10])
		self.percentage_availability['1 Rented']=self.WeightedSum([1,3])
		self.percentage_availability['2 to 4 Rented']=self.WeightedSum([6])
		self.percentage_availability['5+ Rented']=self.WeightedSum([4,5])
		self.percentage_availability['Mobile Home Rented']=self.WeightedSum([9])
		self.percentage_availability['Other Rented']=self.WeightedSum([10])

def CalculateSIC(gdf,model_pickle,hc_col,wc_val,dcfcr_val,bc_val,dcl2=False,dcfcp2=False):
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