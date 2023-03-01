import os
import sys
import time
import requests
import numpy as np
import pandas as pd
import pickle as pkl
import pandas as pd
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import matplotlib.pyplot as plt
from io import StringIO
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

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

def ACS_DF(table,data_name=None,state_FIPS='08'):
	pull_request='''
	https://api.census.gov/data/2021/acs/acs5?get=NAME,{}&for=tract:*&in=state:{}
	'''.format(table,state_FIPS)
	# print(pull_request)
	pull_request="".join(line.strip() for line in pull_request.splitlines())
	# print(pull_request)
	results=requests.get(pull_request)
	content=results._content
	content_str=str(content)
	content_str=content_str.replace('[','')
	content_str=content_str.replace(']','')
	content_str=content_str.replace('"','')
	content_str=content_str.replace('b\'','')
	content_str=content_str.replace('\'','')
	content_str=content_str.replace(',\\n','\n')
	df=pd.read_csv(StringIO(content_str),dtype=str)
	df['FIPS']=df['state']+df['county']+df['tract']
	df[[table,'tract','county','state']]=df[[table,'tract','county','state']].apply(pd.to_numeric)
	if data_name!=None:
		df.rename(columns={table:data_name},inplace=True)
	return df

def ACS_Series(table,data_name=None,state_FIPS='08'):
	pull_request='''
	https://api.census.gov/data/2021/acs/acs5?get=NAME,{}&for=tract:*&in=state:{}
	'''.format(table,state_FIPS)
	# print(pull_request)
	pull_request="".join(line.strip() for line in pull_request.splitlines())
	# print(pull_request)
	results=requests.get(pull_request)
	content=results._content
	content_str=str(content)
	content_str=content_str.replace('[','')
	content_str=content_str.replace(']','')
	content_str=content_str.replace('"','')
	content_str=content_str.replace('b\'','')
	content_str=content_str.replace('\'','')
	content_str=content_str.replace(',\\n','\n')
	df=pd.read_csv(StringIO(content_str),dtype=str)
	# df[[table]]=df[[table]].apply(pd.to_numeric)
	if data_name!=None:
		df.rename(columns={table:data_name},inplace=True)
	else:
		data_name=table
	# print(data_name)
	# print(df)
	return pd.to_numeric(df[data_name])

def ACS_AddColumn(df,table,data_name=None,state_FIPS='08'):
	series=ACS_Series(table,data_name=data_name,state_FIPS=state_FIPS)
	df[series._name]=series
	return df

def ACS_Pull_Columns(columns_dict,state_FIPS='08'):
	#Keys from columns DataFrame
	keys_list=list(columns_dict.keys())
	#Creating DataFrame around first column
	df=ACS_DF(keys_list[0],state_FIPS=state_FIPS,data_name=columns_dict[keys_list[0]])
	#Looping through remaining keys
	for key in tqdm(keys_list[1:]):
		df=ACS_AddColumn(df,key,state_FIPS=state_FIPS,data_name=columns_dict[key])
	#Re-setting index
	df.reset_index(inplace=True,drop=True)
	#Dropping unnecessary columns
	df.drop(columns=['NAME','tract','county','state'],inplace=True)

	return df

def ACS_PullTable(table,data_name=None,state_FIPS='08'):
	pull_request='''
	https://api.census.gov/data/2021/acs/acs5?get=NAME,{}&for=tract:*&in=state:{}
	'''.format(table,state_FIPS)
	# print(pull_request)
	pull_request="".join(line.strip() for line in pull_request.splitlines())
	# print(pull_request)
	results=requests.get(pull_request)
	content=results._content
	content_str=str(content)
	content_str=content_str.replace('[','')
	content_str=content_str.replace(']','')
	content_str=content_str.replace('"','')
	content_str=content_str.replace('b\'','')
	content_str=content_str.replace('\'','')
	content_str=content_str.replace(',\\n','\n')
	df=pd.read_csv(StringIO(content_str),dtype=str)
	print(df)
	df['FIPS']=df['state']+df['county']+df['tract']
	df[[table,'tract','county','state']]=df[[table,'tract','county','state']].apply(pd.to_numeric)
	if data_name!=None:
		df.rename(columns={table:data_name},inplace=True)
	return df

def ACS_PullGroup(table,data_name=None,state_FIPS='08'):
	pull_request='''
	https://api.census.gov/data/2021/acs/acs5?get=group({})&for=tract:*&in=state:{}
	'''.format(table,state_FIPS)
	pull_request="".join(line.strip() for line in pull_request.splitlines())
	results=requests.get(pull_request)
	content=results._content
	content_str=str(content)
	content_str=content_str.replace('[','')
	content_str=content_str.replace(']','')
	content_str=content_str.replace('"','')
	content_str=content_str.replace('b\'','')
	content_str=content_str.replace('\'','')
	content_str=content_str.replace(',\\n','\n')
	df=pd.read_csv(StringIO(content_str),dtype=str)
	n=len(table)+5
	columns=list(df.keys())
	drop=[]
	for key in columns:
		if (key[:len(table)]==table)&((len(key)!=n)|(key[-1]!='E')):
			drop.append(key)
	df=df.drop(columns=drop)
	df['FIPS']=df['state']+df['county']+df['tract']
	df[['tract','county','state']]=df[['tract','county','state']].apply(pd.to_numeric)
	if data_name!=None:
		df.rename(columns={table:data_name},inplace=True)
	df.reset_index(inplace=True,drop=True)
	return df

def ACS_Rename(df,shells):
	uids=shells['UniqueID'].to_numpy()
	stub=shells['Stub'].to_numpy()
	keys=df.keys()
	rename_dict={}
	for key in keys:
		idx=np.argwhere(uids==key[:-1]).flatten()
		# print(idx,idx.shape)
		if idx.shape[0]==1:
			# print(np.argwhere(uids==key[:-1]))
			rename_str=stub[uids==key[:-1]][0]
			rename_str=rename_str.replace(':','')
			rename_str=rename_str.replace('$','')
			rename_str=rename_str.replace(',','')
			rename_dict[key]=rename_str
	# print(rename_dict)
	df.rename(columns=rename_dict,inplace=True)
	return df

def ACS_Fix_Total(df,data_keys):
	data=df[data_keys].to_numpy().astype(int)
	totals=data.sum(axis=1)
	df['Total']=totals
	return df

def PullData(state_FIPS='08',path_to_data='../Data/'):
	#Loading in the shells .csv
	t0=time.time()
	print('Loading in the shells .csv:',end='')
	shells=pd.read_csv('{}ACS_2021/ACS2021_Table_Shells.csv'.format(path_to_data))
	print(' {:.4f} seconds'.format(time.time()-t0))

	#Pulling race data
	t0=time.time()
	print('Pulling race data:',end='')
	race_columns={
		'B02001_001E':'Total_Race',
		'B02001_002E':'White Alone',
		'B02001_003E':'Black or African American Alone',
		'B02001_004E':'American Indian and Alaska Native Alone',
		'B02001_005E':'Asian Alone',
		'B02001_006E':'Native Hawaiian and Other Pacific Islander Alone',
		'B02001_007E':'Some Other Race Alone',
		'B02001_008E':'Two or More Races'}
	df_race=ACS_Pull_Columns(race_columns,state_FIPS='08')
	print(' {:.4f} seconds'.format(time.time()-t0))

	#Pulling housing type and tenure data
	t0=time.time()
	print('Pulling housing type and tenure data:',end='')
	housing_columns={
		'B25033_001E':'Total_Housing',
		'B25033_002E':'Owned',
		'B25033_003E':'1 Owned',
		'B25033_004E':'2 to 4 Owned',
		'B25033_005E':'5+ Owned',
		'B25033_006E':'Mobile Home Owned',
		'B25033_007E':'Other Owned',
		'B25033_008E':'Rented',
		'B25033_009E':'1 Rented',
		'B25033_010E':'2 to 4 Rented',
		'B25033_011E':'5+ Rented',
		'B25033_012E':'Mobile Home Rented',
		'B25033_013E':'Other Rented'}
	df_housing=ACS_Pull_Columns(housing_columns,state_FIPS='08')
	print(' {:.4f} seconds'.format(time.time()-t0))

	#Pulling median income data
	t0=time.time()
	print('Pulling median income data:',end='')
	income_columns={
		'B19013_001E':'Median Income'}
	df_income=ACS_Pull_Columns(income_columns,state_FIPS='08')
	df_income['Median Income'][df_income['Median Income']<0]=np.nan
	print(' {:.4f} seconds'.format(time.time()-t0))

	#Loading in census tract geometry data
	t0=time.time()
	print('Loading in census tract geometry data:',end='')
	us_ct=gpd.read_file('{}ACS_2021/Tract_Geometries/cb_2021_us_tract_500k.shp'.format(path_to_data))
	co_ct=us_ct[us_ct['STATEFP']==state_FIPS].copy()
	co_ct['FIPS']=co_ct['GEOID']
	print(' {:.4f} seconds'.format(time.time()-t0))

	#Sorting the DataFrames
	t0=time.time()
	print('Sorting the DataFrames:',end='')
	co_ct.sort_values('FIPS',inplace=True)
	co_ct.reset_index(inplace=True,drop=True)

	df_race.sort_values('FIPS',inplace=True)
	df_race.reset_index(inplace=True,drop=True)

	df_income.sort_values('FIPS',inplace=True)
	df_income.reset_index(inplace=True,drop=True)

	df_housing.sort_values('FIPS',inplace=True)
	df_housing.reset_index(inplace=True,drop=True)
	print(' {:.4f} seconds'.format(time.time()-t0))

	#Joining the DataFrames
	t0=time.time()
	print('Joining the DataFrames:',end='')
	co_ct=co_ct.join(df_housing,lsuffix='_housing')
	co_ct=co_ct.join(df_race,lsuffix='_race')
	co_ct=co_ct.join(df_income,lsuffix='_income')
	print(' {:.4f} seconds'.format(time.time()-t0))

	#Pickling
	t0=time.time()
	print('Pickling:',end='')
	pkl.dump(co_ct,open('{}Generated_Data/Census_Tract_Demographic_Data_{}.pkl'.format(
		path_to_data,state_FIPS),'wb'))
	print(' {:.4f} seconds'.format(time.time()-t0))

	print('Done')

def LoadACSData(filepath):
	gdf=pkl.load(open(filepath,'rb'))
	gdf['centroid_x']=gdf.centroid.x
	gdf['centroid_y']=gdf.centroid.y
	return gdf

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

def AddMaxRadii(gdf):

	#Pulling bounds
	bounds=gdf.bounds

	#Pulling bbox coords
	minx=bounds['minx'].to_numpy()
	miny=bounds['miny'].to_numpy()
	maxx=bounds['maxx'].to_numpy()
	maxy=bounds['maxy'].to_numpy()

	#calculating radii
	max_radii=haversine(minx,miny,maxx,maxy)/2

	#Adding radii
	gdf['max_radius']=max_radii

	return gdf

def AddCentroidLonLat(gdf):

	#Adding the centroids
	gdf['centroid_lon']=gdf['geometry'].centroid.x
	gdf['centroid_lat']=gdf['geometry'].centroid.y

	return gdf

#Class for generating pull requests, pulling, and processing data from Bing Maps
class BingMapsDataPuller():
	def __init__(self,key):
		self.key=key

	def Pull_GDF(self,gdf):

		#Adding centroid locations to gdf
		gdf=AddCentroidLonLat(gdf)

		#Initializing loop varaible
		resources_lons=[None]*gdf.shape[0]
		resources_lats=[None]*gdf.shape[0]
		resources=[None]*gdf.shape[0]

		#Main loop
		for idx in tqdm(range(gdf.shape[0])):
		# for idx in tqdm(range(5)):

			lon=gdf['centroid_lon'][idx]
			lat=gdf['centroid_lat'][idx]

			try:
				rlon,rlat,r=self.PullResources(lon,lat)
				resources_lons[idx]=rlon
				resources_lats[idx]=rlat
				resources[idx]=r

			except:
				print(idx)
				pass

		#Adding data to gdf
		gdf['resource_lons']=resources_lons
		gdf['resource_lats']=resources_lats
		gdf['resources']=resources

		return gdf

	def PullResources(self,lon,lat,radius=5000):

		pull_shop,pull_eat_drink,pull_see_do=self.GeneratePullRequests(lon,lat,radius)

		return self.Pull(pull_shop,pull_eat_drink,pull_see_do)

	def GeneratePullRequests(self,lon,lat,radius):

		pull_shop='''
		https://dev.virtualearth.net/REST/v1/LocalSearch/?userCircularMapView=
		{:.6f},{:.6f},{:.2f}&type=Shop&maxResults=25&key={}
		'''.format(lat,lon,radius,self.key)
		pull_shop="".join(line.strip() for line in pull_shop.splitlines())

		pull_eat_drink='''
		https://dev.virtualearth.net/REST/v1/LocalSearch/?userCircularMapView=
		{:.6f},{:.6f},{:.2f}&type=EatDrink&maxResults=25&key={}
		'''.format(lat,lon,radius,self.key)
		pull_eat_drink="".join(line.strip() for line in pull_eat_drink.splitlines())

		pull_see_do='''
		https://dev.virtualearth.net/REST/v1/LocalSearch/?userCircularMapView=
		{:.6f},{:.6f},{:.2f}&type=SeeDo&maxResults=25&key={}
		'''.format(lat,lon,radius,self.key)
		pull_see_do="".join(line.strip() for line in pull_see_do.splitlines())

		return pull_shop,pull_eat_drink,pull_see_do

	def Pull(self,pull_shop,pull_eat_drink,pull_see_do):

		results_shop=requests.get(pull_shop).json()
		results_eat_drink=requests.get(pull_eat_drink).json()
		results_see_do=requests.get(pull_see_do).json()

		resources=np.array((
			results_shop['resourceSets'][0]['resources']+
			results_eat_drink['resourceSets'][0]['resources']+
			results_see_do['resourceSets'][0]['resources']))

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