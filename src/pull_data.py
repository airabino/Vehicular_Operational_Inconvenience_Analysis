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

if __name__ == "__main__":
	argv=sys.argv
	if len(argv)>=2:
		PullData(argv[1])
	else:
		PullData()
