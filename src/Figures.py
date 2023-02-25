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
# from matplotlib.patches import Circle, Wedge, Polygon
from shapely.geometry import Point
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from shapely.ops import cascaded_union
from itertools import combinations
from shapely.geometry import Polygon,MultiPolygon
from scipy.stats import t
from scipy.stats._continuous_distns import _distn_names
from MapBoxRouter import *

def T_Test(x,y,alpha):
    x_n=len(x)
    y_n=len(y)
    x_mu=x.mean()
    y_mu=y.mean()
    x_sig=x.std()
    y_sig=y.std()
    x_se=x_sig/np.sqrt(x_n)
    y_se=y_sig/np.sqrt(y_n)
    x_y_se=np.sqrt(x_se**2+y_se**2)
    T=(x_mu-y_mu)/x_y_se
    DF=x_n+y_n
    T0=t.ppf(1-alpha,DF)
    P=(1-t.cdf(np.abs(T),DF))*2
    return (P<=alpha),T,P,T0,DF

def Correlation(x,y):
	n=len(x)
	return (n*(x*y).sum()-x.sum()*y.sum())/np.sqrt((n*(x**2).sum()-x.sum()**2)*(n*(y**2).sum()-y.sum()**2))

def Determination(x,y):
	return Correlation(x,y)**2

def FitBestDist(data,bins=200,dist_names=['alpha','beta','gamma','logistic','norm','erlang','lognorm']):
	# dist_names=_distn_names
	# print(dist_names)
	# dist_names.remove('studentized_range')
	densities,bin_edges=np.histogram(data,bins,density=True)
	bin_edges = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0
	bin_edges_plot=np.linspace(bin_edges.min(),bin_edges.max(),1000)
	# print(bin_edges)
	fig=plt.figure(figsize=(8,8))
	# plt.plot(bin_edges,densities,label='densities')
	plt.hist(data,bins=bins,rwidth=.85,ec='black',fc='gray',alpha=.5,lw=3,density=True,label='densities')
	sse=np.empty(len(dist_names))
	run_times=np.empty(len(dist_names))
	dists=[None]*len(dist_names)
	params_list=[None]*len(dist_names)
	for idx,dname in enumerate(dist_names):
		# print([idx,len(dist_names),dname],end='\r')
		dist=getattr(st,dname)
		try:
			with warnings.catch_warnings():
				warnings.filterwarnings('ignore')
				t0=time.time()
				params=dist.fit(data)
				run_times[idx]=time.time()-t0
				arg = params[:-2]
				loc = params[-2]
				scale = params[-1]
				y=dist.pdf(bin_edges,loc=loc,scale=scale,*arg)
				y_plot=dist.pdf(bin_edges_plot,loc=loc,scale=scale,*arg)
				sse[idx]=np.sqrt(((y-densities)**2).sum()/len(y))
				plt.plot(bin_edges_plot,y_plot,lw=5,color='k')
				plt.plot(bin_edges_plot,y_plot,label='{}: RMSE={:.4f}, loc={:.4f}, scale={:.4f}'.format(
					dname,sse[idx],loc,scale),lw=3)
				dists[idx]=dist
				params_list[idx]=params
		except Exception as e:
			print(e)
			sse[idx]=sys.maxsize
	plt.xlabel('SIC [min/km]')
	plt.ylabel('Density [dim]')
	plt.legend()
	plt.grid(linestyle='--')
	return fig,sse,run_times,dist_names,dists,params_list

def CalculateSIC(bc,hc,dcl,ercr,ercp):
	params=[.568,-.123,-.462,-.309,-.340,.347,.257,-.181,.272,-.279,.188,-.171]
	sic=(params[0]+params[1]*bc+params[2]*hc+params[3]*dcl+params[4]*ercr+params[5]*ercp+
		params[6]*dcl*hc+params[7]*ercp*bc+params[8]*ercr*hc+params[9]*ercp*hc+
		params[10]*ercr*dcl+params[11]*ercp*dcl)
	return sic

def OutlierIndices(x):
	return ((x>x.mean()+x.std()*3)|(x<x.mean()-x.std()*3))

def FilterOutliers(resources_lons,resources_lats,resources):
	resources_lons=np.array(resources_lons)
	resources_lats=np.array(resources_lats)
	resources=np.array(resources)
	outliers=OutlierIndices(resources_lons)
	resources_lons=resources_lons[~outliers]
	resources_lats=resources_lats[~outliers]
	resources=resources[~outliers]
	outliers=OutlierIndices(resources_lons)
	resources_lons=resources_lons[~outliers]
	resources_lats=resources_lats[~outliers]
	resources=resources[~outliers]
	outliers=OutlierIndices(resources_lons)
	resources_lons=resources_lons[~outliers]
	resources_lats=resources_lats[~outliers]
	resources=resources[~outliers]
	return resources_lons,resources_lats,resources

#Function for loading in charger data
def LoadChargerData(filename,lon=None,lat=None,radius=None):
	data=json.load(open(filename))
	df_all=pd.DataFrame(data["fuel_stations"])
	if all([lon,lat,radius]):
		radii=haversine(lon,lat,df_all['longitude'].to_numpy(),df_all['latitude'].to_numpy())
		df_all=df_all[radii<=radius]
		df_all.reset_index(inplace=True)
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

def ComputeERCP_Naive(gdf_cb,df_fs,avg_speed=15.65):
	df_fs_dcfc=df_fs[df_fs['ev_dc_fast_num']>0]
	lon_centroids_g,lon_chargers_g=np.meshgrid(gdf_cb['centroid_x'],
		df_fs_dcfc['longitude'].to_numpy(),indexing='ij')
	lat_centroids_g,lat_chargers_g=np.meshgrid(gdf_cb['centroid_y'],
		df_fs_dcfc['latitude'].to_numpy(),indexing='ij')
	distances=haversine(lon_centroids_g,lat_centroids_g,lon_chargers_g,lat_chargers_g)
	return distances.min(axis=1)/avg_speed/60*2

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

def CensusBlocksPlot(gdf_ds,gdf_cb,margin=.05):
	fig,ax=plt.subplots(figsize=(8,8))
	# gdf_cb.plot(ax=ax,facecolor=np.array([[12,192,170]])/255,edgecolor='k',alpha=.2)
	# gdf_ds.plot(ax=ax,facecolor=np.array([[12,192,170]])/255,edgecolor='k')
	gdf_cb.plot(ax=ax,facecolor='gray',edgecolor='k',alpha=.2)
	gdf_ds.plot(ax=ax,facecolor='w',edgecolor='k')
	ax.scatter(gdf_ds['centroid_x'],gdf_ds['centroid_y'],s=5,c=np.array([[28,91,90]])/255,label='Centroids')
	minx=gdf_ds.bounds['minx'].min()
	maxx=gdf_ds.bounds['maxx'].max()
	miny=gdf_ds.bounds['miny'].min()
	maxy=gdf_ds.bounds['maxy'].max()
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	ax.legend()
	ax.set_aspect('equal', 'box')
	return fig

def DCFCPlot(gdf_ds,gdf_cb,df_fs,margin=.05):
	fig,ax=plt.subplots(figsize=(8,8))
	# gdf_cb.plot(ax=ax,facecolor=np.array([[12,192,170]])/255,edgecolor='k',alpha=.2)
	# gdf_ds.plot(ax=ax,facecolor=np.array([[12,192,170]])/255,edgecolor='k')
	gdf_cb.plot(ax=ax,facecolor='gray',edgecolor='k',alpha=.2)
	gdf_ds.plot(ax=ax,facecolor='w',edgecolor='k')
	df_fs_dcfc=df_fs[df_fs['ev_dc_fast_num']>0]
	ax.scatter(df_fs_dcfc['longitude'],df_fs_dcfc['latitude'],
		s=50,c='r',label='DCFC Stations',edgecolor='k')
	# ax.scatter(gdf_ds['centroid_x'],gdf_ds['centroid_y'],s=5,c=np.array([[28,91,90]])/255,label='Centroids')
	minx=gdf_ds.bounds['minx'].min()
	maxx=gdf_ds.bounds['maxx'].max()
	miny=gdf_ds.bounds['miny'].min()
	maxy=gdf_ds.bounds['maxy'].max()
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	ax.legend()
	ax.set_aspect('equal', 'box')
	return fig

def LVL2Plot(gdf_ds,gdf_cb,df_fs,margin=.05):
	fig,ax=plt.subplots(figsize=(8,8))
	# gdf_cb.plot(ax=ax,facecolor=np.array([[12,192,170]])/255,edgecolor='k',alpha=.2)
	# gdf_ds.plot(ax=ax,facecolor=np.array([[12,192,170]])/255,edgecolor='k')
	gdf_cb.plot(ax=ax,facecolor='gray',edgecolor='k',alpha=.2)
	gdf_ds.plot(ax=ax,facecolor='w',edgecolor='k')
	df_fs_dcfc=df_fs[df_fs['ev_level2_evse_num']>0]
	ax.scatter(df_fs_dcfc['longitude'],df_fs_dcfc['latitude'],
		s=50,c='b',label='LVL2 Stations',edgecolor='k')
	# ax.scatter(gdf_ds['centroid_x'],gdf_ds['centroid_y'],s=5,c=np.array([[28,91,90]])/255,label='Centroids')
	minx=gdf_ds.bounds['minx'].min()
	maxx=gdf_ds.bounds['maxx'].max()
	miny=gdf_ds.bounds['miny'].min()
	maxy=gdf_ds.bounds['maxy'].max()
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	ax.legend()
	ax.set_aspect('equal', 'box')
	return fig

def DestinationsScatterPlot(gdf_ds,gdf_cb,resources_lons,resources_lats,margin=.05):
	fig,ax=plt.subplots(figsize=(8,8))
	# gdf_cb.plot(ax=ax,facecolor=np.array([[53,97,143]])/255,edgecolor='k',alpha=.2)
	# gdf_ds.plot(ax=ax,facecolor=np.array([[53,97,143]])/255,edgecolor='k')
	# ax.scatter(resources_lons,resources_lats,s=5,c=np.array([[150,234,78]])/255,label='Resources')
	gdf_cb.plot(ax=ax,facecolor='gray',edgecolor='k',alpha=.2)
	gdf_ds.plot(ax=ax,facecolor='w',edgecolor='k')
	ax.scatter(resources_lons,resources_lats,s=15,c='g',label='Resources',edgecolor='k')
	minx=gdf_ds.bounds['minx'].min()
	maxx=gdf_ds.bounds['maxx'].max()
	miny=gdf_ds.bounds['miny'].min()
	maxy=gdf_ds.bounds['maxy'].max()
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	ax.legend()
	ax.set_aspect('equal', 'box')
	return fig

#Function for plotting DCL
def DCLChoropleth(gdf_ds,gdf_cb,margin=.05,cmap='viridis'):
	fig,ax=plt.subplots(figsize=(8,8))
	gdf_cb.plot(ax=ax,facecolor='gray',edgecolor='k',alpha=.2)
	cs=gdf_ds.plot(ax=ax,column='DCL',edgecolor='k',cmap=cmap)
	# ax.set_title('Destination Charger Likelihood (DCL) For Denver CO Area by Census Block')
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	minx=gdf_ds.bounds['minx'].min()
	maxx=gdf_ds.bounds['maxx'].max()
	miny=gdf_ds.bounds['miny'].min()
	maxy=gdf_ds.bounds['maxy'].max()
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	vmin = 0
	vmax = gdf_ds['DCL'].max()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('bottom', size='2%', pad=.5)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm._A = []
	cbr = fig.colorbar(sm, cax=cax, orientation='horizontal')
	cbr.ax.set_xlabel('Proportion of Long Dwell Locations with Nearby Charger [dim]',
		labelpad=10,fontsize='medium')
	ax.set_aspect('equal', 'box')
	return fig

#Function for plotting ERCP
def DCFCPChoropleth(gdf_ds,gdf_cb,margin=.05,cmap='viridis',column='DCFCP'):
	fig,ax=plt.subplots(figsize=(8,8))
	gdf_cb.plot(ax=ax,facecolor='gray',edgecolor='k',alpha=.2)
	cs=gdf_ds.plot(ax=ax,column=column,edgecolor='k',cmap=cmap)
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	minx=gdf_ds.bounds['minx'].min()
	maxx=gdf_ds.bounds['maxx'].max()
	miny=gdf_ds.bounds['miny'].min()
	maxy=gdf_ds.bounds['maxy'].max()
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	vmin = 0
	vmax = gdf_ds[column].max()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('bottom', size='2%', pad=.5)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm._A = []
	cbr = fig.colorbar(sm, cax=cax, orientation='horizontal')
	cbr.ax.set_xlabel('Round-Trip Travel Time to Nearest DCFC Station [min]',
		labelpad=10,fontsize='medium')
	ax.set_aspect('equal', 'box')
	return fig

def HousingChoropleth(gdf_ds,gdf_cb,margin=.05,cmap='viridis',column='1_owned'):
	fig,ax=plt.subplots(figsize=(8,8))
	gdf_cb.plot(ax=ax,facecolor='gray',edgecolor='k',alpha=.2)
	gdf_ds['plot']=gdf_ds[column]/gdf_ds['total']
	# print(gdf_ds['plot'].max(),gdf_ds[column].max())
	cs=gdf_ds.plot(ax=ax,column='plot',edgecolor='k',cmap=cmap)
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	minx=gdf_ds.bounds['minx'].min()
	maxx=gdf_ds.bounds['maxx'].max()
	miny=gdf_ds.bounds['miny'].min()
	maxy=gdf_ds.bounds['maxy'].max()
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	vmin = 0
	vmax = 1
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('bottom', size='2%', pad=.5)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm._A = []
	cbr = fig.colorbar(sm, cax=cax, orientation='horizontal')
	cbr.ax.set_xlabel('Proportion of Residents [dim]',
		labelpad=10,fontsize='medium')
	ax.set_aspect('equal', 'box')
	return fig

def CalculateSIC_NHTS(gdf,model_pickle,hc_val,wc_val,dcfcr_val,bc_val):
	model,df1,data,maxes,mins=pkl.load(open(model_pickle,'rb'))
	# print(maxes,mins)
	hc_idx,wc_idx,dcl_idx,bc_idx,dcfcr_idx,dcfcp_idx=(1,2,3,7,10,11)
	bc=np.ones(gdf.shape[0])*(bc_val-mins[bc_idx])/(maxes[bc_idx]-mins[bc_idx])
	hc=np.ones(gdf.shape[0])*hc_val
	# hc=gdf['Housing__1']/gdf['Housing__1'].max()
	wc=np.ones(gdf.shape[0])*wc_val
	dcl=np.ones(gdf.shape[0])*(gdf['DCL']-mins[dcl_idx])/(maxes[dcl_idx]-mins[dcl_idx])
	# dcl[dcl<.5]=dcl.mean()
	dcl[dcl<dcl.max()/2]=dcl.max()/2
	# dcl[:]=dcl.max()
	dcfcr=np.ones(gdf.shape[0])*(dcfcr_val-mins[dcfcr_idx])/(maxes[dcfcr_idx]-mins[dcfcr_idx])
	dcfcp=np.ones(gdf.shape[0])*(gdf['DCFCP']*60-mins[dcfcp_idx])/(maxes[dcfcp_idx]-mins[dcfcp_idx])
	# dcfcp[dcfcp>.5]=dcfcp.mean()
	# dcfcp[dcfcp>dcfcp.min()*2]=dcfcp.min()*2
	# dcfcp[:]=dcfcp.min()
	# print(dcfcp.to_numpy())
	df=pd.DataFrame(np.vstack((bc,hc,wc,dcl,dcfcr,dcfcp)).T,columns=['BC','HC','WC','DCL','DCFCR','DCFCP'])
	sic=model.predict(df)
	return sic

def CalculateSIC_NHTS_HO_LA(gdf,model_pickle,hc_col,wc_val,dcfcr_val,bc_val,lodes_array):
	model,df1,data,maxes,mins=pkl.load(open(model_pickle,'rb'))
	hc_idx,wc_idx,dcl_idx,bc_idx,dcfcr_idx,dcfcp_idx=(1,2,3,7,10,11)
	bc=np.ones(gdf.shape[0])*(bc_val-mins[bc_idx])/(maxes[bc_idx]-mins[bc_idx])
	hc=gdf[hc_col]/gdf['total']
	wc=np.ones(gdf.shape[0])*wc_val
	dcl=gdf['DCL'].to_numpy()
	dcl=ApplyLODESArray(dcl,lodes_array)
	dcl=(dcl-mins[dcl_idx])/(maxes[dcl_idx]-mins[dcl_idx])
	dcfcr=np.ones(gdf.shape[0])*(dcfcr_val-mins[dcfcr_idx])/(maxes[dcfcr_idx]-mins[dcfcr_idx])
	dcfcp=gdf['DCFCP'].to_numpy()
	dcfcp=ApplyLODESArray(dcfcp,lodes_array)
	dcfcp=(dcfcp*60-mins[dcfcp_idx])/(maxes[dcfcp_idx]-mins[dcfcp_idx])
	df=pd.DataFrame(np.vstack((bc,hc,wc,dcl,dcfcr,dcfcp)).T,columns=['BC','HC','WC','DCL','DCFCR','DCFCP'])
	sic=model.predict(df)
	sic[sic<0]=0
	return sic,hc

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


def SICDChoropleth(gdf_ds,gdf_cb,margin=.05,cmap='viridis',column='SICD',vmax=None):
	vmin = 0
	if vmax==None:
		vmax = gdf_ds[column].max()
	fig,ax=plt.subplots(figsize=(8,8))
	gdf_cb.plot(ax=ax,facecolor='gray',edgecolor='k',alpha=.2)
	cs=gdf_ds.plot(ax=ax,column=column,edgecolor='k',cmap=cmap,vmin=vmin,vmax=vmax)
	# ax.set_title('Distance Inconvenience Score (SICD) For Denver CO Area by Census Block')
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	minx=gdf_ds.bounds['minx'].min()
	maxx=gdf_ds.bounds['maxx'].max()
	miny=gdf_ds.bounds['miny'].min()
	maxy=gdf_ds.bounds['maxy'].max()
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('bottom', size='2%', pad=.5)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm._A = []
	cbr = fig.colorbar(sm, cax=cax, orientation='horizontal')
	cbr.ax.set_xlabel('SIC [min/km]',
		labelpad=10,fontsize='medium')
	ax.set_aspect('equal', 'box')
	return fig,vmax

def SICTChoropleth(gdf_ds,gdf_cb,margin=.05,cmap='viridis',column='SICT'):
	fig,ax=plt.subplots(figsize=(10,10))
	gdf_cb.plot(ax=ax,facecolor='gray',edgecolor='k',alpha=.2)
	cs=gdf_ds.plot(ax=ax,column=column,edgecolor='k',cmap=cmap)
	ax.set_title('Trip Inconvenience Score (SICT) For Denver CO Area by Census Block')
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	minx=gdf_ds.bounds['minx'].min()
	maxx=gdf_ds.bounds['maxx'].max()
	miny=gdf_ds.bounds['miny'].min()
	maxy=gdf_ds.bounds['maxy'].max()
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	vmin = 0
	vmax = gdf_ds[column].max()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('bottom', size='2%', pad=.5)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm._A = []
	cbr = fig.colorbar(sm, cax=cax, orientation='horizontal')
	cbr.ax.set_xlabel('SICT [min/trip]',
		labelpad=10,fontsize='medium')

def DestChargersPlot(gdf_cb):
	fig,ax=plt.subplots(figsize=(10,10))
	cs=gdf_cb.plot(ax=ax,column='dest_chargers',edgecolor='k')
	ax.set_title('Number of Destination Chargers For Denver CO Area by Census Block')
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	vmin = 0
	vmax = gdf_cb['dest_chargers'].max()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('bottom', size='2%', pad=.5)
	sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm._A = []
	cbr = fig.colorbar(sm, cax=cax, orientation='horizontal')
	cbr.ax.set_xlabel('Proportion of Long Dwell Locations with Nearby Charger [dim]',
		labelpad=10,fontsize='medium')

def DestinationsPlot(gdf_cb):
	fig,ax=plt.subplots(figsize=(10,10))
	cs=gdf_cb.plot(ax=ax,column='destinations',edgecolor='k')
	ax.set_title('Number of Popular Destinations Returned For Denver CO Area by Census Block (Max = 75)')
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	vmin = 0
	vmax = gdf_cb['destinations'].max()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('bottom', size='2%', pad=.5)
	sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm._A = []
	cbr = fig.colorbar(sm, cax=cax, orientation='horizontal')
	cbr.ax.set_xlabel('Proportion of Long Dwell Locations with Nearby Charger [dim]',
		labelpad=10,fontsize='medium')

#Function for plotting ERCP
def IncomePlot(gdf_ds,gdf,column='Housing__1',margin=.05):
	fig,ax=plt.subplots(figsize=(10,10))
	gdf.plot(ax=ax,facecolor='gray',edgecolor='k',alpha=.2)
	cs=gdf_ds.plot(ax=ax,column=column,edgecolor='k')
	ax.set_title('Mean Income Denver CO Area by Census Block')
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	minx=gdf_ds.bounds['minx'].min()
	maxx=gdf_ds.bounds['maxx'].max()
	miny=gdf_ds.bounds['miny'].min()
	maxy=gdf_ds.bounds['maxy'].max()
	# print(minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin)
	# print()
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	vmin = 0
	vmax = gdf_ds[column].max()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('bottom', size='2%', pad=.5)
	sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm._A = []
	cbr = fig.colorbar(sm, cax=cax, orientation='horizontal')
	cbr.ax.set_xlabel('Round-Trip Travel Time to Nearest DCFC Station [min]',
		labelpad=10,fontsize='medium')