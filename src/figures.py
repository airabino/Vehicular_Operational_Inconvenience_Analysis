import os
os.environ['USE_PYGEOS'] = '0'
import sys
import time
import json
import requests
import warnings
import numpy as np
import numpy.random as rand
import pandas as pd
import geopandas as gpd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from shapely.ops import cascaded_union
from itertools import combinations
from shapely.geometry import Point,Polygon,MultiPolygon
from scipy.stats import t
from scipy.stats._continuous_distns import _distn_names

#Defining some 3 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_3_0=["#72e5ef", "#1c5b5a", "#2da0a1"]
color_scheme_3_1=["#256676", "#72b6bc", "#1eefc9"]
color_scheme_3_2=['#40655e', '#a2e0dd', '#31d0a5']

#Defining some 2 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_2_0=["#21f0b6", "#2a6866"]
color_scheme_2_1=["#72e5ef", "#3a427d"]

#Distributions to try (scipy.stats continuous distributions)
dist_names=['alpha','beta','gamma','logistic','norm','lognorm']
dist_labels=['Alpha','Beta','Gamma','Logistic','Normal','Log Normal']

def SelectionPlot(selected,background,figsize=(8,8),margin=.05,alpha=1,colors=color_scheme_2_1,ax=None):

	minx=selected.bounds['minx'].min()
	maxx=selected.bounds['maxx'].max()
	miny=selected.bounds['miny'].min()
	maxy=selected.bounds['maxy'].max()

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	ax.set_prop_cycle(color=colors)

	background.plot(ax=ax,fc=colors[1],ec='k',alpha=alpha)
	selected.plot(ax=ax,fc=colors[0],ec='k',alpha=alpha)
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')
	# ax.set_aspect('equal','box')

	if return_fig:
		return fig

def TractsHexComparisonPlot(tracts,h3_hex,background,figsize=(12,6),margin=.05,alpha=1,
	colors=color_scheme_2_1):
	
	fig,ax=plt.subplots(1,2,figsize=figsize)
	SelectionPlot(tracts,background,ax=ax[0],margin=margin,alpha=alpha,
		colors=colors)
	SelectionPlot(h3_hex,background,ax=ax[1],margin=margin,alpha=alpha,
		colors=colors)

	return fig

def TractsHexAreaHistogram(tracts,h3_hex,figsize=(8,8),cutoff=2e4,bins=100,colors=color_scheme_2_1):

	fig,ax=plt.subplots(figsize=figsize)

	ax.set_facecolor('lightgray')

	out=ax.hist(tracts.to_crs(2163).area/1e3,bins=bins,rwidth=.9,color=colors[1],ec='k',label='Census Tracts')
	ax.plot([h3_hex.to_crs(2163).area.mean()/1e3]*2,[out[0].min(),out[0].max()],lw=7,color='k')
	ax.plot([h3_hex.to_crs(2163).area.mean()/1e3]*2,[out[0].min(),out[0].max()],lw=5,color=colors[0],
		label='Hex Cells')
	ax.grid(ls='--')
	ax.set_xlim([0,cutoff])
	ax.legend()
	ax.set_xlabel('Geometry Area [km]')
	ax.set_ylabel('Bin Size [-]')

	return fig

def DataColumnPlot(selected,background,figsize=(8,8),margin=.05,alpha=1,colors=color_scheme_2_1,ax=None,
	column=None,color_axis_label=None):
	
	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	minx=selected.bounds['minx'].min()
	maxx=selected.bounds['maxx'].max()
	miny=selected.bounds['miny'].min()
	maxy=selected.bounds['maxy'].max()

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	ax.set_prop_cycle(color=colors)

	background.plot(ax=ax,fc='lightgray',ec='k',alpha=alpha)
	im=selected.plot(ax=ax,column=column,ec='k',alpha=alpha,cmap=cmap)
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	ax.set_xlabel('Longitude [deg]')
	ax.set_ylabel('Latitude [deg]')

	vmin=0
	vmax=selected[column].max()
	divider=make_axes_locatable(ax)
	cax=divider.append_axes('bottom', size='2%', pad=.5)
	sm=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	# sm._A=[]
	cbr=plt.colorbar(sm, cax=cax, orientation='horizontal')
	cbr.ax.set_xlabel(color_axis_label,
		labelpad=10,fontsize='medium')
	# ax.set_aspect('equal','box')

	if return_fig:
		return fig

def DataColumnComparisonPlot(tracts,h3_hex,background,horizontal=True,figsize=(12,6),
	margin=.05,alpha=1,colors=color_scheme_2_1,column=None,color_axis_label=None):
	
	if horizontal:
		fig,ax=plt.subplots(1,2,figsize=figsize)
	else:
		fig,ax=plt.subplots(2,1,figsize=figsize)
	DataColumnPlot(tracts,background,ax=ax[0],margin=margin,alpha=alpha,
		colors=colors,column=column,color_axis_label=color_axis_label)
	DataColumnPlot(h3_hex,background,ax=ax[1],margin=margin,alpha=alpha,
		colors=colors,column=column,color_axis_label=color_axis_label)

	return fig

def HistogramComparisonPlot(tracts,h3_hex,horizontal=True,figsize=(12,6),data_label=None,
	colors=color_scheme_2_1,column=None,cutoff=None,bins=100,
	dist_names=dist_names,dist_labels=dist_labels):
	
	if data_label == None:
		data_label=column
	
	if horizontal:
		fig,ax=plt.subplots(1,2,figsize=figsize)
	else:
		fig,ax=plt.subplots(2,1,figsize=figsize)
	HistogramDist(tracts[column],ax=ax[0],cutoff=cutoff,bins=bins,colors=colors,data_label=data_label,
		dist_names=dist_names,dist_labels=dist_labels)
	HistogramDist(h3_hex[column],ax=ax[1],cutoff=cutoff,bins=bins,colors=colors,data_label=data_label,
		dist_names=dist_names,dist_labels=dist_labels)

	return fig

def HistogramDist(data,figsize=(8,8),cutoff=None,bins=100,colors=color_scheme_2_1,ax=None,
	data_label=None,dist_names=dist_names,dist_labels=dist_labels):

	data=data.to_numpy()
	data=data[~np.isnan(data)]

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	dist_name,dist,params,_=FitBestDist(data,bins=bins,dist_names=dist_names,dist_labels=dist_labels)

	densities,bin_edges=np.histogram(data,bins)
	integral=(densities*np.diff(bin_edges)).sum()
	bin_edges=(bin_edges+np.roll(bin_edges,-1))[:-1]/2.0
	y=dist.pdf(bin_edges,loc=params[-2],scale=params[-1],*params[:-2])*integral
	rmse=np.sqrt(((y-densities)**2).sum()/len(y))
	x=np.linspace(bin_edges.min(),bin_edges.max(),1000)
	y=dist.pdf(x,loc=params[-2],scale=params[-1],*params[:-2])*integral

	ax.set_facecolor('lightgray')
	
	out=ax.hist(data,bins=bins,rwidth=.9,color=colors[1],ec='k',label='Data')
	ax.plot(x,y,lw=7,color='k')
	ax.plot(x,y,lw=5,color=colors[0],label='Best-Fit Distribution:\n{}, RMSE={:.4f}'.format(dist_name,rmse))
	ax.grid(ls='--')
	if cutoff != None:
		ax.set_xlim([0,cutoff])
	ax.legend()
	ax.set_xlabel(data_label)
	ax.set_ylabel('Bin Size [-]')

	if return_fig:
		return fig

def Correlation(x,y):
	n=len(x)
	return (n*(x*y).sum()-x.sum()*y.sum())/np.sqrt((n*(x**2).sum()-x.sum()**2)*(n*(y**2).sum()-y.sum()**2))

def Determination(x,y):
	return Correlation(x,y)**2

def FitBestDist(data,bins=200,dist_names=dist_names,dist_labels=dist_labels):

	densities,bin_edges=np.histogram(data,bins,density=True)
	bin_edges=(bin_edges+np.roll(bin_edges,-1))[:-1]/2.0

	rmse=np.empty(len(dist_names))
	params_list=[None]*len(dist_names)

	for idx,dist_name in enumerate(dist_names):

		dist=getattr(st,dist_name)

		try:

			params=dist.fit(data)
			params_list[idx]=params

			arg=params[:-2]
			loc=params[-2]
			scale=params[-1]
			y=dist.pdf(bin_edges,loc=loc,scale=scale,*arg)

			rmse[idx]=np.sqrt(((y-densities)**2).sum()/len(y))

		except Exception as e:

			rmse[idx]=sys.maxsize

	best_dist_index=np.argmin(rmse)

	return (dist_labels[best_dist_index],getattr(st,dist_names[best_dist_index]),
		params_list[best_dist_index],np.min(rmse))