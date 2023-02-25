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

	if return_fig:
		return fig

def TractsHexComparisonPlot(tracts,h3_hex,background,figsize=(12,6),margin=.05,alpha=1,
	colors=color_scheme_2_1):
	
	fig,ax=plt.subplots(1,2,figsize=(12,6))
	SelectionPlot(tracts,background,ax=ax[0],margin=margin,alpha=alpha,
		colors=colors)
	SelectionPlot(h3_hex,background,ax=ax[1],margin=margin,alpha=alpha,
		colors=colors)

	return fig

def TractsHexAreaHistogram(tracts,h3_hex,figsize=(8,8),cutoff=2e4,bins=100,colors=color_scheme_2_1):

	fig,ax=plt.subplots(figsize=(8,8))

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