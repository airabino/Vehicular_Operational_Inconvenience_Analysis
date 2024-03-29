import os
import sys
import time
import smopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm
from routingpy.routers import get_router_by_name

def MergeRoutes(routes):
	route1=routes[0]
	for key in route1.keys():
		route1[key]=np.concatenate([route[key] for route in routes])
	return route1

def ChopArray(array,size):
	out=[]
	start=0
	for i in range(array.shape[0]//size):
		out.append(array[start:start+size])
		start+=size
	if array.shape[0]%size>0:
		out.append(array[start:start+array.shape[0]%size])
	return out

def ReturnRouters(key):
	return {'mapbox_osrm': {'api_key': key, 'display_name': 'MapBox (OSRM)','profile': 'driving'},
	'mapbox_valhalla': {'api_key': key,'display_name': 'MapBox (Valhalla)','profile': 'auto'}}

def GetRouter(key,backend='OSRM'):
	routers=ReturnRouters(key)
	if backend=='OSRM':
		return {'api':get_router_by_name('mapbox_osrm')(api_key=routers['mapbox_osrm']['api_key']),
			'information':routers['mapbox_osrm']}
	elif ackend=='Valhalla':
		return {'api':get_router_by_name('mapbox_valhalla')(api_key=routers['mapbox_valhalla']['api_key']),
			'information':routers['mapbox_valhalla']}
	else:
		print('Please select ORSM or Valhalla')

def RouteSummary(router,start,finish):
	t0=time.time()
	route=router['api'].directions(profile=router['information']['profile'],
		locations=(start,finish))
	distances,durations,speeds=PullRouteInfo(route)
	route_full={'locations':route._geometry,'distances':distances,'durations':durations,
		'speeds':distances/durations}
	return route_full

def Route(router,start,finish):
	t0=time.time()
	route=router['api'].directions(profile=router['information']['profile'],locations=(start,finish))
	location_arrays=ChopArray(np.array(route._geometry),25)
	routes=[None]*len(location_arrays)
	for i in range(len(location_arrays)):
		route=router['api'].directions(profile=router['information']['profile'],
			locations=location_arrays[i])
		distances,durations,speeds=PullRouteInfo(route)
		routes[i]={'locations':location_arrays[i],'distances':distances,'durations':durations,
			'speeds':speeds}
	route_full=MergeRoutes(routes)
	route_full['run_time']=np.array([time.time()-t0])
	return route_full

def PullRouteInfo(route):
	route_legs=route._raw['routes'][0]['legs']
	nrl=len(route_legs)
	distances=np.empty(nrl)
	durations=np.empty(nrl)
	for idx in range(nrl):
		distances[idx]=route_legs[idx]['distance']
		durations[idx]=route_legs[idx]['duration']
	valid=((distances!=0)&(durations!=0))
	return distances[valid],durations[valid],distances[valid]/durations[valid]
