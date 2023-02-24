import os
import sys
import time
import smopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm
from routingpy.routers import get_router_by_name,MapboxValhalla,MapboxOSRM

default_key='pk.eyJ1IjoiYWlyYWJpbm8iLCJhIjoiY2wzZGR3cHZ3MDdpbzNqcXBxZ2RvZXV0dSJ9.5SgDnIFa2heDP27DTC0L7g'

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

def GetRouterAPI(backend='OSRM',key=default_key):
	routers=ReturnRouters(key)
	if backend=='OSRM':
		return {'api':get_router_by_name('mapbox_osrm')(api_key=routers['mapbox_osrm']['api_key']),
			'information':routers['mapbox_osrm']}
	elif ackend=='Valhalla':
		return {'api':get_router_by_name('mapbox_valhalla')(api_key=routers['mapbox_valhalla']['api_key']),
			'information':routers['mapbox_valhalla']}
	else:
		print('Please select ORSM or Valhalla')

def RouteSummary(start,finish,router_api=GetRouterAPI()):
	t0=time.time()
	route=router_api['api'].directions(profile=router_api['information']['profile'],
		locations=(start,finish))
	# print(route.__dict__)
	# print(route._raw['routes'][0])
	# distances=np.array([route._raw['routes'][0]['distance']])
	# durations=np.array([route._raw['routes'][0]['distance']])
	# speeds=np.array(route._raw['routes'][0]['distance'])
	distances,durations,speeds=PullRouteInfo(route)
	route_full={'locations':route._geometry,'distances':distances,'durations':durations,
		'speeds':distances/durations}
	return route_full

def Route(start,finish,router_api=GetRouterAPI()):
	t0=time.time()
	route=router_api['api'].directions(profile=router_api['information']['profile'],locations=(start,finish))
	# print(route._geometry,len(route._geometry))
	# route_full=router_api['api'].matrix(profile=router_api['information']['profile'],locations=route._geometry)
	# print(route_full)
	location_arrays=ChopArray(np.array(route._geometry),25)
	routes=[None]*len(location_arrays)
	for i in range(len(location_arrays)):
		route=router_api['api'].directions(profile=router_api['information']['profile'],
			locations=location_arrays[i])
		distances,durations,speeds=PullRouteInfo(route)
		routes[i]={'locations':location_arrays[i],'distances':distances,'durations':durations,
			'speeds':speeds}
	# print(routes)
	route_full=MergeRoutes(routes)
	route_full['run_time']=np.array([time.time()-t0])
	return route_full
	# print('a',route_full)

	# route_full=router_api['api'].directions(profile=router_api['information']['profile'],
	# 	locations=route._geometry[:25])
	# distances,durations,speeds=PullRouteInfo(route_full)
	# run_time=time.time()-t0
	# return {'route_object':route_full,'distances':distances,'durations':durations,
	# 	'speeds':speeds,'run_time':run_time}

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

def SmopyPlot(path,pos0,pos1):
	linepath=np.array([*path])
	
	bounds=(np.min(linepath[:,1]),np.min(linepath[:,0]),
		np.max(linepath[:,1]),np.max(linepath[:,0]))

	m = smopy.Map(bounds, z=15, margin=.1)
	x, y = m.to_pixels(linepath[:, 1], linepath[:, 0])
	
	ax = m.show_mpl(figsize=(8, 8))
	# Mark our two positions.
	ax.plot(x[0], y[0], 'ob', ms=20)
	ax.plot(x[-1], y[-1], 'or', ms=20)
	# Plot the itinerary.
	ax.plot(x, y, '-k', lw=3)
	return x,y
