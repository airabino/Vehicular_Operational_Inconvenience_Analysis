import sys
import time
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm

#Function for importing, cleaning, and processing trips data
def LoadNHTSData(trips_file):
	#Loading in the trips data
	trips_df=pd.read_csv(trips_file)
	#Filtering out non-vehicle trips
	trips_df=trips_df[(trips_df['VEHID']>0)&(trips_df['VEHID']<12)]
	#Removing unnecessary columns
	trips_df=trips_df[(['HOUSEID','VEHID','STRTTIME','ENDTIME','TRVLCMIN','TRPMILES','TRPTRANS',
		'DWELTIME','WHYTRP1S','OBHUR','HHSTFIPS','HH_CBSA'])]
	return trips_df

#Itinerary class
class Itinerary():
	def __init__(self,vehicle_id,data):
		self.Populate(vehicle_id,data)

	def Populate(self,vehicle_id,data):
		self.vehicle_id=vehicle_id
		# self.trip_start_time=data[:,2]
		# self.trip_finish_time=data[:,3]
		self.durations=data[:,4].astype(float)
		self.distances=data[:,5].astype(float)
		self.trans_mode=data[:,6].astype(int)
		self.dwell_time=data[:,7].astype(float)
		self.dest_type=data[:,8].astype(int)
		self.dest_urb_type=data[:,9].astype(str)
		self.dest_state_fips=data[:,10].astype(str)
		self.dest_msa_fips=data[:,11].astype(str)
		self.hh_urb=self.dest_urb_type[0]
		self.hh_state=self.dest_state_fips[0]
		self.hh_msa=self.dest_msa_fips[0]


#Create Itinerary objects for every vehicle in the dataset
def CreateItineraries(trips_df):
	#Identifying unique vehicle IDs
	trips_df['vehicle_id']=trips_df['HOUSEID'].astype(str)+trips_df['VEHID'].astype(str)
	trips_df['vehicle_id'].astype(int)
	unique_vehicle_ids,unique_vehicle_id_indices=np.unique(trips_df['vehicle_id'],return_index=True)
	itineraries=np.empty(unique_vehicle_ids.shape,dtype='O')
	data=trips_df.to_numpy()
	for idx in tqdm(range(unique_vehicle_ids.shape[0])):
		itineraries[idx]=Itinerary(unique_vehicle_ids[idx],data[data[:,-1]==unique_vehicle_ids[idx]])
		# if idx==10:
		# 	break
	return itineraries

def CountItineraryLengths(itineraries):
	itinerary_lengths=np.zeros(len(itineraries))
	for idx in tqdm(range(len(itineraries))):
		itinerary_lengths[idx]=itineraries[idx].durations.shape[0]
	return itinerary_lengths

def FilterLongItineraries(itineraries,min_length):
	itinerary_lengths=CountItineraryLengths(itineraries)
	selection_indices=(itinerary_lengths>min_length)
	return itineraries[selection_indices]

def FilterByAttribute(itineraries,prop,value):
	keep=[False]*len(itineraries)
	for idx in tqdm(range(len(itineraries))):
		# print(type(getattr(itineraries[idx],prop)),getattr(itineraries[idx],prop)==value,value)
		keep[idx]=getattr(itineraries[idx],prop)==value
		# break
	return itineraries[keep]



