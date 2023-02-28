import sys
import time
import argparse
import numpy as np
import numpy.random as rand
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
from datetime import datetime
from .process_nhts_data import FilterByAttribute, Itinerary

class BEV_Compiled():
	"""
	BEV object simulates optimal charging behavior for a Battery Electric Vehicle on a set itinerary with an assumed charging grid.
	
	The vehicle is defined by its battery size and electric consumption at low, medium, and high speeds.
	The grid is defined by the charging rate and price of available chargers at home, en route, and at destinations. Every
	Destination is assigned a charger but the rate and price can be set to zero which will function as a non-charger destination.
	
	The optimal charging behavior is calculated using Dynamic Programming (DP). The cost function is a combination of time added
	to itinerary and cost of charging. The relative values of time added and cost of charge are determined by tuning parameters
	
	Data required to run the simulation is shown in the following DataFrame:
	df['tripDist_mi'] - distance of trip preceeding park [miles]
	df['tripTime'] - time to complete trip preceeding park [s]
	df['dwellTime'] - duration of park [s]
	df['locationType'] - 'H' for home, 'W' for work, and 'O' for other
	
	Required inputs:
	df - see above
	Dest_Charger_Likelihood - likelihood that a given destination will have an available charger
	Consumption_City/Mixed/Highway - Electric consumption of the vehicle at various speeds [J/m]
	Battery_Capacity - Maximum energy which can be stroed in the battery [J]
	
	The DP method is contained in two functions:
	 - BEV.Optimize() generates the optimal control matrices for each control and the cost-to-go matrix
	 - BEV.Evaluate() uses the optimal control matrices to find the optimal controls
	There is one state: X - battery SOC
	There are two controls: U1 - time spent charging at destination (or home)
							U2 - time spent charging en route
	Disallowable states are any SOCs which are outside the set bounds
	Disallowable controls are any U1 which exceeds the dwell time from the itinerary
	
	(All units are SI unless specified otherwise)
	"""
	def __init__(self,itin,
		Dest_Charger_Likelihood=.1,
		Consumption_City=385.2,
		Consumption_Mixed=478.8,
		Consumption_Highway=586.8,
		Battery_Capacity=82*1000*3600,
		Starting_SOC=.5,
		Final_SOC=.5,
		Plug_In_Penalty=60,
		Dest_Charger_P_AC=12100,
		En_Route_Charger_P_AC=150000,
		Home_Charger_P_AC=12100,
		Work_Charger_P_AC=12100,
		Nu_AC_DC=.88,
		SOC_Max=1,
		Range_Min=25000,
		SOC_Quanta=50,
		LVL2_Charging_Quanta=2,
		DCFC_Charging_Quanta=10,
		En_Route_Charging_Max=7200,
		SpeedThresholds=np.array([35,65])*0.44704,
		En_Route_Charging_Penalty=15*60,
		Final_SOC_Penalty=1e10,
		Out_Of_Bounds_Penalty=1e50,
		tiles=7):
		
		self.Starting_SOC=Starting_SOC #[dim] BEV's SOC at start of itinerary
		self.Final_SOC=Final_SOC #[dim] BEV's SOC at end of itinerary
		self.Plug_In_Penalty=Plug_In_Penalty #[min] Equivalent inconvenience penalty added to all charge events
		#to account for plug-in time and to account for inconvenience associated with pluging in
		self.Dest_Charger_Likelihood=Dest_Charger_Likelihood #[dim] Likelihood that a given destination will have an available charger
		self.Dest_Charger_P_AC=Dest_Charger_P_AC #[W] AC Power delivered by destination chargers
		self.En_Route_Charger_P_AC=En_Route_Charger_P_AC #[W] AC Power delivered by en route chargers
		self.Home_Charger_P_AC=Home_Charger_P_AC #[W] AC Power delivered by home charger
		self.Work_Charger_P_AC=Work_Charger_P_AC #[W] AC Power delivered by home charger
		self.SpeedThresholds=SpeedThresholds #[m/s] Upper average speed thresholds for city and mixed driving
		self.Consumption_City=Consumption_City #[J/m] Electric consumption at urban speeds
		self.Consumption_Mixed=Consumption_Mixed #[J/m] Electric consumption at medium speeds
		self.Consumption_Highway=Consumption_Highway #[J/m] Electric consumption at highway speeds
		self.Battery_Capacity=Battery_Capacity #[J] Maximum energy which can be stored in the battery
		self.Nu_AC_DC=Nu_AC_DC #[dim] itinicle AC/DC converter efficiency
		self.SOC_Max=SOC_Max #[dim] Highest allowable SOC
		self.Range_Min=Range_Min #[m] Lowest allowable remaining range
		self.SOC_Min=self.Range_Min*self.Consumption_Mixed/self.Battery_Capacity #[dim] Lowest allowable SOC
		self.SOC_Quanta=SOC_Quanta #[dim] number of SOC quanta for optimization
		self.X=np.linspace(0,1,SOC_Quanta) #[dim] (n_X,) array of discreet SOC values for optimization
		self.LVL2_Charging_Quanta=LVL2_Charging_Quanta #[dim] number of charge time quanta for optimization
		self.DCFC_Charging_Quanta=DCFC_Charging_Quanta #[dim] number of charge time quanta for optimization
		self.U1=np.linspace(0,1,LVL2_Charging_Quanta) #[s] (n_U1,) array of discreet charging time values for optimization
		self.En_Route_Charging_Max=En_Route_Charging_Max #[s] Maximum time that can be spent charging en route
		self.U2=np.linspace(0,1,DCFC_Charging_Quanta) #[s] (n_U2,) array of discreet charging time values for optimization
		self.En_Route_Charging_Penalty=En_Route_Charging_Penalty #[s] Additional time required to travel to en route charging station
		self.Final_SOC_Penalty=Final_SOC_Penalty
		self.Out_Of_Bounds_Penalty=Out_Of_Bounds_Penalty
		self.tiles=tiles #Number of times the itinerary is repeated
		
		self.ItineraryArrays(itin)

	def ItineraryArrays(self,itin):
		# print(itin.dwell_time)
		#Fixing any non-real dwells
		dwell_time=itin.dwell_time.copy()
		durations=itin.durations.copy()
		dwell_time[dwell_time<0]=dwell_time[dwell_time>=0].mean()
		# print(dwell_time)
		# print(durations)
		#Padding with overnight dwell
		sum_of_times=durations.sum()+dwell_time[:-1].sum()
		# print(sum_of_times)
		if sum_of_times>=1440:
			ratio=1440/sum_of_times
			dwell_time*=ratio
			durations*=ratio
		else:
			final_dwell=1440-durations.sum()-dwell_time[:-1].sum()
			dwell_time[-1]=final_dwell
		# print(dwell_time)
		#Populates itinerary arrays
		self.Trip_Distances=np.tile(itin.distances,self.tiles)*1609.34 #[m] Distances of trips preceeding parks
		self.Trip_Times=np.tile(durations,self.tiles)*60 #[s] Durations of trips preceeding parks
		self.Trip_Mean_Speeds=self.Trip_Distances/self.Trip_Times #[m/s] Speeds of trips preceeding parks
		self.Parks=np.tile(dwell_time,self.tiles)*60 #[s] Durations of parks
		# self.Parks[self.Parks<0]=self.Parks[self.Parks>=0].mean()
		self.location_types=np.tile(itin.dest_type,self.tiles) #Destination type (see NHTS 2017 codebook: WHYTRIP1S)
		self.isHome=self.location_types==1 #[s] Boolean for home locations
		self.isWork=self.location_types==10 #[s] Boolean for home locations
		self.isOther=(~self.isHome&~self.isWork)
		self.Dest_Charger_P_AC_Array=np.array([self.Dest_Charger_P_AC]*len(self.Parks)) #Array of AC charging powers for all parks
		ChargerSelection=rand.rand(len(self.Dest_Charger_P_AC_Array)) #Assigning charger probability to each visited location in whole itinerary
		NoCharger=ChargerSelection>=self.Dest_Charger_Likelihood #Selecting non-charger locations
		self.Dest_Charger_P_AC_Array[NoCharger]=0 #Removing chargers from non-charger locations
		#Adding home chargers to home destinations
		self.Dest_Charger_P_AC_Array[self.isHome]=self.Home_Charger_P_AC
		#Adding work chargers to work destinations
		self.Dest_Charger_P_AC_Array[self.isWork]=self.Work_Charger_P_AC

	def Optimize(self):

		soc_vals=self.X
		u1_vals=self.U1
		u2_vals=self.U2
		soc_grid,u1_grid,u2_grid=np.meshgrid(soc_vals,u1_vals,u2_vals,indexing='ij')

		#Pre-calculating discharge events for each trip
		discharge_events=np.ones(len(self.Trip_Distances))*self.Consumption_Mixed
		discharge_events[self.Trip_Mean_Speeds<self.SpeedThresholds[0]]=self.Consumption_City
		discharge_events[self.Trip_Mean_Speeds>=self.SpeedThresholds[1]]=self.Consumption_Highway
		discharge_events=self.Trip_Distances*discharge_events/self.Battery_Capacity

		optimal_u1,optimal_u2,cost_to_go=OCS_OptimizeComp(
			self.Parks,self.SOC_Min,self.SOC_Max,
			soc_vals,soc_grid,u1_vals,u1_grid,u2_vals,u2_grid,self.En_Route_Charging_Max,
			self.Dest_Charger_P_AC_Array,self.En_Route_Charger_P_AC,
			self.En_Route_Charging_Penalty,self.isOther,self.Plug_In_Penalty,
			discharge_events,self.Final_SOC,self.Final_SOC_Penalty,
			self.Out_Of_Bounds_Penalty,self.Battery_Capacity,self.Nu_AC_DC)

		return [optimal_u1,optimal_u2],cost_to_go

	def Evaluate(self,optimal_controls):

		soc_vals=self.X
		u1_vals=self.U1
		u2_vals=self.U2

		#Pre-calculating discharge events for each trip
		discharge_events=np.ones(len(self.Trip_Distances))*self.Consumption_Mixed
		discharge_events[self.Trip_Mean_Speeds<self.SpeedThresholds[0]]=self.Consumption_City
		discharge_events[self.Trip_Mean_Speeds>=self.SpeedThresholds[1]]=self.Consumption_Highway
		discharge_events=self.Trip_Distances*discharge_events/self.Battery_Capacity

		optimal_u1_trace,optimal_u2_trace,soc_trace,sicd=OCS_EvaluateComp(
			optimal_controls[0],optimal_controls[1],self.Starting_SOC,
			self.Trip_Distances,self.Parks,self.En_Route_Charging_Max,soc_vals,
			self.Dest_Charger_P_AC_Array,self.En_Route_Charger_P_AC,
			self.En_Route_Charging_Penalty,self.isOther,self.Plug_In_Penalty,
			discharge_events,self.Battery_Capacity,self.Nu_AC_DC)

		return [optimal_u1_trace,optimal_u2_trace],soc_trace,sicd



@jit(nopython=True,cache=True)
def OCS_OptimizeComp(dwell_times,soc_lb,soc_ub,
	soc_vals,soc_grid,u1_vals,u1_grid,u2_vals,u2_grid,u2_max,
	location_charge_rates,en_route_charge_rate,
	en_route_charging_penalty,is_other,plug_in_penalty,
	discharge_events,final_soc,final_soc_penalty,
	out_of_bounds_penalty,battery_capacity,nu_ac_dc):

	#Length of the trips vector
	n=len(dwell_times)

	#Initializing loop variables
	cost_to_go=np.empty((n,len(soc_vals)))
	cost_to_go[:]=np.nan
	optimal_u1=np.empty((n,len(soc_vals)))
	optimal_u1[:]=np.nan
	optimal_u2=np.empty((n,len(soc_vals)))
	optimal_u2[:]=np.nan

	#Main loop
	for idx in np.arange(n-1,-1,-1):

		#Initializing state and control
		soc=soc_grid.copy()
		u1=u1_grid.copy()
		u2=u2_grid.copy()

		#Assigning charging rate for current time-step
		u1*=dwell_times[idx] #Control for location charging is the charging time
		u2*=u2_max #Control for en-route charging is charge time

		#Updating state
		soc-=discharge_events[idx]

		#Initializing cost array
		cost=np.zeros(soc_grid.shape)

		#Applying location charging control
		if location_charge_rates[idx]>0:
			soc+=CalculateArrayCharge_AC(
					location_charge_rates[idx],soc,u1,nu_ac_dc,battery_capacity)
			if is_other[idx]:
				cost+=plug_in_penalty

		#Applying en-route charging control
		if en_route_charge_rate>0:
			soc+=CalculateArrayCharge_DC(
						en_route_charge_rate,soc,u2,nu_ac_dc,battery_capacity)
		for idx1 in range(soc_vals.size):
			for idx2 in range(u1_vals.size):
				for idx3 in range(u2_vals.size):
					if u2[idx1,idx2,idx3]>0:
						cost[idx1,idx2,idx3]+=u2[idx1,idx2,idx3]+(en_route_charging_penalty+
							plug_in_penalty)

		#Applying boundary costs
		for idx1 in range(soc_vals.size):
			for idx2 in range(u1_vals.size):
				for idx3 in range(u2_vals.size):
					if soc[idx1,idx2,idx3]>soc_ub:
						cost[idx1,idx2,idx3]+=out_of_bounds_penalty
					elif soc[idx1,idx2,idx3]<soc_lb:
						cost[idx1,idx2,idx3]+=out_of_bounds_penalty

		if idx==n-1:
			#Applying the final-state penalty
			diff=soc-final_soc
			penalty=final_soc_penalty*(diff)**2*final_soc_penalty
			# penalty[diff>0]=0
			for idx1 in range(soc_vals.size):
				for idx2 in range(u1_vals.size):
					for idx3 in range(u2_vals.size):
						if diff[idx1,idx2,idx3]>0:
							penalty[idx1,idx2,idx3]=0
			cost+=penalty
		else:
			#Adding cost-to-go
			cost+=np.interp(soc,soc_vals,cost_to_go[idx+1])

		#Finding optimal controls and cost-to-go - Optimal controls for each starting SOC are the controls which result in
		#the lowest cost at that SOC. Cost-to-go is the cost of the optimal controls at each starting SOC
		for idx1 in range(soc_vals.size):
			mins=np.zeros(u1_vals.size) #minimum for each row
			min_inds=np.zeros(u1_vals.size) #minimum index for each row
			for idx2 in range(u1_vals.size):
				mins[idx2]=np.min(cost[idx1,idx2,:]) #minimum for each row
				min_inds[idx2]=np.argmin(cost[idx1,idx2,:])
			min_row=np.argmin(mins) #row of minimum
			min_col=min_inds[int(min_row)] #column of minimum
			optimal_u1[idx,idx1]=u1_vals[int(min_row)]
			optimal_u2[idx,idx1]=u2_vals[int(min_col)]
			cost_to_go[idx,idx1]=cost[idx1,int(min_row),int(min_col)]

	return optimal_u1,optimal_u2,cost_to_go

@jit(nopython=True,cache=True)
def OCS_EvaluateComp(optimal_u1,optimal_u2,initial_soc,
	trip_distances,dwell_times,u2_max,soc_vals,
	location_charge_rates,en_route_charge_rate,
	en_route_charging_penalty,is_other,plug_in_penalty,
	discharge_events,battery_capacity,nu_ac_dc):

	#Length of the time vector
	n=len(dwell_times)

	#Initializing loop variables
	optimal_u1_trace=np.empty(n)
	optimal_u2_trace=np.empty(n)

	soc_trace=np.empty(n+1)
	soc_trace[0]=initial_soc

	soc=initial_soc
	dedicated_energizing_time=0
	

	#Main loop
	soc=initial_soc
	for idx in np.arange(0,n,1):

		#Updating state
		soc-=discharge_events[idx]

		#Applying location charging control
		optimal_u1_trace[idx]=np.interp(soc,soc_vals,optimal_u1[idx])*dwell_times[idx]

		if location_charge_rates[idx]>0:
			if optimal_u1_trace[idx]>0:
				soc+=CalculateCharge_AC(
						location_charge_rates[idx],soc,optimal_u1_trace[idx],
						nu_ac_dc,battery_capacity)
				if is_other[idx]:
					dedicated_energizing_time+=plug_in_penalty


		#Applying en-route charging control
		optimal_u2_trace[idx]=np.interp(soc,soc_vals,optimal_u2[idx])*u2_max

		if en_route_charge_rate>0:
			soc+=CalculateCharge_DC(
						en_route_charge_rate,soc,optimal_u2_trace[idx],nu_ac_dc,battery_capacity)
			if optimal_u2_trace[idx]>0:
				dedicated_energizing_time+=optimal_u2_trace[idx]+(en_route_charging_penalty+
					plug_in_penalty)

		soc_trace[idx+1]=soc

	sicd=(dedicated_energizing_time/60)/(trip_distances.sum()/1000)

	return optimal_u1_trace,optimal_u2_trace,soc_trace,sicd

@jit(nopython=True,cache=True)
def CalculateCharge_DC(P_AC,SOC,td_charge,Nu_AC_DC,Battery_Capacity):
	P_DC=P_AC*Nu_AC_DC #[W] DC power received from charger after accounting for AC/DC converter loss
	Lambda_Charging=P_DC/Battery_Capacity/.2 #Exponential charging factor
	t_80=(.8-SOC)*Battery_Capacity/P_DC
	if td_charge<=t_80:
		Delta_SOC=P_DC/Battery_Capacity*td_charge
	else:
		Delta_SOC=.8-SOC+.2*(1-np.exp(-Lambda_Charging*(td_charge-t_80)))
	return Delta_SOC

@jit(nopython=True,cache=True)
def CalculateArrayCharge_DC(P_AC,SOC,td_charge,Nu_AC_DC,Battery_Capacity):
	#Calcualting the SOC gained from a charging event of duration td_charge
	#The CC-CV curve is an 80/20 relationship where the charging is linear for the first 80%
	#and tails off for the last 20% approaching 100% SOC at t=infiniti
	Delta_SOC=np.zeros(SOC.shape) #Initializing the SOC delta vector
	for idx1 in range(SOC.shape[0]):
		for idx2 in range(SOC.shape[1]):
			for idx3 in range(SOC.shape[2]):
				Delta_SOC[idx1,idx2,idx3]=CalculateCharge_DC(
					P_AC,SOC[idx1,idx2,idx3],td_charge[idx1,idx2,idx3],Nu_AC_DC,Battery_Capacity)
	return Delta_SOC

@jit(nopython=True,cache=True)
def CalculateCharge_AC(P_AC,SOC,td_charge,Nu_AC_DC,Battery_Capacity):
	P_DC=P_AC*Nu_AC_DC #[W] DC power received from charger after accounting for AC/DC converter loss
	t_100=(1-SOC)*Battery_Capacity/P_DC
	if td_charge<=t_100:
		Delta_SOC=P_DC/Battery_Capacity*td_charge
	else:
		Delta_SOC=1.-SOC
	return Delta_SOC

@jit(nopython=True,cache=True)
def CalculateArrayCharge_AC(P_AC,SOC,td_charge,Nu_AC_DC,Battery_Capacity):
	#Calcualting the SOC gained from a charging event of duration td_charge
	#The CC-CV curve is an 80/20 relationship where the charging is linear for the first 80%
	#and tails off for the last 20% approaching 100% SOC at t=infiniti
	Delta_SOC=np.zeros(SOC.shape) #Initializing the SOC delta vector
	for idx1 in range(SOC.shape[0]):
		for idx2 in range(SOC.shape[1]):
			for idx3 in range(SOC.shape[2]):
				Delta_SOC[idx1,idx2,idx3]=CalculateCharge_AC(
					P_AC,SOC[idx1,idx2,idx3],td_charge[idx1,idx2,idx3],Nu_AC_DC,Battery_Capacity)
	return Delta_SOC

class ICV():
	"""
	ICV object simulates optimal charging behavior for a Internal Combustion Vehicle on a set itinerary with an assumed charging grid.
	
	ICV functions similarly to BEV but uses fueling rather than charging. All fueling is done in the same amount of time with regards to the simulation

	"""
	def __init__(self,veh,
				Consumption_City=2599.481,
				Consumption_Mixed=2355.779,
				Consumption_Highway=2094.026,
				Fuel_Tank_Capacity=528*1000*3600,
				Starting_SOF=.5,
				Final_SOF=.5,
				SOF_Max=1,
				Range_Min=25000,
				SOF_Quanta=20,
				SpeedThresholds=np.array([35,65])*0.44704,
				Fueling_Time_Penalty=300,
				Fueling_Rate=121300000*7/60):
		
		self.Starting_SOF=Starting_SOF #[dim] ICV's SOF at start of itinerary
		self.Final_SOF=Final_SOF #[dim] ICV's SOF at end of itinerary
		self.SpeedThresholds=SpeedThresholds #[m/s] Upper average speed thresholds for city and mixed driving
		self.Consumption_City=Consumption_City #[J/m] Fuel consumption at urban speeds
		self.Consumption_Mixed=Consumption_Mixed #[J/m] Fuel consumption at medium speeds
		self.Consumption_Highway=Consumption_Highway #[J/m] Fuel consumption at highway speeds
		self.Fuel_Tank_Capacity=Fuel_Tank_Capacity #[J] Maximum energy which can be stored in the fuel tank
		self.SOF_Max=SOF_Max #[dim] Maximum allowable SOF
		self.Range_Min=Range_Min #[m] Lowest allowable remaining range
		self.SOF_Min=self.Range_Min*self.Consumption_Mixed/self.Fuel_Tank_Capacity #[dim] Lowest allowable SOF
		self.X=np.linspace(0,self.SOF_Max,SOF_Quanta) #[dim] (n_X,) array of discreet SOF values for optimization
		self.U=np.array([0,1]) #[s] (n_U,) array of discreet SOF deltas for optimization
		self.Fueling_Time_Penalty=Fueling_Time_Penalty #[s] time penalty for traveling to and re-fueling at a re-feuling station
		self.Fueling_Rate=Fueling_Rate #[W] rate of energization while fueling
		
		self.ItineraryArrays(veh)
	
	def ItineraryArrays(self,veh):
		#Populates itinerary arrays
		vdf=veh.df.head(100) #Need a better way to down-select the itinerary
		self.Trip_Distances=vdf['tripDist_mi'].to_numpy()*1609.34 #[m] Distances of trips preceeding parks
		self.Trip_Times=vdf['tripTime'].to_numpy() #[s] Durations of trips preceeding parks
		self.Trip_Mean_Speeds=self.Trip_Distances/self.Trip_Times #[m/s] Speeds of trips preceeding parks
		self.Parks=vdf['dwellTime'].to_numpy() #[s] Durations of parks
	
	def Optimize(self):
		#the optimize step is the first step in the DP solver in which optimal control matrices are created. The optimize step involves
		#backwards iteration through the exogenous input tracres while the optimal control matrices are populated.
		N=len(self.Parks) #Length of itinerary
		#Initializing loop variables
		Cost_to_Go=np.empty((N,len(self.X)))
		Cost_to_Go[:]=np.nan
		Optimal_U=np.empty((N,len(self.X)))
		Optimal_U[:]=np.nan
		#Pre-calculating discharge events for each trip
		FCRate=np.ones(len(self.Trip_Distances))*self.Consumption_Mixed
		FCRate[self.Trip_Mean_Speeds<self.SpeedThresholds[0]]=self.Consumption_City
		FCRate[self.Trip_Mean_Speeds>=self.SpeedThresholds[1]]=self.Consumption_Highway
		Trip_SOF_Deltas=self.Trip_Distances*FCRate/self.Fuel_Tank_Capacity
		#Main loop (backwards iteration)
		for k in np.arange(N-1,-1,-1):
			#Initializing state and controls arrays
			SOF,Fuel=np.meshgrid(self.X,self.U,indexing='ij') #(n_X,n_U) arrays of values for state and control
			#every combination of state and control is evaluated
			# print(SOF,Fuel)
			#Discharging
			SOF-=Trip_SOF_Deltas[k] #Applying discharge events to the SOF
			
			#Initializing cost array
			Cost=np.zeros((len(self.X),len(self.U))) #Array of same dimensions as SOF/Fuel which will store the
			#cost of each combination of state and control
			
			#ICVs can only fuel en-route.
			SOF[Fuel==1]=self.SOF_Max
			Cost+=((Fuel>0)*self.Fueling_Time_Penalty)
			
			#Enforcing constraints - disallowable combinations of state and controls are assigned huge costs - these will later
			#be used to identify which states are disallowable. Common practice is to assign NaN of Inf cost to disallowable
			#combinations here but this leads to complications later in the code so a very high number works better
			Cost[SOF<self.SOF_Min]=1e50 #SOF too low
			Cost[SOF>self.SOF_Max]=1e50 #SOF too high
			
			#Penalty and Cost to Go - the penalty for failing to meet the final SOF constraint is applied at step N-1
			#(the last step in the itinerary but the first processed in the loop). for all other stpes the cost-to-go is applied.
			#Cost-to-go is the cost to go from step k+1 to step k+2 (k is the current step). Cost-to-go is the "memory" element
			#of the method which ties the steps together
			if k==N-1:
				#Assinging penalty for failing to meet the final SOF constraints
				diff=self.Final_SOF-SOF
				Penalty=diff**2*1e10
				Penalty[diff<0]=0
				# print(Penalty)
				Cost+=Penalty
			else:
				#Assigning the cost-to-go
				Cost+=np.interp(SOF,self.X,Cost_to_Go[k+1])
			 
			#Finding optimal controls and cost-to-go - Optimal controls for each starting SOF are the controls which result in
			#the lowest cost at that SOC. Cost-to-go is the cost of the optimal controls at each starting SOC
			mins=np.min(Cost,axis=1) #minimum for each row
			
			min_inds=np.argmin(Cost,axis=1) #minimum axis for each row
			Optimal_U[k]=self.U[min_inds]
			Cost_to_Go[k]=mins
			
		#Identifying disallowable optimal controls - optimal controls resulting from disallowable combinations are set to -1
		#so that they can be easily filtered out (all others will be >=0). Disallowable combinations can be "optimal" if all
		#combinations are disallowable for a given SOF and step
		Optimal_U[Cost_to_Go>=1e50]=-1
		
		#Outputs of BEV.Optimize() are a list of the optimal control matrices for each control and the cost-to-go matrix
		return Optimal_U,Cost_to_Go
	
	def Evaluate(self,Optimal_Control):
		#The evaluate step is the second step in the DP method in which optimal controls and states are found for each
		#time-step in the itinerary. This is accomplished through forward iteration. Unlike the optimize step which 
		#considers all possible SOF values, the evaluate step follows only one as it iterates forward using the optimal
		#controls for the current SOF and step.
		#Initializations
		N=len(self.Parks)
		Optimal_U=np.empty(N)
		Optimal_U[:]=np.nan
		Fueling_Time_Penalty=np.zeros(N)
		SOF_Trace=np.empty(N+1)
		SOF_Trace[0]=self.Starting_SOF
		SOF=self.Starting_SOF
		FCRate=np.ones(len(self.Trip_Distances))*self.Consumption_Mixed
		FCRate[self.Trip_Mean_Speeds<self.SpeedThresholds[0]]=self.Consumption_City
		FCRate[self.Trip_Mean_Speeds>=self.SpeedThresholds[1]]=self.Consumption_Highway
		Trip_SOF_Deltas=self.Trip_Distances*FCRate/self.Fuel_Tank_Capacity 
		#Main loop (forwards iteration)
		for k in np.arange(0,N,1):
			#Discharging
			SOF-=Trip_SOF_Deltas[k]
			#Charging - the optimal, admissable controls are selected for the current step based on current SOF
			admissable=Optimal_Control[k]>=0
			Optimal_U[k]=(np.around(np.interp(SOF,self.X[admissable],Optimal_Control[k][admissable]))*
				(self.SOF_Max-SOF)*self.Fuel_Tank_Capacity/self.Fueling_Rate)
			# print((Optimal_U[k]>0)*self.SOF_Max)
			SOF+=(Optimal_U[k]>0)*(self.SOF_Max-SOF)
			SOF_Trace[k+1]=SOF

		Dedicated_Energizing_Time=(np.zeros(N)+
			(Optimal_U>sys.float_info.epsilon)*(self.Fueling_Time_Penalty)+
			Optimal_U*(Optimal_U>sys.float_info.epsilon))

		# SIC=(Dedicated_Energizing_Time.sum()/60)/N
		SICD=(Dedicated_Energizing_Time.sum()/60)/(self.Trip_Distances.sum()/1000)


		return Optimal_U,SOF_Trace,SICD