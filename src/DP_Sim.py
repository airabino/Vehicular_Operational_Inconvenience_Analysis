import sys
import time
import numpy as np
import numpy.random as rand
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

def EVTracePlot1(bev,dedicated_energizing_time,optimal_control,soc_trace,max_dwells_disp=100):
	selection=[0,len(soc_trace)-1]
	if selection[1]>max_dwells_disp:
		selection[1]=max_dwells_disp
	indices=np.arange(selection[0],selection[1],1)
	indices1=np.arange(selection[0],selection[1]+1,1)
	isHomeCharge=(optimal_control[0][indices]>0)&(bev.isHome[indices])
	isWorkCharge=(optimal_control[0][indices]>0)&(bev.isWork[indices])
	isDestCharge=(optimal_control[0][indices]>0)&(bev.isOther[indices])
	isEnRouteCharge=optimal_control[1][indices]>0
	fig=plt.figure(figsize=(8,12))
	plt.subplot(411)
	plt.plot(soc_trace[indices1],color='b',linewidth=2)
	plt.plot([0,len(bev.Parks[indices])],[0,0],color='k',linestyle='--')
	plt.plot([0,len(bev.Parks[indices])],[1,1],color='k',linestyle='--')
	plt.grid()
	plt.ylabel('SOC [dim]')
	plt.subplot(412)
	plt.bar(indices,optimal_control[0][indices]*isDestCharge/3600,color='g')
	plt.bar(indices,optimal_control[0][indices]*isHomeCharge/3600,color='k')
	plt.bar(indices,optimal_control[0][indices]*isWorkCharge/3600,color='b')
	plt.bar(indices,optimal_control[1][indices]*isEnRouteCharge/3600,color='r')
	plt.legend(['Destination','Home','Work','En Route'])
	plt.grid()
	plt.ylabel('Energizing Time [h]')
	plt.subplot(413)
	plt.bar(indices,dedicated_energizing_time[indices]*isDestCharge/3600,color='g')
	plt.bar(indices,dedicated_energizing_time[indices]*isHomeCharge/3600,color='k')
	plt.bar(indices,dedicated_energizing_time[indices]*isWorkCharge/3600,color='b')
	plt.bar(indices,dedicated_energizing_time[indices]*isEnRouteCharge/3600,color='r')
	plt.legend(['Destination','Home','Work','En Route'])
	plt.grid()
	plt.ylabel('Dedicated Energizing Time [h]')
	plt.subplot(414)
	plt.bar(np.arange(0,len(bev.Trip_Distances[indices]),1),bev.Trip_Distances[indices]/1000,color='b')
	plt.grid()
	plt.xlabel('Trip/Park Event')
	plt.ylabel('Trip Distance [km]')
	return fig

def EVTracePlot(bev,dedicated_energizing_time,optimal_control,soc_trace,max_dwells_disp=100):
	selection=[0,len(soc_trace)-1]
	if selection[1]>max_dwells_disp:
		selection[1]=max_dwells_disp
	indices=np.arange(selection[0],selection[1],1)
	indices1=np.arange(selection[0],selection[1]+1,1)
	isHomeCharge=(optimal_control[0][indices]>0)&(bev.isHome[indices])
	isWorkCharge=(optimal_control[0][indices]>0)&(bev.isWork[indices])
	isDestCharge=(optimal_control[0][indices]>0)&(bev.isOther[indices])
	isEnRouteCharge=optimal_control[1][indices]>0
	fig=plt.figure(figsize=(8,8))
	plt.subplot(311)
	plt.plot(soc_trace[indices1],color='b',linewidth=2)
	plt.plot([0,len(bev.Parks[indices])],[0,0],color='k',linestyle='--')
	plt.plot([0,len(bev.Parks[indices])],[1,1],color='k',linestyle='--')
	plt.grid()
	plt.ylabel('SOC [dim]')
	plt.subplot(312)
	plt.bar(indices,optimal_control[0][indices]*isDestCharge/3600,color='g')
	plt.bar(indices,optimal_control[0][indices]*isHomeCharge/3600,color='k')
	plt.bar(indices,optimal_control[0][indices]*isWorkCharge/3600,color='b')
	plt.bar(indices,optimal_control[1][indices]*isEnRouteCharge/3600,color='r')
	plt.legend(['Destination','Home','Work','En Route'])
	plt.grid()
	plt.ylabel('Energizing Time [h]')
	plt.subplot(313)
	plt.bar(np.arange(0,len(bev.Trip_Distances[indices]),1),bev.Trip_Distances[indices]/1000,color='b')
	plt.grid()
	plt.xlabel('Trip/Park Event')
	plt.ylabel('Trip Distance [km]')
	return fig

def ICVTracePlot(bev,dedicated_energizing_time,soc_trace,max_dwells_disp=100):
	selection=[0,len(soc_trace)-1]
	# print(selection)
	if selection[1]>max_dwells_disp:
		selection[1]=max_dwells_disp
	# print(selection,len(soc_trace))
	indices=np.arange(selection[0],selection[1],1)
	indices1=np.arange(selection[0],selection[1]+1,1)
	fig=plt.figure(figsize=(8,8))
	plt.subplot(311)
	plt.plot(soc_trace[indices1],color='b',linewidth=2)
	plt.plot([0,selection[1]],[0,0],color='k',linestyle='--')
	plt.plot([0,selection[1]],[1,1],color='k',linestyle='--')
	plt.grid()
	plt.ylabel('SOC [dim]')
	plt.subplot(312)
	# plt.bar(np.arange(0,len(bev.Parks[indices]),1),bev.Parks[indices]/3600,color='b',alpha=.2)
	plt.bar(np.arange(0,len(bev.Parks[indices]),1),dedicated_energizing_time[indices]/3600,
		color='r')
	plt.legend(['Fueling'])
	plt.grid()
	plt.ylabel('Dedicated Energizing Time [h]')
	plt.subplot(313)
	plt.bar(np.arange(0,len(bev.Trip_Distances[indices]),1),bev.Trip_Distances[indices]/1000,color='b')
	plt.grid()
	plt.xlabel('Trip/Park Event')
	plt.ylabel('Trip Distance [km]')

class BEV():
	"""
	BEV object simulates optimal charging behavior for a Battery Electric itinicle on a set itinerary with an assumed charging grid.
	
	The itinicle is defined by its battery size and electric consumption at low, medium, and high speeds.
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
	Consumption_City/Mixed/Highway - Electric consumption of the itinicle at various speeds [J/m]
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
				Dest_Charger_P_AC=10000,
				En_Route_Charger_P_AC=150000,
				Home_Charger_P_AC=10000,
				Work_Charger_P_AC=10000,
				Nu_AC_DC=.88,
				SOC_Max=1,
				Range_Min=25000,
				SOC_Quanta=50,
				LVL2_Charging_Quanta=2,
				DCFC_Charging_Quanta=10,
				En_Route_Charging_Max=7200,
				SpeedThresholds=np.array([35,65])*0.44704,
				En_Route_Charging_Penalty=15*60,
				Dest_Charging_Function='CalculateCharge_LVL2',
				En_Route_Charging_function='CalculateCharge_DCFC',
				tiles=7):
		
		self.Starting_SOC=Starting_SOC #[dim] BEV's SOC at start of itinerary
		self.Final_SOC=Final_SOC #[dim] BEV's SOC at end of itinerary
		self.Plug_In_Penalty=Plug_In_Penalty #[min] Equivalent inconvenience penalty added to all charge events
		#to account for plug-in time and to account for inconvenience associated with pluging in
		self.Dest_Charger_Likelihood=Dest_Charger_Likelihood #[dim] Likelihood that a given destination will have an available charger
		self.Dest_Charger_P_AC=Dest_Charger_P_AC #[W] AC Power delivered by destination chargers
		self.En_Route_Charger_P_AC=En_Route_Charger_P_AC #[W] AC Power delivered by en route chargers
		self.Home_Charger_P_AC=Home_Charger_P_AC #[W] AC Power delivered by home charger
		self.Work_Charger_P_AC=Work_Charger_P_AC #[W] AC Power delivered by work charger
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
		self.Dest_Charging_Function=Dest_Charging_Function #[method] Method for calculating destination charge delta SOC
		self.En_Route_Charging_function=En_Route_Charging_function #[method] Method for calculating en route charge delta SOC
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
	
	def CalculateCharge_DCFC(self,P_AC,SOC,td_charge):
		#Calcualting the SOC gained from a charging event of duration td_charge
		#The CC-CV curve is an 80/20 relationship where the charging is linear for the first 80%
		#and tails off for the last 20% approaching 100% SOC at t=infiniti
		Delta_SOC=np.zeros(np.shape(SOC)) #Initializing the SOC delta vector
		P_DC=P_AC*self.Nu_AC_DC #[W] DC power received from charger after accounting for AC/DC converter loss
		Lambda_Charging=P_DC/self.Battery_Capacity/.2 #Exponential charging factor
		Total_Energy=P_AC*td_charge #[J] Total energy provided by the charger
		t_80=(.8-SOC)*self.Battery_Capacity/P_DC #[s] Time required to charge up to 80% SOC
		lt=td_charge<=t_80 #Sorting for events which stay below 80%
		gt=td_charge>t_80 #Events which go above 80%
		Delta_SOC[lt]=P_DC/self.Battery_Capacity*td_charge[lt] #Linear charging below 80%
		Delta_SOC[gt]=.8-SOC[gt]+.2*(1-np.exp(-Lambda_Charging*(td_charge[gt]-t_80[gt]))) #Inverse exponential charging above 80%
		return Delta_SOC
	
	def CalculateCharge_LVL2(self,P_AC,SOC,td_charge):
		#Calcualting the SOC gained from a charging event of duration td_charge
		#The CC-CV curve is an 80/20 relationship where the charging is linear for the first 80%
		#and tails off for the last 20% approaching 100% SOC at t=infiniti
		Delta_SOC=np.zeros(np.shape(SOC)) #Initializing the SOC delta vector
		P_DC=P_AC*self.Nu_AC_DC #[W] DC power received from charger after accounting for AC/DC converter loss
		Total_Energy=P_AC*td_charge #[J] Total energy provided by the charger
		t_100=(1-SOC)*self.Battery_Capacity/P_DC #[s] Time required to charge up to 100% SOC
		lt=td_charge<=t_100 #Sorting for events which stay below 80%
		gt=td_charge>t_100 #Events which go above 80%
		Delta_SOC[lt]=P_DC/self.Battery_Capacity*td_charge[lt] #Linear charging below 100%
		Delta_SOC[gt]=1-SOC[gt] #Charging capped at 100%
		return Delta_SOC
	
	def Optimize(self):
		#the optimize step is the first step in the DP solver in which optimal control matrices are created. The optimize step involves
		#backwards iteration through the exogenous input tracres while the optimal control matrices are populated.
		N=len(self.Parks) #Length of itinerary
		#Initializing loop variables
		Cost_to_Go=np.empty((N,len(self.X)))
		Cost_to_Go[:]=np.nan
		Optimal_U1=np.empty((N,len(self.X)))
		Optimal_U1[:]=np.nan
		Optimal_U2=np.empty((N,len(self.X)))
		Optimal_U2[:]=np.nan
		#Pre-calculating discharge events for each trip
		DischargeRates=np.ones(len(self.Trip_Distances))*self.Consumption_Mixed
		DischargeRates[self.Trip_Mean_Speeds<self.SpeedThresholds[0]]=self.Consumption_City
		DischargeRates[self.Trip_Mean_Speeds>=self.SpeedThresholds[1]]=self.Consumption_Highway
		DischargeEvents=self.Trip_Distances*DischargeRates/self.Battery_Capacity
		# print(self.Dest_Charger_P_AC_Array)
		# print(self.Plug_In_Penalty)
		#Main loop (backwards iteration)
		# print(self.X,self.U1,self.U2)
		for k in np.arange(N-1,-1,-1):
			#Initializing state and controls arrays
			SOC,U1,U2=np.meshgrid(self.X,self.U1,self.U2,indexing='ij') #(n_X,n_U1,n_U2) arrays of values for state and controls
			#every combination of starting SOC and each control is evaluated
			
			U1*=self.Parks[k]
			U2*=7200
			
			#Discharging
			SOC-=DischargeEvents[k] #Applying discharge events to the SOC
			
			#Initializing cost array
			Cost=np.zeros((len(self.X),len(self.U1),len(self.U2))) #Array of same dimensions as SOC/U1/U2 which will store the
			#cost of each combination of starting SOC and controls
			
			#Option 1: destination charging - if destination charging is available the resulting change in SOC and resulting cost
			#are calculated
			if self.Dest_Charger_P_AC_Array[k]>0:
				Delta_SOC_Charge=getattr(self,self.Dest_Charging_Function)(
					self.Dest_Charger_P_AC_Array[k],SOC,U1)
				SOC+=Delta_SOC_Charge
				if self.isOther[k]:
					Cost+=self.Plug_In_Penalty
				# Cost[U1<self.Parks[k]]+=1e20
			
			#Option 2: en-route charging - en route charging is always available so the resulting change in SOC and resulting cost
			#are calculated
			Delta_SOC_Charge=getattr(self,self.En_Route_Charging_function)(
				self.En_Route_Charger_P_AC,SOC,U2)
			SOC+=Delta_SOC_Charge
			Cost+=(U2+(U2>0)*(self.En_Route_Charging_Penalty+self.Plug_In_Penalty))
			
			#Enforcing constraints - disallowable combinations of state and controls are assigned huge costs - these will later
			#be used to identify which states are disallowable. Common practice is to assign NaN of Inf cost to disallowable
			#combinations here but this leads to complications later in the code so a very high number works better
			Cost[SOC<self.SOC_Min]=1e50 #SOC too low
			Cost[SOC>self.SOC_Max]=1e50 #SOC too high
			
			#Penalty and Cost to Go - the penalty for failing to meet the final SOC constraint is applied at step N-1
			#(the last step in the itinerary but the first processed in the loop). for all other stpes the cost-to-go is applied.
			#Cost-to-go is the cost to go from step k+1 to step k+2 (k is the current step). Cost-to-go is the "memory" element
			#of the method which ties the steps together
			if k==N-1:
				#Assinging penalty for failing to meet the final SOC constraints
				diff=self.Final_SOC-SOC
				Penalty=diff**2*1e10
				# Penalty[diff<0]=0
				# print(Penalty)
				Cost+=Penalty
			else:
				#Assigning the cost-to-go
				Cost+=np.interp(SOC,self.X,Cost_to_Go[k+1])
			 
			#Finding optimal controls and cost-to-go - Optimal controls for each starting SOC are the controls which result in
			#the lowest cost at that SOC. Cost-to-go is the cost of the optimal controls at each starting SOC
			for i in range(len(self.X)):
				mins=np.min(Cost[i],axis=1) #minimum for each row
				min_inds=np.argmin(Cost[i],axis=1) #minimum axis for each row
				min_row=np.argmin(mins) #row of minimum
				min_col=min_inds[min_row] #column of minimum
				Optimal_U1[k,i]=self.U1[min_row]
				Optimal_U2[k,i]=self.U2[min_col]
				Cost_to_Go[k,i]=Cost[i][min_row,min_col]
			
		#Identifying disallowable optimal controls - optimal controls resulting from disallowable combinations are set to -1
		#so that they can be easily filtered out (all others will be >=0). Disallowable combinations can be "optimal" if all
		#combinations are disallowable for a given SOC and step
		Optimal_U1[Cost_to_Go>=1e50]=-1
		Optimal_U2[Cost_to_Go>=1e50]=-1
		
		#Outputs of BEV.Optimize() are a list of teh optimal control matrices for each control and the cost-to-go matrix
		return ([Optimal_U1,Optimal_U2],Cost_to_Go)
	
	def Evaluate(self,Optimal_Controls):
		#The evaluate step is the second step in the DP method in which optimal controls and states are found for each
		#time-step in the itinerary. This is accomplished through forward iteration. Unlike the optimize step which 
		#considers all possible SOC values, the evaluate step follows only one as it iterates forward using the optimal
		#controls for the current SOC and step.
		#Initializations
		N=len(self.Parks)
		Optimal_U1=np.empty(N)
		Optimal_U1[:]=np.nan
		Optimal_U2=np.empty(N)
		Optimal_U2[:]=np.nan
		En_Route_Penalty=np.zeros(N)
		SOC_Trace=np.empty(N+1)
		SOC_Trace[0]=self.Starting_SOC
		SOC=self.Starting_SOC
		DischargeRates=np.ones(len(self.Trip_Distances))*self.Consumption_Mixed
		DischargeRates[self.Trip_Mean_Speeds<self.SpeedThresholds[0]]=self.Consumption_City
		DischargeRates[self.Trip_Mean_Speeds>=self.SpeedThresholds[1]]=self.Consumption_Highway
		DischargeEvents=self.Trip_Distances*DischargeRates/self.Battery_Capacity 
		#Main loop (forwards iteration)
		for k in np.arange(0,N,1):
			#Discharging
			SOC-=DischargeEvents[k]
			#Charging - the optimal, admissable controls are selected for the current step based on current SOC
			admissable=Optimal_Controls[0][k]>=0
			Optimal_U1[k]=np.around(np.interp(SOC,self.X[admissable],
				Optimal_Controls[0][k][admissable]))*self.Parks[k]
			admissable=Optimal_Controls[1][k]>=0
			Optimal_U2[k]=np.interp(SOC,self.X[admissable],Optimal_Controls[1][k][admissable])*7200
			if self.Dest_Charger_P_AC_Array[k]>0:
				Delta_SOC_Charge=getattr(self,self.Dest_Charging_Function)(
					self.Dest_Charger_P_AC_Array[k],SOC,Optimal_U1[k])
				SOC+=Delta_SOC_Charge
			Delta_SOC_Charge=getattr(self,self.En_Route_Charging_function)(
				self.En_Route_Charger_P_AC,SOC,Optimal_U2[k])
			SOC+=Delta_SOC_Charge
			SOC_Trace[k+1]=SOC
		
		#Time penalty for charging en route is calcualted
		En_Route_Penalty[Optimal_U2>sys.float_info.epsilon]=self.En_Route_Charging_Penalty

		Dedicated_Energizing_Time=(np.zeros(N)+
			(Optimal_U1>sys.float_info.epsilon)*self.isOther*self.Plug_In_Penalty+
			Optimal_U2+
			(self.En_Route_Charging_Penalty+self.Plug_In_Penalty)*(Optimal_U2>sys.float_info.epsilon))
		# Dedicated_Energizing_Time[Dedicated_Energizing_Time>sys.float_info.epsilon]+=self.Plug_In_Penalty

		SIC=(Dedicated_Energizing_Time.sum()/60)/N
		SICD=(Dedicated_Energizing_Time.sum()/60)/(self.Trip_Distances.sum()/1000)
		
		return [Optimal_U1,Optimal_U2],SOC_Trace,Dedicated_Energizing_Time,SIC,SICD

class ICV():
	"""
	ICV object simulates optimal charging behavior for a Internal Combustion itinicle on a set itinerary with an assumed charging grid.
	
	ICV functions similarly to BEV but uses fueling rather than charging. All fueling is done in the same amount of time with regards to the simulation

	"""
	def __init__(self,itin,
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
				Fueling_Rate=121300000*7/60,
				tiles=7):
		
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
		self.tiles=tiles #Number of times the itinerary is repeated
		
		self.ItineraryArrays(itin)
	
	def ItineraryArrays(self,itin):
		#Fixing any non-real dwells
		dwell_time=itin.dwell_time.copy()
		durations=itin.durations.copy()
		dwell_time[dwell_time<0]=dwell_time[dwell_time>=0].mean()
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
		#Populates itinerary arrays
		self.Trip_Distances=np.tile(itin.distances,self.tiles)*1609.34 #[m] Distances of trips preceeding parks
		self.Trip_Times=np.tile(durations,self.tiles)*60 #[s] Durations of trips preceeding parks
		self.Trip_Mean_Speeds=self.Trip_Distances/self.Trip_Times #[m/s] Speeds of trips preceeding parks
		self.Parks=np.tile(dwell_time,self.tiles)*60 #[s] Durations of parks
	
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

		SIC=(Dedicated_Energizing_Time.sum()/60)/N
		SICD=(Dedicated_Energizing_Time.sum()/60)/(self.Trip_Distances.sum()/1000)


		return Optimal_U,SOF_Trace,Dedicated_Energizing_Time,SIC,SICD

class PHEV():
	"""
	BEV object simulates optimal charging behavior for a Battery Electric itinicle on a set itinerary with an assumed charging grid.
	
	The itinicle is defined by its battery size and electric consumption at low, medium, and high speeds.
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
	Consumption_City/Mixed/Highway - Electric consumption of the itinicle at various speeds [J/m]
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
				Consumption_EM_City=385.2,
				Consumption_EM_Mixed=478.8,
				Consumption_EM_Highway=586.8,
				Battery_Capacity=82*1000*3600,
				Starting_SOC=.5,
				Final_SOC=.5,
				Plug_In_Penalty=60,
				Dest_Charger_P_AC=10000,
				En_Route_Charger_P_AC=150000,
				Home_Charger_P_AC=10000,
				Work_Charger_P_AC=10000,
				Nu_AC_DC=.88,
				SOC_Max=1,
				Range_Min=25000,
				SOC_Quanta=50,
				LVL2_Charging_Quanta=2,
				DCFC_Charging_Quanta=10,
				En_Route_Charging_Max=7200,
				SpeedThresholds=np.array([35,65])*0.44704,
				En_Route_Charging_Penalty=15*60,
				Dest_Charging_Function='CalculateCharge_LVL2',
				En_Route_Charging_function='CalculateCharge_DCFC',
				tiles=7):
		
		self.Starting_SOC=Starting_SOC #[dim] BEV's SOC at start of itinerary
		self.Final_SOC=Final_SOC #[dim] BEV's SOC at end of itinerary
		self.Plug_In_Penalty=Plug_In_Penalty #[min] Equivalent inconvenience penalty added to all charge events
		#to account for plug-in time and to account for inconvenience associated with pluging in
		self.Dest_Charger_Likelihood=Dest_Charger_Likelihood #[dim] Likelihood that a given destination will have an available charger
		self.Dest_Charger_P_AC=Dest_Charger_P_AC #[W] AC Power delivered by destination chargers
		self.En_Route_Charger_P_AC=En_Route_Charger_P_AC #[W] AC Power delivered by en route chargers
		self.Home_Charger_P_AC=Home_Charger_P_AC #[W] AC Power delivered by home charger
		self.Work_Charger_P_AC=Work_Charger_P_AC #[W] AC Power delivered by work charger
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
		self.Dest_Charging_Function=Dest_Charging_Function #[method] Method for calculating destination charge delta SOC
		self.En_Route_Charging_function=En_Route_Charging_function #[method] Method for calculating en route charge delta SOC
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
	
	def CalculateCharge_DCFC(self,P_AC,SOC,td_charge):
		#Calcualting the SOC gained from a charging event of duration td_charge
		#The CC-CV curve is an 80/20 relationship where the charging is linear for the first 80%
		#and tails off for the last 20% approaching 100% SOC at t=infiniti
		Delta_SOC=np.zeros(np.shape(SOC)) #Initializing the SOC delta vector
		P_DC=P_AC*self.Nu_AC_DC #[W] DC power received from charger after accounting for AC/DC converter loss
		Lambda_Charging=P_DC/self.Battery_Capacity/.2 #Exponential charging factor
		Total_Energy=P_AC*td_charge #[J] Total energy provided by the charger
		t_80=(.8-SOC)*self.Battery_Capacity/P_DC #[s] Time required to charge up to 80% SOC
		lt=td_charge<=t_80 #Sorting for events which stay below 80%
		gt=td_charge>t_80 #Events which go above 80%
		Delta_SOC[lt]=P_DC/self.Battery_Capacity*td_charge[lt] #Linear charging below 80%
		Delta_SOC[gt]=.8-SOC[gt]+.2*(1-np.exp(-Lambda_Charging*(td_charge[gt]-t_80[gt]))) #Inverse exponential charging above 80%
		return Delta_SOC
	
	def CalculateCharge_LVL2(self,P_AC,SOC,td_charge):
		#Calcualting the SOC gained from a charging event of duration td_charge
		#The CC-CV curve is an 80/20 relationship where the charging is linear for the first 80%
		#and tails off for the last 20% approaching 100% SOC at t=infiniti
		Delta_SOC=np.zeros(np.shape(SOC)) #Initializing the SOC delta vector
		P_DC=P_AC*self.Nu_AC_DC #[W] DC power received from charger after accounting for AC/DC converter loss
		Total_Energy=P_AC*td_charge #[J] Total energy provided by the charger
		t_100=(1-SOC)*self.Battery_Capacity/P_DC #[s] Time required to charge up to 100% SOC
		lt=td_charge<=t_100 #Sorting for events which stay below 80%
		gt=td_charge>t_100 #Events which go above 80%
		Delta_SOC[lt]=P_DC/self.Battery_Capacity*td_charge[lt] #Linear charging below 100%
		Delta_SOC[gt]=1-SOC[gt] #Charging capped at 100%
		return Delta_SOC
	
	def Optimize(self):
		#the optimize step is the first step in the DP solver in which optimal control matrices are created. The optimize step involves
		#backwards iteration through the exogenous input tracres while the optimal control matrices are populated.
		N=len(self.Parks) #Length of itinerary
		#Initializing loop variables
		Cost_to_Go=np.empty((N,len(self.X)))
		Cost_to_Go[:]=np.nan
		Optimal_U1=np.empty((N,len(self.X)))
		Optimal_U1[:]=np.nan
		Optimal_U2=np.empty((N,len(self.X)))
		Optimal_U2[:]=np.nan
		#Pre-calculating discharge events for each trip
		DischargeRates=np.ones(len(self.Trip_Distances))*self.Consumption_Mixed
		DischargeRates[self.Trip_Mean_Speeds<self.SpeedThresholds[0]]=self.Consumption_City
		DischargeRates[self.Trip_Mean_Speeds>=self.SpeedThresholds[1]]=self.Consumption_Highway
		DischargeEvents=self.Trip_Distances*DischargeRates/self.Battery_Capacity
		# print(self.Dest_Charger_P_AC_Array)
		# print(self.Plug_In_Penalty)
		#Main loop (backwards iteration)
		# print(self.X,self.U1,self.U2)
		for k in np.arange(N-1,-1,-1):
			#Initializing state and controls arrays
			SOC,U1,U2=np.meshgrid(self.X,self.U1,self.U2,indexing='ij') #(n_X,n_U1,n_U2) arrays of values for state and controls
			#every combination of starting SOC and each control is evaluated
			
			U1*=self.Parks[k]
			U2*=7200
			
			#Discharging
			SOC-=DischargeEvents[k] #Applying discharge events to the SOC
			
			#Initializing cost array
			Cost=np.zeros((len(self.X),len(self.U1),len(self.U2))) #Array of same dimensions as SOC/U1/U2 which will store the
			#cost of each combination of starting SOC and controls
			
			#Option 1: destination charging - if destination charging is available the resulting change in SOC and resulting cost
			#are calculated
			if self.Dest_Charger_P_AC_Array[k]>0:
				Delta_SOC_Charge=getattr(self,self.Dest_Charging_Function)(
					self.Dest_Charger_P_AC_Array[k],SOC,U1)
				SOC+=Delta_SOC_Charge
				Cost+=self.Plug_In_Penalty
				# Cost[U1<self.Parks[k]]+=1e20
			
			#Option 2: en-route charging - en route charging is always available so the resulting change in SOC and resulting cost
			#are calculated
			Delta_SOC_Charge=getattr(self,self.En_Route_Charging_function)(
				self.En_Route_Charger_P_AC,SOC,U2)
			SOC+=Delta_SOC_Charge
			Cost+=(U2+(U2>0)*(self.En_Route_Charging_Penalty+self.Plug_In_Penalty))
			
			#Enforcing constraints - disallowable combinations of state and controls are assigned huge costs - these will later
			#be used to identify which states are disallowable. Common practice is to assign NaN of Inf cost to disallowable
			#combinations here but this leads to complications later in the code so a very high number works better
			Cost[SOC<self.SOC_Min]=1e50 #SOC too low
			Cost[SOC>self.SOC_Max]=1e50 #SOC too high
			
			#Penalty and Cost to Go - the penalty for failing to meet the final SOC constraint is applied at step N-1
			#(the last step in the itinerary but the first processed in the loop). for all other stpes the cost-to-go is applied.
			#Cost-to-go is the cost to go from step k+1 to step k+2 (k is the current step). Cost-to-go is the "memory" element
			#of the method which ties the steps together
			if k==N-1:
				#Assinging penalty for failing to meet the final SOC constraints
				diff=self.Final_SOC-SOC
				Penalty=diff**2*1e10
				Penalty[diff<0]=0
				# print(Penalty)
				Cost+=Penalty
			else:
				#Assigning the cost-to-go
				Cost+=np.interp(SOC,self.X,Cost_to_Go[k+1])
			 
			#Finding optimal controls and cost-to-go - Optimal controls for each starting SOC are the controls which result in
			#the lowest cost at that SOC. Cost-to-go is the cost of the optimal controls at each starting SOC
			for i in range(len(self.X)):
				mins=np.min(Cost[i],axis=1) #minimum for each row
				min_inds=np.argmin(Cost[i],axis=1) #minimum axis for each row
				min_row=np.argmin(mins) #row of minimum
				min_col=min_inds[min_row] #column of minimum
				Optimal_U1[k,i]=self.U1[min_row]
				Optimal_U2[k,i]=self.U2[min_col]
				Cost_to_Go[k,i]=Cost[i][min_row,min_col]
			
		#Identifying disallowable optimal controls - optimal controls resulting from disallowable combinations are set to -1
		#so that they can be easily filtered out (all others will be >=0). Disallowable combinations can be "optimal" if all
		#combinations are disallowable for a given SOC and step
		Optimal_U1[Cost_to_Go>=1e50]=-1
		Optimal_U2[Cost_to_Go>=1e50]=-1
		
		#Outputs of BEV.Optimize() are a list of teh optimal control matrices for each control and the cost-to-go matrix
		return ([Optimal_U1,Optimal_U2],Cost_to_Go)
	
	def Evaluate(self,Optimal_Controls):
		#The evaluate step is the second step in the DP method in which optimal controls and states are found for each
		#time-step in the itinerary. This is accomplished through forward iteration. Unlike the optimize step which 
		#considers all possible SOC values, the evaluate step follows only one as it iterates forward using the optimal
		#controls for the current SOC and step.
		#Initializations
		N=len(self.Parks)
		Optimal_U1=np.empty(N)
		Optimal_U1[:]=np.nan
		Optimal_U2=np.empty(N)
		Optimal_U2[:]=np.nan
		En_Route_Penalty=np.zeros(N)
		SOC_Trace=np.empty(N+1)
		SOC_Trace[0]=self.Starting_SOC
		SOC=self.Starting_SOC
		DischargeRates=np.ones(len(self.Trip_Distances))*self.Consumption_Mixed
		DischargeRates[self.Trip_Mean_Speeds<self.SpeedThresholds[0]]=self.Consumption_City
		DischargeRates[self.Trip_Mean_Speeds>=self.SpeedThresholds[1]]=self.Consumption_Highway
		DischargeEvents=self.Trip_Distances*DischargeRates/self.Battery_Capacity 
		#Main loop (forwards iteration)
		for k in np.arange(0,N,1):
			#Discharging
			SOC-=DischargeEvents[k]
			#Charging - the optimal, admissable controls are selected for the current step based on current SOC
			admissable=Optimal_Controls[0][k]>=0
			Optimal_U1[k]=np.around(np.interp(SOC,self.X[admissable],
				Optimal_Controls[0][k][admissable]))*self.Parks[k]
			admissable=Optimal_Controls[1][k]>=0
			Optimal_U2[k]=np.interp(SOC,self.X[admissable],Optimal_Controls[1][k][admissable])*7200
			if self.Dest_Charger_P_AC_Array[k]>0:
				Delta_SOC_Charge=getattr(self,self.Dest_Charging_Function)(
					self.Dest_Charger_P_AC_Array[k],SOC,Optimal_U1[k])
				SOC+=Delta_SOC_Charge
			Delta_SOC_Charge=getattr(self,self.En_Route_Charging_function)(
				self.En_Route_Charger_P_AC,SOC,Optimal_U2[k])
			SOC+=Delta_SOC_Charge
			SOC_Trace[k+1]=SOC
		
		#Time penalty for charging en route is calcualted
		En_Route_Penalty[Optimal_U2>sys.float_info.epsilon]=self.En_Route_Charging_Penalty

		Dedicated_Energizing_Time=(np.zeros(N)+
			(Optimal_U1>sys.float_info.epsilon)*self.Plug_In_Penalty+
			Optimal_U2+
			(self.En_Route_Charging_Penalty+self.Plug_In_Penalty)*(Optimal_U2>sys.float_info.epsilon))
		# Dedicated_Energizing_Time[Dedicated_Energizing_Time>sys.float_info.epsilon]+=self.Plug_In_Penalty

		SIC=(Dedicated_Energizing_Time.sum()/60)/N
		SICD=(Dedicated_Energizing_Time.sum()/60)/(self.Trip_Distances.sum()/1000)
		
		return [Optimal_U1,Optimal_U2],SOC_Trace,Dedicated_Energizing_Time,SIC,SICD