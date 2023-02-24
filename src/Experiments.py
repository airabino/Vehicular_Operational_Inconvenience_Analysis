import sys
import time
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from DP_Sim import *
from ProcessData import *

# np.set_printoptions(threshold=20,suppress=True)

def fullfact(levels):
	n = len(levels)  # number of factors
	nb_lines = np.prod(levels)  # number of trial conditions
	H = np.zeros((nb_lines, n))
	level_repeat = 1
	range_repeat = np.prod(levels)
	for i in range(n):
		range_repeat /= levels[i]
		lvl = []
		for j in range(levels[i]):
			lvl += [j]*level_repeat
		rng = lvl*int(range_repeat)
		level_repeat *= levels[i]
		H[:,i] = rng
	return H

def RunCase(case,itineraries,row):
	data=np.ones((len(itineraries),16))*-1
	for idx in tqdm(range(len(itineraries))):
		try:
			bev=BEV(itineraries[idx],
				Dest_Charger_Likelihood=row[2],
				Consumption_City=row[3],
				Consumption_Mixed=row[4],
				Consumption_Highway=row[5],
				Battery_Capacity=row[6],
				Plug_In_Penalty=row[7],
				Dest_Charger_P_AC=row[8],
				Home_Charger_P_AC=row[0]*row[8],
				Work_Charger_P_AC=row[1]*row[8],
				En_Route_Charger_P_AC=row[9],
				En_Route_Charging_Penalty=row[10])
			optimal_controls,cost_to_go=bev.Optimize()
			optimal_control,soc_trace,dedicated_energizing_time,sic,sicd=bev.Evaluate(optimal_controls)
			# print(sic,sicd)
			data[idx]=np.hstack((case,row,bev.Trip_Distances.sum()/bev.tiles,
				bev.Parks.sum()/bev.tiles,sic,sicd))
		except:
			print('a',idx)
			data[idx,0]=case
		# break
	return data

def RunExperiment(itineraries, #[iterable] Itinerary objects
				HC=np.array([1]), #[bool] home charging allowed (LVL2)
				WC=np.array([1]), #[bool] work charging allowed (LVL2)
				DCL=np.array([.1]), #[dim] proportion of likely destinations with LVL2 charger available
				CC=np.array([385.2]), #[J/m] vehicle energy consumption rate in city conditions
				CM=np.array([478.8]), #[J/m] vehicle energy consumption rate in mixed conditions
				CH=np.array([586.8]), #[J/m] vehicle energy consumption rate in highway conditions
				BC=np.array([82*1000*3600]), #[J] vehicle battery energy storage capacity
				PIP=np.array([60]), #[s] penalty for plugging a vehicle in
				L2CR=np.array([10000]), #[W] charging rate for LVL2 chargers
				DCFCR=np.array([150000]), #[W] charging rate for DCFC chargers
				DCFCP=np.array([15*60]), #[s] penalty for traveling to DCFC charger
				FRQ=1): #Allows down-selection of itineraries for debugging
	#Down-selecting itineraries
	selected_itineraries=itineraries[np.arange(0,len(itineraries),FRQ).astype(int)]
	#Creating the experiment levels
	num_levels=([len(HC),len(WC),len(DCL),len(CC),len(CM),
		len(CH),len(BC),len(PIP),len(L2CR),len(DCFCR),len(DCFCP)])
	rows=fullfact(num_levels).astype(int)
	rows_exp=np.empty(rows.shape)
	rows_exp[:,0]=HC[rows[:,0]]
	rows_exp[:,1]=WC[rows[:,1]]
	rows_exp[:,2]=DCL[rows[:,2]]
	rows_exp[:,3]=CC[rows[:,3]]
	rows_exp[:,4]=CM[rows[:,4]]
	rows_exp[:,5]=CH[rows[:,5]]
	rows_exp[:,6]=BC[rows[:,6]]
	rows_exp[:,7]=PIP[rows[:,7]]
	rows_exp[:,8]=L2CR[rows[:,8]]
	rows_exp[:,9]=DCFCR[rows[:,9]]
	rows_exp[:,10]=DCFCP[rows[:,10]]
	#Pre-allocating the data array
	#Data array will have form [idx,HC,WC,DCL,CC,CM,CH,BC,PIP,L2CR,DCFCR,DCFCP,DT,DD,SICT,SICD]
	#DT - Daily Travel Distance [m]
	#DD - Daily Dwell Time [s]
	#SICT - Inconvenience Score (Trip) [min/trip]
	#SICD - Inconvenience Score (Distance) [min/km]
	data=np.zeros((len(rows_exp)*len(selected_itineraries),16))
	# print(data.shape)
	k=0
	#Main Loop
	for idx in range(len(rows_exp)):
		row=rows_exp[idx]
		print('Row {} of {}'.format(idx,len(rows_exp)))
		# print(row)
		#Generating results for a given case
		data[k:k+len(selected_itineraries)]=RunCase(idx,selected_itineraries,row)
		print(data[k:k+len(selected_itineraries),14].mean(),data[k:k+len(selected_itineraries),15].mean())
		k+=len(selected_itineraries)
	return data

def ConsolidateResults(df):
	data=df.to_numpy()
	unique_cases,unique_case_indices=np.unique(data[:,0],return_inverse=True)
	# print(unique_cases,unique_case_indices)
	data_out=np.empty((unique_cases.shape[0],data.shape[1]))
	for idx in unique_cases.astype(int):
		data_out[idx]=data[unique_case_indices==idx].mean(axis=0)
	return data_out


if __name__ == "__main__":
	print('Loading Data')
	itineraries=pkl.load(open('long_itineraries_DEN_MSA.pkl','rb'))
	print('Running Experiment')
	data=RunExperiment(itineraries,
				HC=np.array([0,1]),
				WC=np.array([0,1]),
				DCL=np.array([0,.075,.15]),
				BC=np.array([40,80,120])*1000*3600,
				DCFCR=np.array([50,150,250])*1000,
				DCFCP=np.array([0,25,50])*60,
				FRQ=1)
	# data=RunExperiment(long_itineraries,
	# 			HC=np.array([0,1]),
	# 			FRQ=1000)
	print('Saving Data')
	df=pd.DataFrame(data,
		columns=['idx','HC','WC','DCL','CC','CM','CH','BC','PIP','L2CR','DCFCR','DCFCP','DT','DD','SICT','SICD'])
	pkl.dump(df,open('NHTS_Exp_DEN_MSA_0.pkl','wb'))
	print('Done')

