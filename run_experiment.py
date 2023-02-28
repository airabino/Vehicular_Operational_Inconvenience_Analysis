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
from src.process_nhts_data import FilterByAttribute, Itinerary
from src.designed_experiment import Run

now=datetime.now()
parser=argparse.ArgumentParser(description='Input arguments for designed experiments')
parser.add_argument('-i','--in_file',type=str,default='Data/Generated_Data/NHTS_Itineraries.pkl',
	help='Path to itineraries pickle')
parser.add_argument('-o','--out_file',type=str,
	default='Data/Generated_Data/Experiment_Results_'+now.strftime("%d_%m_%Y_%H_%M_%S_%f")+'.pkl',
	help='Path to location to save outputs')
parser.add_argument('-hc','--home_charging',type=float,nargs='+',default=[0.,1.],
	help='Experimental values for home charging [-]')
parser.add_argument('-wc','--work_charging',type=float,nargs='+',default=[0.,1.],
	help='Experimental values for work charging [-]')
parser.add_argument('-dcl','--destination_charging_likelihood',type=float,nargs='+',
	default=[0,.075,.15],
	help='Experimental values for destination charging likelihood [-]')
parser.add_argument('-dcfcr','--dcfc_rate',type=float,nargs='+',default=[50,150,250],
	help='Experimental values for dcfc rate [kW]')
parser.add_argument('-dcfcp','--dcfc_penalty',type=float,nargs='+',default=[0,25,50],
	help='Experimental values for dcfc penalty [min]')
parser.add_argument('-bc','--battery_capacity',type=float,nargs='+',default=[40,80,120],
	help='Experimental values for battery capacity [kWh]')

parser.add_argument('-frq','--frequency',type=int,default=1)
parser.add_argument('-iter','--iterations',type=int,default=3)

parser.add_argument('-s','--state',type=str,
	help='Down-select itineraries to state by FIPS (do not include 0 before number)')
parser.add_argument('-m','--msa',type=str,help='Down-select itineraries to MSA')

args=parser.parse_args(sys.argv[1:])

print(args)

print('Loading Data')
itineraries=pkl.load(open(args.in_file,'rb'))



if args.state != None:
	itineraries=FilterByAttribute(itineraries,'hh_state',args.state)

if args.msa != None:
	itineraries=FilterByAttribute(itineraries,'hh_msa',args.msa)

df=Run(itineraries,
	HC=np.array(args.home_charging),
	WC=np.array(args.work_charging),
	DCL=np.array(args.destination_charging_likelihood),
	BC=np.array(args.battery_capacity)*1000*3600,
	DCFCR=np.array(args.dcfc_rate)*1000,
	DCFCP=np.array(args.dcfc_penalty)*60,
	FRQ=args.frequency,
	iterations=args.iterations,
	)

print('Saving Data')
pkl.dump(df,open(args.out_file,'wb'))