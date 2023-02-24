import sys
import time
import numpy as np
import numpy.random as rand
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statsmodels.formula.api import ols, wls
from sklearn.preprocessing import normalize
from scipy.stats import f as f_dist

def Normalize(data):
	maxes=data.max(axis=0)
	mins=data.min(axis=0)
	data=(data-np.tile(mins,(data.shape[0],1)))/np.tile(maxes-mins,(data.shape[0],1))
	return data,maxes,mins

def ReScale(data,maxes,mins):
	return data*np.tile(maxes-mins,(data.shape[0],1))+np.tile(mins,(data.shape[0],1))

def ProduceRegressionResultsTrip(df):
	data=df.to_numpy()
	data,maxes,mins=Normalize(data)
	df1=pd.DataFrame(data,columns=df.keys())
	model=ols('SICT ~ HC * WC * BC * DCL * DCFCR * DCFCP',data=df1).fit()
	return model,df1,data,maxes,mins

def ProduceRegressionResultsDist(df):
	data=df.to_numpy()
	data,maxes,mins=Normalize(data)
	df1=pd.DataFrame(data,columns=df.keys())
	model=ols('SICD ~ HC * WC * BC * DCL * DCFCR * DCFCP',data=df1).fit()
	return model,df1,data,maxes,mins

def ProduceRegressionResults_ICV(data):
	Results=data['Results']
	Regressors=data['Regressors']
	# Regressors[:,1][Regressors[:,1]==0]=.0001
	data=np.hstack((np.array([[res.mean('InconvenienceScore') for res in Results]]).T,Regressors))
	data,maxes,mins=Normalize(data)
	df=pd.DataFrame(data,columns=['IC','FTC','FTP'])
	model=ols('IC ~ FTC * FTP',data=df).fit()
	return model,df,data,maxes,mins

def SignificantParametersPlot(model,alpha=.05):
	params=model._results.params[1:]
	error=model._results.bse[1:]
	pvalues=model._results.pvalues[1:]
	names=np.array(list(dict(model.params).keys()))[1:]
	params=params[pvalues<alpha]
	error=error[pvalues<alpha]
	names=names[pvalues<alpha]
	pvalues1=pvalues[pvalues<alpha]
	name_lengths=[len(name) for name in names]
	name_length_order=np.flip(np.argsort(name_lengths))
	fig,ax=plt.subplots(figsize=(8,8))
	plt.barh(list(range(len(params))),params[name_length_order],xerr=error,
		ec=(0,0,0,1),ls='-',lw=2,fc=(0,0,0,.2))
	ax.set_xlabel('Beta [dim]',fontsize='x-large')
	ax.set_ylabel('Coefficient',fontsize='x-large')
	ax.set_yticks(list(range(len(names))))
	ax.set_yticklabels(names[name_length_order])
	ax.grid(linestyle='--')
	return fig

def SignificantParametersComparisonPlot(model1,model2,model3,alpha=.05):
	params=model1._results.params[1:]
	error=model1._results.bse[1:]
	pvalues=model1._results.pvalues[1:]
	names=np.array(list(dict(model1.params).keys()))[1:]
	params=params[pvalues<alpha]
	error=error[pvalues<alpha]
	names=names[pvalues<alpha]
	pvalues1=pvalues[pvalues<alpha]
	params1=model2._results.params[1:][pvalues<alpha]
	params2=model3._results.params[1:][pvalues<alpha]
	name_lengths=[len(name) for name in names]
	name_length_order=np.argsort(name_lengths)
	fig,ax=plt.subplots(figsize=(12,6))
	# plt.bar(np.arange(0,len(names),1)-.25,params[name_length_order],width=.2,ls='-',lw=2,
	# 	fc=(88/255,181/255,225/255,1),ec=(0,0,0,1))
	# plt.bar(np.arange(0,len(names),1),params1[name_length_order],width=.2,ls='-',lw=2,
	# 	fc=(28/255,91/255,90/255,1),ec=(0,0,0,1))
	# plt.bar(np.arange(0,len(names),1)+.25,params2[name_length_order],width=.2,ls='-',lw=2,
	# 	fc=(16/255,237/255,220/255,1),ec=(0,0,0,1))
	plt.bar(np.arange(0,len(names),1)-.25,params[name_length_order],width=.2,ls='-',lw=2,
		fc='r',alpha=.5,ec=(0,0,0,1))
	plt.bar(np.arange(0,len(names),1),params1[name_length_order],width=.2,ls='-',lw=2,
		fc='g',alpha=.5,ec=(0,0,0,1))
	plt.bar(np.arange(0,len(names),1)+.25,params2[name_length_order],width=.2,ls='-',lw=2,
		fc='b',alpha=.5,ec=(0,0,0,1))
	# plt.barh(list(range(len(params))),params[name_length_order],xerr=error,
	# 	ec=(0,0,0,1),ls='-',lw=2,fc=(0,0,0,.2))
	ax.set_xlabel('Coefficient',fontsize='x-large')
	ax.set_ylabel('Beta [dim]',fontsize='x-large')
	ax.set_xticks(list(range(len(names))))
	ax.set_xticklabels(names[name_length_order],rotation='vertical')
	ax.grid(linestyle='--')
	ax.legend(['National','Colorado','Denver MSA'])
	return fig

def RSS(x,y):
	return ((x-y)**2).sum()

def MSS(x,y):
	return ((y-x.mean())**2).sum()

def TSS(x):
	return ((x-x.mean())**2).sum()

def RSquared(x,y):
	return 1-(RSS(x,y)/TSS(x))

def AdjustedRSquared(x,y,n,p):
	return 1-(((1-RSquared(x,y))*(n-1))/(n-p-1))

def ANOVA(x,y,n,p):
	sse=RSS(x,y)
	ssm=MSS(x,y)
	sst=TSS(x)
	dfe=n-p
	dfm=p-1
	dft=n-1
	mse=sse/dfe
	msm=ssm/dfm
	mst=sst/dft
	f=msm/mse
	pf=f_dist.sf(f,dfm,dfe)
	r2=1-(sse/sst)
	ar2=1-(((1-r2)*dft)/(dfe-1))
	return {'R':np.sqrt(r2),'RSquared':r2,'AdjustedRSquared':ar2,'StdError':(x-y).std()/n,
		'SSE':sse,'SSM':ssm,'SST':sst,'DFM':dfm,
		'DFE':dfe,'DFT':dft,'MSM':msm,'MSE':mse,'MST':mst,'F':f,'P(F)':pf}

def PrintLaTeXTabular(model,alpha=.05):
	params=model._results.params
	tvalues=model._results.tvalues
	pvalues=model._results.pvalues
	names=np.array(list(dict(model.params).keys()))
	params=params[pvalues<alpha]
	tvalues=tvalues[pvalues<alpha]
	names=names[pvalues<alpha]
	pvalues=pvalues[pvalues<alpha]
	# print(names)
	name_lengths=[len(name) for name in names]
	# print(name_lengths)
	# print(name_lengths[1:])
	# print(np.argsort(name_lengths[1:]))
	# print(np.sort(name_lengths[1:]))
	name_length_order=np.append(0,np.argsort(name_lengths[1:])+1)
	# print(name_length_order)
	# name_length_order=np.argsort(name_lengths)
	# print(name_length_order)

	params=params[name_length_order]
	tvalues=tvalues[name_length_order]
	names=names[name_length_order]
	pvalues=pvalues[name_length_order]
	out_string=""
	for i in range(len(names)):
		out_string+="\\hline {{\\small {} }} & {:.3f} & {:.3f} & {:.3f} \\\\\n".format(names[i],params[i],tvalues[i],pvalues[i])
	print(out_string)