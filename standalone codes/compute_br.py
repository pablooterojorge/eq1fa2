#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:56:55 2022

@author: pablosanchez
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import scipy.optimize as opt



def scale2one(myarray):
    array_to_one = np.zeros(len(myarray))
    for i in range(len(myarray)):
        array_to_one[i] = (myarray[i] - np.amin(myarray))/(np.amax(myarray)-np.amin(myarray))
    return array_to_one



## input voltaje vs brightness of sample (fiber) and background (lc-retarder)

df = pd.read_pickle("./brf.pkl") 

vrms = df.iloc[:,0].to_numpy()
inty = df.iloc[:,3].to_numpy()
inty_sample = df.iloc[:,1].to_numpy()



## compute datapoints for sample and background


def compute_retims(vrms, inty, inty_sample):
    
    vrms = df.iloc[:,0].to_numpy()
    
    ii0 = scale2one(inty)
    ii0_sample = scale2one(inty_sample)
    
    ii0max = signal.argrelextrema(ii0, np.greater)[0]
    ii0min = signal.argrelextrema(ii0, np.less)[0]
    

    for i in range(len(ii0max)):
        if vrms[ii0max[i]] >= 4.1 and vrms[ii0max[i]] <= 4.5:
            tmax = ii0max[i]
    for i in range(len(ii0min)):
        if vrms[ii0min[i]] >= 2.5 and vrms[ii0min[i]] <= 2.9:
            tmin = ii0min[i]
        
    retims = np.zeros(len(ii0))

    for i in range(len(ii0)):
        if vrms[i] <=  vrms[tmin]:
            retims[i] = 1 + np.arcsin(np.sqrt(ii0[i]))/np.pi
        elif vrms[i] <= vrms[tmax]:
            retims[i] = 1 - np.arcsin(np.sqrt(ii0[i]))/np.pi
        else:
            retims[i] = np.arcsin(np.sqrt(ii0[i]))/np.pi
    
    
    return retims, ii0, ii0_sample


## compute fitting where deltan is taking from 

retims, ii0, ii0_sample = compute_retims(vrms, inty, inty_sample)

def compute_retsample(retims, ii0, ii0_sample):
    
    def lc(x, a, o):
        return a * np.sin(np.pi*(x))**2 + o

    def fibre(x, a, b, o):
        return a * np.sin(np.pi*(x+b))**2 + o
    
    paramslc, covlc = opt.curve_fit(lc, retims, ii0, [100, 10])
    
    paramsfibre, covfibre = opt.curve_fit(fibre, retims, ii0_sample, [100, 0.3,10])
    
    return paramslc, paramsfibre


paramslc, paramsfibre = compute_retsample(retims, ii0, ii0_sample)


## plot data and fitting, remove when the whole system is built

aux = np.linspace(0, 1.5, num=200)


plt.plot(retims, ii0, 'or')
plt.plot(aux, paramslc[0] * np.sin(np.pi*(aux))**2 + paramslc[1] , '-r')


plt.plot(retims, ii0_sample, 'ob')
plt.plot(aux, paramsfibre[0] * np.sin(np.pi*(aux + paramsfibre[1] ))**2 + paramsfibre[2] , '-b')

