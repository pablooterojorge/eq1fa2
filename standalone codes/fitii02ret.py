#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:07:36 2022

@author: pablosanchez
"""

import pandas as pd
import numpy as np
from lmfit import Minimizer, Parameters, report_fit

## def function

def fitii02ret (params, x, y):
    a = params['a']
    b = params['b']
    o = params['o']
    fitii0 = a * np.sin(np.pi * (x + b))**2 + o
    return (fitii0 - y)**2 



# Define and pass fitting parameters

params = Parameters()
params.add('a'    , value = 1.0  , min= 0.2, max= 1.5  ,  vary = True)
params.add('b'    , value = 0.3  , min= 0.0, max= 1.0  ,  vary = True)
params.add('o'    , value = 10.0 , min= 0.0, max= 100. ,  vary = True)



# fitting data to functiom

minner0 = Minimizer(fitii02ret, params, fcn_args=(x0,y0))
minner1 = Minimizer(fitii02ret, params, fcn_args=(x0,y1))

res0 = minner0.minimize(method='leastsq')
res1 = minner1.minimize(method='leastsq')



# report, convert results in numpy array
ffit0 = np.array([res0.params['a'].value, res0.params['b'].value, res0.params['o'].value])
ffit1 = np.array([res1.params['a'].value, res1.params['b'].value, res1.params['o'].value])