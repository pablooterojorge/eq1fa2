# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 10:54:54 2021

@author: po-po
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#filename = r'C:\Users\po-po\Desktop\DOC\Fibras\Programas\data\dr2todr4e01121121.csv'
filename = r'C:\Users\po-po\Desktop\DOC\Fibras\Programas\data\drgodet\r5pa1dr2e0f10.csv'
clean_arr = pd.read_csv(filename)
file = str(os.path.splitext(os.path.basename(filename))[0])

#plot formating

params = {'figure.figsize':      (6, 4),
          'font.size':           18,
          'font.sans-serif':     'Arial',
          'lines.linewidth':     2.0,
          'axes.linewidth':      1.5,
          'axes.formatter.use_mathtext': True,
          'axes.formatter.min_exponent': False,
          'axes.formatter.useoffset': False,
          'axes.grid': False,
          'axes.grid.axis': 'both',
          'xtick.minor.visible': True,
          'ytick.minor.visible': True,
          'xtick.direction':     'in',
          'xtick.top':           True,
          'ytick.direction':     'in',
          'ytick.right':         True,
          'xtick.major.size':     10,
          'xtick.minor.size':     5,
          'xtick.major.width':    1,
          'ytick.major.size':     10,
          'ytick.minor.size':     5,
          'ytick.major.width':    1,
          'legend.frameon':       True,
         }
plt.rcParams.update(params)

fig = plt.figure()
#perform rolling average on pandas dataframe of clean data
interval = 100

clean_arr['Average'] = clean_arr['Diameter'].rolling(window = interval, center = True, min_periods = 1).mean()
clean_arr['Std'] = clean_arr['Diameter'].rolling(window = interval, center = True, min_periods = 1).std()

clean_arr['Clean'] = clean_arr.Diameter[(clean_arr['Diameter'] >= clean_arr['Average']-clean_arr['Std']) & (clean_arr['Diameter'] <= clean_arr['Average']+clean_arr['Std'])]
clean_arr['Dirty'] = clean_arr.Diameter[(clean_arr['Diameter'] <= clean_arr['Average']-clean_arr['Std']) | (clean_arr['Diameter'] >= clean_arr['Average']+clean_arr['Std'])]

clean_arr['CAverage'] = clean_arr['Clean'].rolling(window = interval, center = True, min_periods = 1).mean()
clean_arr['Marked'] = clean_arr.Time[clean_arr['Event Flag'] == 1]

#plot diameter array

stflag = 1

if stflag == 1:
    
    plt.plot(clean_arr['Time'],clean_arr['Clean'],'kx')
    plt.plot(clean_arr['Time'],clean_arr['CAverage'],'b-')
    plt.plot(clean_arr['Marked'], clean_arr['Event Flag'], 'go')
    
else:
    
    plt.plot(clean_arr['Time'],clean_arr['Clean'],'kx')
    plt.plot(clean_arr['Time'],clean_arr['CAverage'],'b-')
    plt.plot(clean_arr['Marked'], clean_arr['Event Flag'], 'go')
    plt.plot(clean_arr['Time'],clean_arr['Average']-clean_arr['Std'],'r--')
    plt.plot(clean_arr['Time'],clean_arr['Average']+clean_arr['Std'],'r--')
    plt.plot(clean_arr['Time'],clean_arr['Dirty'],'rx')
    
plt.xlabel('Time (s)')
plt.ylabel('Fiber Diameter (um)')
plt.title('%s'%file)

plt.show()
