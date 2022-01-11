# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:15:04 2021

@author: Pablo Otero
"""

from ximea import xiapi
import cv2
import time
import pyvisa 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import scipy.optimize as opt

#create instance for first connected camera 
cam = xiapi.Camera()

#start communication
print('Opening first camera...')
cam.open_device()

#ximea camera settings
cam.set_exposure(2000)
cam.set_gain(10)
cam.set_imgdataformat('XI_RGB24')

#siglent signal generator settings
rm = pyvisa.ResourceManager()
rm.list_resources() 
sdg = rm.open_resource("USB0::0xF4ED::0xEE3A::SDG08CBX3R0499::INSTR")

#create instance of Image to store image data and metadata
cap = xiapi.Image()

#start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

#initialize time and iterators
t0 = time.time()
framecounter = 0
iv = 0

#define time periods for BR computation
fps = 10
avgtime = 0.1
perf = fps

#compute loop delay and number of frames to average per vrms measure
loopdelay = int(1/fps*1000)
vframes = avgtime*1000/loopdelay

#initialize graphic windows
cv2.namedWindow('Input Frame')
#cv2.namedWindow('Processed Frame')

#define empty function for trackbars
def nothing(x):
    pass

#initialize trackbar for br computation trigger
cv2.createTrackbar('BR Enable','Input Frame',0,1,nothing)
#initialize trackbar for offset of fiber ii0 zone
cv2.createTrackbar('Offset ii0','Input Frame',0,512,nothing)
#create voltage array
vlist = np.arange(0.1,20.1,0.1)

#set siglent state
sdg.write("C1:OUTP OFF")
time.sleep(0.05)
sdg.write("C1:BSWV WVTP, SQUARE")
time.sleep(0.05)
sdg.write("C1:BSWV FRQ, 2000")
time.sleep(0.05)

#set voltage value
v = vlist[0]
sdg.write("C1:BSWV AMP, %s"%v)
time.sleep(0.05)

#create empty dataframe for data saving between iterations
brf = pd.DataFrame(index = range(len(vlist)))
brf['vrms'] = vlist/2
brf['ii0f'] = np.NAN
brf['ii0fn'] = np.NAN
brf['ii0lc'] = np.NAN
brf['ii0lcn'] = np.NAN
brf['retlcn'] = np.NAN
brf['retf'] = np.NAN

#create empty dataframe to store data of experiment
data = pd.DataFrame(index = np.arange(10000))
data['lca'] = np.NAN
data['lcaSE'] = np.NAN
data['lco'] = np.NAN
data['lcoSE'] = np.NAN
data['fa'] = np.NAN
data['faSE'] = np.NAN
data['fb'] = np.NAN
data['fbSE'] = np.NAN
data['fo'] = np.NAN
data['foSE'] = np.NAN
datacounter = 0

#initialize mean brightness for fiber
ii0favg = 0;
ii0lcavg = 0;

#initialize fit parameters
paramsfibre = [0,0,0]
paramslc = [0,0]
tmaximum = 0
tminimum = 0

#set section to measure
topinit = 512;
bottinit = 512;
off = 20

#plotting
#initialize plotting window
fig, ax = plt.subplots()

ax.set_xlabel('Retardance')
ax.set_ylabel('ii0')

pointslc, = ax.plot(0, 0, 'kx', label='LC Data')
pointsret, = ax.plot(0, 0, 'rx', label='Fiber Data')
fitlc, = ax.plot(0, 0, 'k-', label='LC Fit')
fitfiber, = ax.plot(0, 0, 'r-', label='Fiber Fit')

ax.set_xlim(0, 1.5)
ax.set_ylim(0,180)

ax.legend()

plt.ion()
plt.show()

#enable siglent output
sdg.write("C1:OUTP ON")
time.sleep(0.05)

try:
    print('Starting video. Press CTRL+C to exit.')
    while True:    
        
        t1 = time.perf_counter()
        #get data and pass them from camera to img
        cam.get_image(cap)
        
        #create numpy array with data from camera. Dimensions of the array are 
        #determined by imgdataformat
        img = cap.get_image_data_numpy()
        
        #get offset
        off = cv2.getTrackbarPos('Offset ii0', 'Input Frame')
        top = int(topinit - off)
        bot = int(bottinit + off)
        
        #check br trigger
        if  cv2.getTrackbarPos('BR Enable', 'Input Frame') == 1:
            
            #add mean brightness for fiber
            ii0favg = (ii0favg + np.mean(img[top:bot,:,1]))/2
        
            #add mean brightness for lc retarder
            ii0lcavg = (ii0lcavg + np.mean(img[0,:,1]))/2
            
            #check if its time for br computation, avgtime has elapsed
            if framecounter == vframes:
                
                #set new voltage value
                v = str(vlist[iv])
                sdg.write("C1:BSWV AMP, %s"%v)
                
                #get data and pass them from camera to img
                cam.get_image(cap)
                
                #create numpy array with data from camera. Dimensions of the array are 
                #determined by imgdataformat
                img = cap.get_image_data_numpy()
                
                #save current voltage level and average brightness
                brf.vrms[iv] = vlist[iv]
                brf.ii0f[iv] = ii0favg
                brf.ii0lc[iv] = ii0lcavg
                
                #reset average values
                ii0favg = 0
                ii0lcavg = 0
                
                #increase iv loop by 1
                iv += 1
                
                #reset framecounter
                framecounter = 0
                
            #check if iv loop is complete
            if iv == len(vlist):
                
                #compute br from saved data for current ivloop
                
                #compute normalized data, CHECK NORMALIZATION, IS CORRECT?
                for i in range(len(brf.ii0f)):
                    brf.ii0fn[i] = (brf.ii0f[i] - np.amin(brf.ii0f))/(np.amax(brf.ii0f)-np.amin(brf.ii0f))
                    brf.ii0lcn[i] = (brf.ii0lc[i] - np.amin(brf.ii0lc))/(np.amax(brf.ii0lc)-np.amin(brf.ii0lc))
                
                #find local extrema for cal data in fiber lc, first we check for
                #local extrema, then we check for true minimum and maximum with vrms
                
                maximums = signal.argrelextrema(brf.ii0lc.to_numpy(), np.greater)[0]
                minimums = signal.argrelextrema(brf.ii0lc.to_numpy(), np.less)[0]
                
                for i in range(len(maximums)):
                    if brf.vrms[maximums[i]] >= 4.1 and brf.vrms[maximums[i]] <= 4.5:
                        tmaximum = maximums[i]
                for i in range(len(minimums)):
                    if brf.vrms[minimums[i]] >= 2.5 and brf.vrms[minimums[i]] <= 2.9:
                        tminimum = minimums[i]
                        
                #compute retardance from normalized ii0 lc data      
                for i in brf.index:
                    
                    if brf.vrms[i] <=  brf.vrms[tminimum]:
                        brf.retlcn[i] = 1 + np.arcsin(np.sqrt(brf.ii0lcn[i]))/np.pi
                    elif brf.vrms[i] <= brf.vrms[tmaximum]:
                        brf.retlcn[i] = 1 - np.arcsin(np.sqrt(brf.ii0lcn[i]))/np.pi
                    else:
                        brf.retlcn[i] = np.arcsin(np.sqrt(brf.ii0lcn[i]))/np.pi
               
                #fit data
                def lc(x, a, o):
                    return a * np.sin(np.pi*(x))**2 + o
                def fibre(x, a, b, o):
                    return a * np.sin(np.pi*(x+b))**2 + o
                try:
                    paramslc, covlc = opt.curve_fit(lc, brf.retlcn, brf.ii0lc, [100,10])
                    lcse = np.sqrt(np.diag(covlc))
                except RuntimeError:
                    print("Error - curve_fit lc failed")
                try:
                    paramsfibre, covfibre = opt.curve_fit(fibre, brf.retlcn, brf.ii0f, [100,0.3,10]) 
                    fibrese = np.sqrt(np.diag(covfibre))
                except RuntimeError:
                    print("Error - curve_fit fibre failed")
                
                #save data to storing array
                data.lca[datacounter] = paramslc[0]
                data.lco[datacounter] = paramslc[1]
                data.fa[datacounter] = paramsfibre[0]
                data.fb[datacounter] = paramsfibre[1]
                data.fo[datacounter] = paramsfibre[2]
                data.lcaSE[datacounter] = lcse[0]
                data.lcoSE[datacounter] = lcse[1]
                data.faSE[datacounter] = fibrese[0]
                data.fbSE[datacounter] = fibrese[1]
                data.foSE[datacounter] = fibrese[2]
                
                datacounter +=1
                if datacounter == len(data):
                    datacounter = 0
                    
                #generate fit data to plot
                fitx = np.linspace(-0.02,1.5,200)
                fity = lc(fitx,paramslc[0],paramslc[1])
                fityf = fibre(fitx,paramsfibre[0],paramsfibre[1],paramsfibre[2])
                
                #update plot data                
                pointslc.set_xdata(brf['retlcn'])
                pointslc.set_ydata(brf['ii0lc'])
                pointsret.set_xdata(brf['retlcn'])
                pointsret.set_ydata(brf['ii0f'])
                
                fitlc.set_xdata(fitx)
                fitlc.set_ydata(fity)
                fitfiber.set_xdata(fitx)
                fitfiber.set_ydata(fityf)
                
                fig.canvas.draw()
                
                #reset iv counter
                iv = 0  
                
            #step framecounter
            framecounter +=1
            
        #plot measurement lines
        cv2.line(img,(0,top),(len(img[1]),top),(0,0,255),2)   
        cv2.line(img,(0,bot),(len(img[1]),bot),(0,0,255),2) 
        
        #plot previous frame fps
        cv2.putText(img,r'%s FPS'%perf, (50,1000),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
        #show input image
        scale = 75
        rszx = int(img.shape[1]*scale/100)
        rszy = int(img.shape[0]*scale/100)
        
        imgrsz = cv2.resize(img, (rszx,rszy))
        cv2.imshow('Input Frame',imgrsz)
                
        #wait for loop delay
        
        t3 = time.perf_counter()
        if (loopdelay-int(((t3-t1)*1000))) > 0:
            cv2.waitKey(loopdelay-int(((t3-t1)*1000)))
        else:
            cv2.waitKey(100)
            
        #compute timer 
        t2 = time.perf_counter()
        perf = int(1/(t2-t1))
        
except KeyboardInterrupt:
    
    cv2.destroyAllWindows()
    
#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()
#stop communication
cam.close_device()
print('Done.')
















