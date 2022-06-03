# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:47:40 2022

@author: po-po
"""

from ximea import xiapi
import cv2
import time
import pyvisa 
import numpy as np
import pandas as pd
from scipy import signal
import scipy.optimize as opt
from tqdm import tqdm
import matplotlib.pyplot as plt

np.seterr(invalid='ignore')

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
sdg = rm.open_resource("USB0::0xF4EC::0x1103::SDG1XDCC6R0469::INSTR")

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
cv2.namedWindow('Computed BR')
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
#cv2.namedWindow('Processed Frame')

#define empty function for trackbars
def nothing(x):
    pass

##initialize trackbar for br computation trigger
cv2.createTrackbar('BR Enable','Trackbars',0,1,nothing)
#initalize trackbar for background brightness taking
cv2.createTrackbar('Background ii0','Trackbars',0,1,nothing)
#initialize trackbar for offset of fiber ii0 zone
cv2.createTrackbar('Offset ii0','Trackbars',100,512,nothing)


#create voltage array
#vlist = np.arange(0.1,20.1,0.1)
vlist = np.array([ 0. ,  1.2,  1.6,  2. ,  2.2,  2.4,  2.6,  2.8,  3. ,  3.2,  3.4, 3.4,  3.6,  3.8,  4.2,  4.4,  4.8,  5.4,  6.2,  7. ,  7.8,  8.8, 10. , 11.6, 20. ])

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

#create empty dataframe for data passing to br calculation between iterations
brf = pd.DataFrame(index = range(len(vlist)))
brf['vrms'] = vlist/2
brf['ii0f'] = np.NAN
brf['ii0fn'] = np.NAN
brf['ii0lc'] = np.NAN
brf['ii0lcn'] = np.NAN
brf['retlcn'] = np.NAN
brf['retf'] = np.NAN

#create 3D array for brightness saving on fiber region of xres and yres blocks
yres = 20
xres = 1

ii3Darray = np.zeros((yres,xres,len(vlist)))

#create array for retardance/br saving at the end of iteration
ret2darray = np.zeros((yres,xres))

#initialize mean brightness for fiber
ii0favg = np.zeros((yres,xres));
ii0lcavg = 0;

#initialize fit parameters
paramsfibre = [100,0.3,10]
paramslc = [100,10]
tmaximum = 0
tminimum = 0

#set section to measure
topinit = 512;
bottinit = 512;
off = 20

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
        
        bg_flag = cv2.getTrackbarPos('Background ii0', 'Trackbars')
        if bg_flag == 1:
            #loop arround ivlist once to get all background brightness
            for bgi in range(len(vlist)):
                #set siglent state
                v = str(vlist[bgi])
                sdg.write("C1:BSWV AMP, %s"%v)
                #wait to ensure siglent state is OK
                cv2.waitKey(100)
                #get image
                #get data and pass them from camera to img
                cam.get_image(cap)
                #create numpy array with data from camera. Dimensions of the array are 
                #determined by imgdataformat
                img = cap.get_image_data_numpy()
                #save mean brightness to array
                brf.ii0lc[bgi] = np.mean(img[:,:,1])
                
            #set siglent state
            v = str(vlist[iv])
            sdg.write("C1:BSWV AMP, %s"%v) 
            cv2.waitKey(100)
            #change trackbar pos and flag
            cv2.setTrackbarPos('Background ii0', 'Trackbars', 0)
            bg_flag = cv2.getTrackbarPos('Background ii0', 'Trackbars')
        
        #get offset
        off = cv2.getTrackbarPos('Offset ii0', 'Trackbars')
        top = int(topinit - off)
        bot = int(bottinit + off)
        #save original size
        xsize = np.size(img[top:bot,:,1],1)
        ysize = np.size(img[top:bot,:,1],0)
        
        #check br trigger
        if  cv2.getTrackbarPos('BR Enable', 'Trackbars') == 1:
              
            #get data and pass them from camera to img
            cam.get_image(cap)
            
            #create numpy array with data from camera. Dimensions of the array are 
            #determined by imgdataformat
            img = cap.get_image_data_numpy()
            
            #add mean brightness for fiber
            #resize measurement area to number of blocks with brigthness averaging
            ii0favg_current = cv2.resize(img[top:bot,:,1],(xres,yres),interpolation = cv2.INTER_AREA)
            
            #loop thourhg rsz area
            ii0favg = (ii0favg + ii0favg_current)/(framecounter + 1)
            
            #step framecounter
            framecounter +=1
            
            #check if its time for br computation, avgtime has elapsed
            if framecounter == vframes:
                
                #save current voltage level and average brightness
                brf.vrms[iv] = vlist[iv]
                ii3Darray[:,:,iv] = ii0favg
           
                #reset average values
                ii0favg[:,:] = 0
                
                #set voltage value
                v = str(vlist[iv])
                sdg.write("C1:BSWV AMP, %s"%v)
                cv2.waitKey(5)
                
                #increase iv loop by 1
                iv += 1
                
                #reset framecounter
                framecounter = 0
                
            #check if iv loop is complete
            if iv == len(vlist):
                
                #compute br from saved data for current 
                #loop on y and x direction of area of measurement
                for y in tqdm(range(ii3Darray.shape[0])):
                    for x in range(ii3Darray.shape[1]):
                        brf.ii0f = ii3Darray[y,x,:]
                        #compute normalized data, CHECK NORMALIZATION, IS CORRECT?
                        for i in range(len(brf.ii0f)):
                            brf.ii0fn[i] = (brf.ii0f[i] - np.amin(brf.ii0f))/(np.amax(brf.ii0f)-np.amin(brf.ii0f))
                            brf.ii0lcn[i] = (brf.ii0lc[i] - np.amin(brf.ii0lc))/(np.amax(brf.ii0lc)-np.amin(brf.ii0lc))
                        
                        #find local extrema for cal data in fiber lc, first we check for
                        #local extrema, then we check for true minimum and maximum with vrms
                        
                        maximums = signal.argrelextrema(brf.ii0lc.to_numpy(), np.greater)[0]
                        minimums = signal.argrelextrema(brf.ii0lc.to_numpy(), np.less)[0]
                        
                        for i in range(len(maximums)):
                            if brf.vrms[maximums[i]] >= 3.7 and brf.vrms[maximums[i]] <= 4.7:
                                tmaximum = maximums[i]
                        for i in range(len(minimums)):
                            if brf.vrms[minimums[i]] >= 2.2 and brf.vrms[minimums[i]] <= 3:
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
                        
                        #save to correct position of 2d array
                        ret2darray[y,x] = paramsfibre[1]

                #reset iv counter
                iv = 0  
                #set trackbar to 0
                cv2.setTrackbarPos('BR Enable', 'Trackbars', 0)
                #set voltage value
                v = str(vlist[iv])
                sdg.write("C1:BSWV AMP, %s"%v)
                cv2.waitKey(50)
            
        #plot measurement lines
        cv2.line(img,(0,top),(len(img[1]),top),(0,0,255),2)   
        cv2.line(img,(0,bot),(len(img[1]),bot),(0,0,255),2) 
        
        #plot previous frame fps
        cv2.putText(img,r'%s FPS'%perf, (50,1000),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
        #show input image
        scale = 0.66
        imgrsz = cv2.resize(img, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)
        cv2.imshow('Input Frame',imgrsz)
        
        #show computed retardance
        
        retim = img[:,:,1]
        #apply colormap to retardance data
        #scale retardance array
        #remove non decimal values of retardance mainting sign of decimals. 
        #ambiguity of +-m on retardance due to single wavelength measurement
        ret2darrayo = ret2darray
        for reti in range(ret2darray.shape[0]):
            for retj in range(ret2darray.shape[1]):
                if ret2darray[reti,retj] > 0:
                    ret2darray[reti,retj] = ret2darray[reti,retj] %1
                elif ret2darray[reti,retj] < 0:
                    ret2darray[reti,retj] = -1*(ret2darray[reti,retj] %-1)
        
        #scale ret array for image display
        ret2darray_n = 255*(ret2darray - np.min(ret2darray))/(np.max(ret2darray)-np.min(ret2darray))
        #substitute in img
        retim[top:bot,:] = cv2.resize(ret2darray_n,(xsize,ysize))
        #display img
        retimrsz = cv2.resize(retim,None,fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
        cv2.imshow('Computed BR', retimrsz)        
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


















