# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:15:04 2021

@author: Pablo Otero
"""

from ximea import xiapi
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#create instance for first connected camera 
cam = xiapi.Camera()

#start communication
print('Opening first camera...')
cam.open_device()

#settings
cam.set_exposure(2000)
cam.set_gain(10)
cam.set_imgdataformat('XI_RGB24')

#create instance of Image to store image data and metadata
cap = xiapi.Image()

#start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

#initialize time and general purpose iterator
t0 = time.time()
i = 0

#create empty array pandas dataframe
data = pd.DataFrame()

#initialize graphic windows
#cv2.namedWindow('Input Frame')
#cv2.namedWindow('Processed Frame')

#initialize plot
fig, ax = plt.subplots()
ax.set_xlabel('Pixel Intensity')
ax.set_ylabel('Counts')

#initialize line objects
bins = 256

lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', label='Red')
lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', label='Green')
lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', label='Blue')

ax.set_xlim(0, bins-1)
ax.set_ylim(0,300000)
ax.legend()
plt.ion()
plt.show()

#define empty function for trackbars
def nothing(x):
    pass

#initialize trackbar for array saving
#cv2.createTrackbar('Save Array','Lined Feed',0,1,nothing)

try:
    print('Starting video. Press CTRL+C to exit.')
    while True:    
        
        #get data and pass them from camera to img
        cam.get_image(cap)
        
        #create numpy array with data from camera. Dimensions of the array are 
        #determined by imgdataformat
        img = cap.get_image_data_numpy()
        npixels = np.size(img)/3
        
        #histogram calculation for the 3 RGB channels
        histB = cv2.calcHist([img],[0],None,[256],[0,256])
        histG = cv2.calcHist([img],[1],None,[256],[0,256])
        histR = cv2.calcHist([img],[2],None,[256],[0,256])
        
        #save to pd dataframe
        
        data['bc'] = histB[:,0]
        data['gc'] = histG[:,0]
        data['rc'] = histR[:,0]
        
        #plot
        lineR.set_ydata(data.rc)
        lineG.set_ydata(data.gc)
        lineB.set_ydata(data.bc)
        
        fig.canvas.draw()
               
        #show input image
        cv2.imshow('Input Image',img)
        
        cv2.waitKey(100)
        
except KeyboardInterrupt:
    
    cv2.destroyAllWindows()
    
#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()
#stop communication
cam.close_device()
print('Done.')
















