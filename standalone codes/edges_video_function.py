# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:46:44 2021

@author: po-po
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from im2D import *

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
filename = r'C:\Users\po-po\Desktop\DOC\Fibras\validation_cases\videos\e05dr5rt120.avi'
cap = cv2.VideoCapture(filename)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  
#initialize time to display diameter
t0 = time.time()
i = 0

#create empty array for diameter saving
diam_arr = np.zeros((1000000,2))

#initialize graphic windows
cv2.namedWindow('Processed Frame')
cv2.namedWindow('Lined Feed')

#define empty function for trackbars
def nothing(x):
    pass

#iniatilize trackbar for P parameter
cv2.createTrackbar('P Selector','Processed Frame',8,100,nothing)

#initialize trackbar for array saving
cv2.createTrackbar('Save Array','Lined Feed',0,1,nothing)

# Read until video is completed

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  
  if ret == True:

    # Display the resulting frame
    orgimg = frame
    img = frame

    #image thresholding
    P = cv2.getTrackbarPos('P Selector', 'Processed Frame');
    
    coord,edgesblur = im2D(img,P)
    coord = coord.astype(int)
    
    for i in range(len(coord)):
   
        cv2.line(edgesblur,(coord[i][0],coord[i][1]),(coord[i][2],coord[i] [3]),(255,255,255),1)
    
    #add center line of fiber diameter  
    x3 = int(len(img[0])/2)
    y3 = int((coord[0][1]+coord[0][3])/2)
    y4 = int((coord[1][1]+coord[1][3])/2)  
    cv2.line(img,(x3,y3),(x3,y4),(0,0,255),2)
    
    #compute fiber diameter and show
    
    fiber_diam_pixels = (y3-y4)
    fiber_diam_micras = str(np.round(203/464 * fiber_diam_pixels, decimals = 0))
    cv2.putText(img,'Fiber Diameter = %s micron'% fiber_diam_micras, (50,1000),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
    #cv2.putText(img,'e05dr5rt120', (50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
    
    #save fiber diameter to array if saved array flag is 1
    save_flag = cv2.getTrackbarPos('Save Array', 'Lined Feed');
    
    if save_flag == 1:
        
        diam_arr[i,0] = time.time()-t0
        diam_arr[i,1] = fiber_diam_micras
        i += 1
        if i == len(diam_arr):
            i = 0
        
    # resize images and show
    
    scale = 50
    rszx = int(img.shape[1]*scale/100)
    rszy = int(img.shape[0]*scale/100)
    
    imgrsz = cv2.resize(img, (rszx,rszy))
    edgesrsz = cv2.resize(edgesblur, (rszx, rszy))
    framersz = cv2.resize(frame, (rszx, rszy))
    
    cv2.imshow('Processed Frame',edgesrsz)
    cv2.imshow('Lined Feed',imgrsz)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
  
     
    # Press P on keyboard to pause
    if cv2.waitKey(100) & 0xFF == ord('p'):
      cv2.waitKey(5000)
    
    
  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()


#delete 0 values from array and transform to pandas dataframe
mod_diam_arr = diam_arr[diam_arr[:,1] != 0]
clean_arr = pd.DataFrame(data = mod_diam_arr, columns = ['Time','Diameter'])

#perform rolling average on pandas dataframe of clean data
interval = 25

clean_arr['Average'] = clean_arr['Diameter'].rolling(window = interval, center = True, min_periods = 1).mean()
clean_arr['Std'] = clean_arr['Diameter'].rolling(window = interval, center = True, min_periods = 1).std()

clean_arr['Clean'] = clean_arr.Diameter[(clean_arr['Diameter'] >= clean_arr['Average']-clean_arr['Std']) & (clean_arr['Diameter'] <= clean_arr['Average']+clean_arr['Std'])]
clean_arr['Dirty'] = clean_arr.Diameter[(clean_arr['Diameter'] <= clean_arr['Average']-clean_arr['Std']) | (clean_arr['Diameter'] >= clean_arr['Average']+clean_arr['Std'])]

#plot diameter array
plt.plot(clean_arr['Time'],clean_arr['Dirty'],'rx')
plt.plot(clean_arr['Time'],clean_arr['Clean'],'kx')
plt.plot(clean_arr['Time'],clean_arr['Average'],'b-')
plt.plot(clean_arr['Time'],clean_arr['Average']-clean_arr['Std'],'r--')
plt.plot(clean_arr['Time'],clean_arr['Average']+clean_arr['Std'],'r--')
plt.show()

#save array to csv, plot file

filename = input('Input filename: ')
clean_arr.to_csv('%s.csv'%filename)
plt.savefig(filename+'_plot.jpg')
