"""
Created on Tue Mar  1 09:55:26 2022

@author: po-po
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from simpleim2D import *

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
filename = r'C:\Users\po-po\Desktop\DOC\Fibras\validation_cases\videos\e05dr2rt120.avi'
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
cv2.namedWindow('Lined Feed')

#define empty function for trackbars
def nothing(x):
    pass

#iniatilize trackbar for P parameter
#cv2.createTrackbar('Tol Selector','Lined Feed',33,100,nothing)

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

    #simple border search
    #get center slice as array
    slice_center = img[:,int(np.size((img,1))/2),1]
    y3 = 0
    y4 = 0
    
    #compute histogram and get most common value, background brightness
    hist = np.histogram(slice_center,255,[0, 255])[0]
    bg_ii0 = np.argmax(hist)
    
    if bg_ii0 == 0:
        bg_ii0 = 1
        
    #compute change array
    change = np.empty(len(slice_center))
    
    for i in range(len(change)):
        change[i] = np.abs(slice_center[i]-bg_ii0)/(255)
        
    #compute most common change value after bg, this should correlate to fiber region
    histchange = np.histogram(change,100,[0, 1])[0]
    fb_change = np.max(change)

    #top down search for fiber edge
    y3 = np.argmax(change > (33/100*fb_change))
    #flip array and do bottom up search for fiber edge
    change = np.flip(change)
    y4 = len(change) - np.argmax(change > (33/100*fb_change))
     
    #add center line of fiber diameter  
    x3 = int(len(img[0])/2)
    
    cv2.line(img,(x3,y3),(x3,y4),(0,0,255),2)
    
    #compute fiber diameter and show
    
    fiber_diam_pixels = (y4-y3)
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
    framersz = cv2.resize(frame, (rszx, rszy))
    
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


