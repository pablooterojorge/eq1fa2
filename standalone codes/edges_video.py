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

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
filename = r'C:\Users\po-po\Desktop\DOC\Fibras\videos\e05dr5rt120.avi'
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

    img = frame
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #image thresholding
    P = cv2.getTrackbarPos('P Selector', 'Processed Frame');
    retval,imgthresh = cv2.threshold(gray,P,255,cv2.THRESH_BINARY)
    
    # plt.imshow(imgthresh)
    
    #image edging
    edges = cv2.Canny(imgthresh,254,255)
    edgesblur  = cv2.GaussianBlur(edges,(5,5),0)
    # plt.imshow(edgesblur)
    # plt.imshow(edges)
    
    #apply hough transform
    lines = cv2.HoughLines(edgesblur,1,np.pi/60,500)
    
    #avoid NONE array when no lines are detected
    if lines is None :
        lines = np.ones((2,2))
    
    #reshape array for calculation
    lines = np.reshape(lines,(len(lines),2))
    
    #select appropiate lines for fiber edge
    lines_selected = np.zeros((2,2))
    lines_selected[0] = lines.max(0)
    lines_selected[1] = lines.min(0)
    
    #represent lines overimposed on original img.
    for rho,theta in lines_selected:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + len(img[0])*(-b))
        y1 = int(y0 + len(img[1])*(a))
        x2 = int(x0 - len(img[0])*(-b))
        y2 = int(y0 - len(img[1])*(a))
    
        cv2.line(edgesblur,(x1,y1),(x2,y2),(255,255,255),1)

    #add center line of fiber diameter
    x3  = int(len(img[0])/2)
    y3 = int(-np.cos(lines_selected[0,1])/np.sin(lines_selected[0,1])*x3 + lines_selected[0,0]/np.sin(lines_selected[0,1]))
    y4 = int((-np.cos(lines_selected[1,1])/(np.sin(lines_selected[1,1])+0.01)*x3 + lines_selected[1,0]/(np.sin(lines_selected[1,1])+0.01)))
            
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
