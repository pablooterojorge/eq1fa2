from ximea import xiapi
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from simpleim2D import *

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
#initialize time to display diameter
t0 = time.time()
i = 0
igraph = 0
#create empty array for diameter saving
diam_arr = np.zeros((100000000,3))
#initialize matplotlib image

#initialize graphic windows
cv2.namedWindow('Video')
cv2.namedWindow('Trackbars',cv2.WINDOW_NORMAL)
#cv2.namedWindow('Lined Feed')
#define empty function for trackbars
def nothing(x):
    pass
#iniatilize trackbar for P parameter
#cv2.createTrackbar('P Selector','Processed Frame',8,100,nothing)
#initialize trackbar for array saving
cv2.createTrackbar('Save Array','Trackbars',0,1,nothing)
#iniatilize trackbar for event positioning
cv2.createTrackbar('Event Location','Trackbars',0,1,nothing)

#initialize graphing window

fig, ax = plt.subplots()

ax.set_xlabel('Time (s)')
ax.set_ylabel('Fiber Diameter (um)')

points, = ax.plot(0, 0, 'kx', label='Diameter')
mean, = ax.plot(0, 0, 'b-', label='Rolling Avg')
flag, = ax.plot(0, 0, 'go', label='Events')

ax.set_xlim(0, 3600)
ax.set_ylim(0,500)

ax.legend()
plt.ion()
plt.show()


try:
    print('Starting video. Press CTRL+C to exit.')
    while True:    
        #get data and pass them from camera to img
        cam.get_image(cap)
        #create numpy array with data from camera. Dimensions of the array are 
        #determined by imgdataformat
        img = cap.get_image_data_numpy()
        
        y3, y4 = simpleim2D(img)          
        
        #add center line of fiber diameter
        x3  = int(len(img[0])/2)
        cv2.line(img,(x3,y3),(x3,y4),(0,0,255),2)
        
        #compute fiber diameter and show
        fiber_diam_pixels = (y4-y3)
        fiber_diam_micras = str(np.round(203/464 * fiber_diam_pixels, decimals = 0))
        cv2.putText(img,r'Fiber Diameter = %s um'% fiber_diam_micras, (50,1000),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        
        #save fiber diameter to array if saved array flag is 1
        save_flag = cv2.getTrackbarPos('Save Array', 'Trackbars');
        event_flag = cv2.getTrackbarPos('Event Location', 'Trackbars')
        
        if save_flag == 1:
        
            diam_arr[i,0] = time.time()-t0
            diam_arr[i,1] = fiber_diam_micras
            diam_arr[i,2] = event_flag
            i += 1
            if i == len(diam_arr):
                i = 0
                
        # resize images and show
        scale = 75
        rszx = int(img.shape[1]*scale/100)
        rszy = int(img.shape[0]*scale/100)
        
        imgrsz = cv2.resize(img, (rszx,rszy))

        #cv2.imshow('Processed Frame',edgesrsz)
        cv2.imshow('Video',imgrsz)
        
        #data analysis and plotting
        if igraph == 10:
            
            #delete 0 values from array and transform to pandas dataframe
            mod_diam_arr = diam_arr[diam_arr[:,1] != 0]
            clean_arr = pd.DataFrame(data = mod_diam_arr, columns = ['Time','Diameter','Event Flag'])
    
            #perform rolling average on pandas dataframe of clean data
            interval = 20
    
            clean_arr['Average'] = clean_arr['Diameter'].rolling(window = interval, center = True, min_periods = 1).mean()
            clean_arr['Std'] = clean_arr['Diameter'].rolling(window = interval, center = True, min_periods = 1).std()
    
            clean_arr['Clean'] = clean_arr.Diameter[(clean_arr['Diameter'] >= clean_arr['Average']-clean_arr['Std']) & (clean_arr['Diameter'] <= clean_arr['Average']+clean_arr['Std'])]
            clean_arr['Dirty'] = clean_arr.Diameter[(clean_arr['Diameter'] <= clean_arr['Average']-clean_arr['Std']) | (clean_arr['Diameter'] >= clean_arr['Average']+clean_arr['Std'])]
    
            clean_arr['CAverage'] = clean_arr['Clean'].rolling(window = interval, center = True, min_periods = 1).mean()
            clean_arr['Marked'] = clean_arr.Time[clean_arr['Event Flag'] == 1]
    
            #plot diameter array
            points.set_xdata(clean_arr['Time'])
            points.set_ydata(clean_arr['Clean'])
            mean.set_xdata(clean_arr['Time'])
            mean.set_ydata(clean_arr['CAverage'])
            flag.set_xdata(clean_arr['Marked'])
            flag.set_ydata(clean_arr['Event Flag'])
            
            fig.canvas.draw()
            igraph = 0
            
        cv2.waitKey(80)
        igraph = igraph + 1
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    
    
#delete 0 values from array and transform to pandas dataframe
mod_diam_arr = diam_arr[diam_arr[:,1] != 0]
clean_arr = pd.DataFrame(data = mod_diam_arr, columns = ['Time','Diameter','Event Flag'])

#perform rolling average on pandas dataframe of clean data
interval = 20

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

plt.show()


#save array to csv, plot file

filename = input('Input filename: ')
clean_arr.to_csv('%s.csv'%filename)
plt.savefig(filename+'_plot.jpg')

#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()
#stop communication
cam.close_device()
print('Done.')