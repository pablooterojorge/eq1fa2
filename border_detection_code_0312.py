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
#initialize time to display diameter
t0 = time.time()
i = 0
igraph = 0
#create empty array for diameter saving
diam_arr = np.zeros((100000000,3))
#initialize matplotlib image

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
#iniatilize trackbar for event positioning
cv2.createTrackbar('Event Location','Lined Feed',0,1,nothing)

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
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        #image thresholding
        P = cv2.getTrackbarPos('P Selector', 'Processed Frame');
        retval,imgthresh = cv2.threshold(gray,P,255,cv2.THRESH_BINARY)
        # plt.imshow(imgthresh)
        
        #image edging
        edges = cv2.Canny(imgthresh,250,255)
        edgesblur  = cv2.GaussianBlur(edges,(5,5),0)
        # plt.imshow(edgesblur)
        # plt.imshow(edges)
        
        #apply hough transform
        lines = cv2.HoughLines(edgesblur,1,np.pi/60,500)
        if lines is None :
            lines = np.ones((2,2))
            
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
        y3 = int(-np.cos(lines_selected[0,1])/(np.sin(lines_selected[0,1]+0.01))*x3 + lines_selected[0,0]/(np.sin(lines_selected[0,1])+0.01))
        y4 = int((-np.cos(lines_selected[1,1])/(np.sin(lines_selected[1,1])+0.01)*x3 + lines_selected[1,0]/(np.sin(lines_selected[1,1])+0.01)))
                   
        cv2.line(img,(x3,y3),(x3,y4),(0,0,255),2)
        
        #compute fiber diameter and show
        fiber_diam_pixels = (y3-y4)
        fiber_diam_micras = str(np.round(203/464 * fiber_diam_pixels, decimals = 0))
        cv2.putText(img,r'Fiber Diameter = %s um'% fiber_diam_micras, (50,1000),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        
        #save fiber diameter to array if saved array flag is 1
        save_flag = cv2.getTrackbarPos('Save Array', 'Lined Feed');
        event_flag = cv2.getTrackbarPos('Event Location', 'Lined Feed')
        
        if save_flag == 1:
        
            diam_arr[i,0] = time.time()-t0
            diam_arr[i,1] = fiber_diam_micras
            diam_arr[i,2] = event_flag
            i += 1
            if i == len(diam_arr):
                i = 0
                
        # resize images and show
        scale = 50
        rszx = int(img.shape[1]*scale/100)
        rszy = int(img.shape[0]*scale/100)
        
        imgrsz = cv2.resize(img, (rszx,rszy))
        edgesrsz = cv2.resize(edgesblur, (rszx, rszy))
        
        cv2.imshow('Processed Frame',edgesrsz)
        cv2.imshow('Lined Feed',imgrsz)
        
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