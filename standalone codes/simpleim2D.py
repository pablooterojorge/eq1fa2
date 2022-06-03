# -*- coding: utf-8 -*-

def simpleim2D(img,direction):
    
    import numpy as np
    from scipy.signal import argrelextrema 
    
    #simple border search
    if direction == 0:
        #get center slice as array
        slice_center = img[:,int(np.size((img,1))/2),1]
        y3 = 0
        y4 = 0
    elif direction == 1:
        #get center slice as array
        slice_center = img[int(np.size((img,0))/2),:,1]
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
        
    return y3, y4

