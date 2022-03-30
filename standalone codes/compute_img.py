#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:56:55 2022

@author: pablosanchez
"""

import numpy as np
import cv2

def igray(img):
    _gray = np.mean(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    return _gray

def iblue(img):
    _blue = np.mean(cv2.imread(img)[:,:,0])
    return _blue 

def igreen(img):
    _green = np.mean(cv2.imread(img)[:,:,1])
    return _green

def ired(img):
    _red = np.mean(cv2.imread(img)[:,:,2])
    return _red
