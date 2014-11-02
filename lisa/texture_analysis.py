#! /usr/bin/python
# -*- coding: utf-8 -*-
from PIL import Image
import ctypes
import math
import numpy as np
import scipy as sp
import sed3

import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/lbp/"))

import lbpLibrary as lbpLib



def segmentation(data3d, segmentation, params):
    """ Texture analysis by LBP algorithm.
    data: CT (or MRI) 3D data
    segmentation: labeled image with same size as data where label:
    1 mean liver pixels,
    -1 interesting tissuse (bones)
    0 otherwise
    """
    
    MAXPATTERNS = 10

    slices = data3d.shape[2]
    data3d = data3d.astype(np.int16)
    
    params[2] = 0
    data3d = data3d * segmentation;
    data3d = sp.ndimage.filters.gaussian_filter(data3d,params*2.5)  
    pyed = sed3.sed3(data3d)
    pyed.show()
    
    lbp = lbpLib.loadLbpLibrary()    

    lbpReftmp = (ctypes.c_long * 256)()
    lbpRef = np.zeros([MAXPATTERNS,256])

    actualpatterns = 0

    # vytvoreni referencnich LBP obrazu
    for j in range(15,data3d.shape[0]-16):
        if (actualpatterns == MAXPATTERNS) :
            break
        for k in range(15,data3d.shape[1]-16): 
            if (actualpatterns == MAXPATTERNS) :
                break
            if ((pyed.seeds[j,k,pyed.actual_slice] == 1)):                
                lbpReftmp = lbpLib.realTimeLbpImNp(lbp, data3d[j-15:j+16,k-15:k+16,pyed.actual_slice])
                for z in range(256):
                    lbpRef[actualpatterns, z] = lbpReftmp[z]
                print lbpRef[actualpatterns,]
                actualpatterns += 1                   

    # uprava velikosti dat
    for i in range(20):
        h2 = data3d.shape[0] - i*16
        if (h2 <= 0):
            h2 = i*16
            break
    for i in range(20):
        w2 = data3d.shape[1] - i*16
        if (w2 <= 0):
            w2 = i*16
            break    
    # obdelnikova data prevedeme na ctverec 
    if (w2 >=  h2 ):     
        h2 = w2
    else :
        w2 = h2
    numOfPartsX = (w2) / 16
    numOfPartsY = (h2) / 16   
    minMaxData = np.zeros([h2,w2,data3d.shape[2]],dtype = np.float)
    lbpSlice = np.zeros([h2,w2])
      
    # hlavni cyklus vypocitavajici odezvy
    for i in range(slices):
        tempData = np.zeros([h2,w2],dtype = np.float)
        tempData[0:data3d.shape[0],0:data3d.shape[1]] = data3d[:,:,i].astype(np.float)   
        lbpSlice = lbpLib.realTimeLbpIm2ImNp(lbp, tempData.astype(np.int16))
        lbpRes = lbpLib.lbp2HistsNp(lbp, lbpSlice)
        for j in range(lbpRes.shape[0]):
            minmaxBestVal = 0
            for k in range(MAXPATTERNS):
                minmaxVal = minmax(lbpRes[j,:],lbpRef[k,:])
                if minmaxVal > minmaxBestVal : 
                    minmaxBestVal = minmaxVal
            minMaxData[16*(j/numOfPartsX):16*(j/numOfPartsX)+16,16*(j % numOfPartsX):16*(j % numOfPartsX)+16,i] = minmaxBestVal        

    # zobrazeni vysledku
    pyed2 = sed3.sed3(minMaxData)
    pyed2.show()

    return segmentation

# Funkce pocitajici min max kriterium porovnani dvou histogramu
def minmax(hist1, hist2):
    
    minmaxValue = 0
    minValue = 0
    maxValue = 0
    for i in range(256):
        minValue =  minValue +  min(hist1[i],hist2[i])
        maxValue =  maxValue +  max(hist1[i],hist2[i])
	
    minmaxValue = float(minValue)/float(maxValue+1)
    return minmaxValue



    



