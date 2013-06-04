#! /usr/bin/python
# -*- coding: utf-8 -*-
"""@package lbpLibrary

 verison 0.3.0
 Author : Petr Neduchal
 Date : 2012
"""

import sys
import os.path

from PIL import Image
import ctypes
import numpy as np





# Function loads dynamic library
def loadLbpLibrary() :
    """Function loads LBP library
    
	Lbp.dll/liblbp.so (win/lin)
    """
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    print path_to_script
    lbplib = ctypes.cdll.LoadLibrary(os.path.join(path_to_script, "liblbp.so"))
    return lbplib
 
def realTimeLbp(lbplib, filename) :
    """Function calls extern function realTimeLbp 

	Params : lbplib - library pointer (see loadLbpLibrary)
                 filename - (string) path to the image file. 
    """
    im = Image.open(filename)
    w,h = im.size
    img = (ctypes.c_long * (w*h))()
    res = (ctypes.c_long * 256)()
    for i in range(w) :
        for j in range(h) :
            img[(w*i) + j] = im.getpixel((i, j))
    lbplib.realTimeLbp(w,h,ctypes.byref(img), ctypes.byref(res))
    return res

def realTimeLbpIm(lbplib, im) :
    """Function calls extern function realTimeLbp 

	Params : lbplib - library pointer (see loadLbpLibrary)
                 im - Instance of Image class from PIL module
    """
    w,h = im.size
    img = (ctypes.c_long * (w*h))()
    res = (ctypes.c_long * 256)()
    for i in range(w) :
        for j in range(h) :
            img[(w*i) + j] = im.getpixel((i, j))
    lbplib.realTimeLbp(w,h,ctypes.byref(img), ctypes.byref(res))
    return res

def realTimeLbpImNp(lbplib, npIM) :
    """Function calls extern function realTimeLbpImNp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 npIM - NumPy matrix
    """
    h = npIM.shape[0]
    w = npIM.shape[1]
    img = (ctypes.c_long * (w*h))()
    res = (ctypes.c_long * 256)()
    for i in range(h) :
        for j in range(w) :
            img[(w*i) + j] = npIM[i,j]
    lbplib.realTimeLbp(w,h,ctypes.byref(img), ctypes.byref(res))
    return res

def realTimeLbpIm2ImNp(lbplib, npIM) :
    """Function calls extern function realTimeLbpImNp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 npIM - NumPy matrix
    """
    h = npIM.shape[0]
    w = npIM.shape[1]
    img = (ctypes.c_long * ((w+1)*(h+1)))()
    res = (ctypes.c_long * ((w+1)*(h+1)))()
    for i in range(h) :
        for j in range(w) :
            img[(w*i) + j] = npIM[i,j]            
    lbplib.realTimeLbpIm(w,h,ctypes.byref(img), ctypes.byref(res))
    res2 = np.zeros([h,w])
    for i in range(h) :
        for j in range(w) :
            res2[i,j] = res[(w*i) + j]
    return res2

def realTimeLbpArr(lbplib, data, w, h) :
    """Function calls extern function realTimeLbp 

	Params : lbplib - library pointer (see loadLbpLibrary)
                 data - (ctypes.c_long * (size of the image)) C type array 
                 with Image data
                 w - image width
                 h - image height
    """
    res = (ctypes.c_long * 256)()
    lbplib.realTimeLbp(w,h,ctypes.byref(data), ctypes.byref(res))
    return res


def lbp2Hists(lbplib, npIM ):
    """Function calls extern function realTimeLbpImNp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 npIM - NumPy matrix
    """
    w = npIM.shape[0]
    h = npIM.shape[1]
    img = (ctypes.c_long * ((w+1)*(h+1)))()
    res = (ctypes.c_long * ((w+1)*(h+1)))()
    for i in range(h) :
        for j in range(w) :
            img[(w*i) + j] = int(npIM[i,j])
    lbplib.lbp2Hists(h,w,ctypes.byref(img), 16, 16, ctypes.byref(res))
    return res

def lbp2HistsNp(lbplib, npIM):
    """Function calls extern function realTimeLbpImNp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 npIM - NumPy matrix
    """
    h = npIM.shape[0]
    w = npIM.shape[1]
    img = (ctypes.c_long * ((w+1)*(h+1)))()
    res = (ctypes.c_long * ((w+1)*(h+1)))()
    for i in range(h) :
        for j in range(w) :
            img[(w*i) + j] = int(npIM[i,j])
    lbplib.lbp2Hists(h,w,ctypes.byref(img), 16, 16, ctypes.byref(res))
    numHist = (((h)/16)*((w)/16))
    res2 = np.zeros([numHist,256])
    for i in range(numHist) :
        for j in range(256) :
            res2[i,j] = res[(256*i) + j]
    return res2


def imageToLbp(lbplib, filename, type = 1, radius = 1, samples = 8) :
    """Function calls extern function imageToLbp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 filename - (string) path to the image file. 
                 type -  Type of LBP algorithm
                 radius - Radius of samples in LBP mask.
                 samples - Number of samples around the main point in LBP mask
    """
    im = Image.open(filename)
    w,h = im.size
    img = (ctypes.c_double * (w*h))()
    for i in range(w) :
        for j in range(h) :
            img[(w*i) + j] = im.getpixel((i, j))
    lbplib.imageToLbp(w,h, ctypes.byref(img), type, radius, samples)
    return img

# Function calls extern function imageToLbp
# Params : lbplib - library pointer (see loadLbpLibrary)
#          im - Instance of Image class from PIL module
#          type -  Type of LBP algorithm
#          radius - Radius of samples in LBP mask.
#          samples - Number of samples around the main point in LBP mask
def imageToLbpIm(lbplib, im, type = 1, radius = 1, samples = 8) :
    """Function calls extern function imageToLbp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 im - Instance of Image class from PIL module
                 type -  Type of LBP algorithm
                 radius - Radius of samples in LBP mask.
                 samples - Number of samples around the main point in LBP mask
    """
    w,h = im.size
    img = (ctypes.c_double * (w*h))()
    for i in range(w) :
        for j in range(h) :
            img[(w*i) + j] = im.getpixel((i, j))
    lbplib.imageToLbp(w,h, ctypes.byref(img), type, radius, samples)
    return img

# Function calls extern function imageToLbp
# Params : lbplib - library pointer (see loadLbpLibrary)
#          data - (ctypes.c_long * (size of the image)) C type array 
#                 with Image data
#          w - image width
#          h - image height
#          type -  Type of LBP algorithm
#          radius - Radius of samples in LBP mask.
#          samples - Number of samples around the main point in LBP mask
def imageToLbpArr(lbplib, data, w, h, type = 1, radius = 1, samples = 8) :
    """Function calls extern function imageToLbp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 with Image data
                 w - image width
                 h - image height
                 type -  Type of LBP algorithm
                 radius - Radius of samples in LBP mask.
                 samples - Number of samples around the main point in LBP mask
    """
    lbplib.imageToLbp(w,h, ctypes.byref(data), type, radius, samples)
    return img
    
