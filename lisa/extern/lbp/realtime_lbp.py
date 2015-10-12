"""@package realtime LBP

 verison 0.4.0
 Author : Petr Neduchal
 Date : 2013
"""

from PIL import Image
import ctypes
import numpy as np
import scipy

# Function loads dynamic library
def loadRealtimeLbpLibrary():
    """Function loads LBP library

			 libRealtimeLbp.dll/libRealtimeLbp.so (win/lin)
    """
    lbplib = ctypes.cdll.LoadLibrary("libRealtimeLbp.so")
    return lbplib


def realTimeLbp(lbplib, filename):
    """Function calls extern function realTimeLbp 

	     Params : lbplib - library pointer (see loadLbpLibrary)
                 filename - (string) path to the image file. 
    """
    im = Image.open(filename)
    w, h = im.size
    img = (ctypes.c_long * (w * h))()
    res = (ctypes.c_long * 256)()
    for i in range(w):
        for j in range(h):
            img[(w * i) + j] = im.getpixel((i, j))
    lbplib.realTimeLbp(w, h, ctypes.byref(img), ctypes.byref(res))
    return res


def realTimeLbpIm(lbplib, im):
    """Function calls extern function realTimeLbp 

	Params : lbplib - library pointer (see loadLbpLibrary)
                 im - Instance of Image class from PIL module
    """
    w, h = im.size
    img = (ctypes.c_long * (w * h))()
    res = (ctypes.c_long * 256)()
    for i in range(w):
        for j in range(h):
            img[(w * i) + j] = im.getpixel((i, j))
    lbplib.realTimeLbp(w, h, ctypes.byref(img), ctypes.byref(res))
    return res


def realTimeLbpImNp(lbplib, npIM):
    """Function calls extern function realTimeLbpImNp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 npIM - NumPy matrix
    """
    w = npIM.shape[0]
    h = npIM.shape[1]
    img = (ctypes.c_long * (w * h))()
    res = (ctypes.c_long * 256)()
    res2 = np.zeros([1, 256])
    for i in range(w):
        for j in range(h):
            img[(w * i) + j] = npIM[i, j]
    lbplib.realTimeLbp(w, h, ctypes.byref(img), ctypes.byref(res))
    for i in range(256):
        res2[0, i] = res[i]
    return res2


def realTimeLbpIm2ImNp(lbplib, npIM):
    """Function calls extern function realTimeLbpImNp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 npIM - NumPy matrix
    """
    w = npIM.shape[0]
    h = npIM.shape[1]
    img = (ctypes.c_long * (w * h))()
    res = (ctypes.c_long * (w * h))()
    for i in range(w):
        for j in range(h):
            img[(w * i) + j] = npIM[i, j]
    lbplib.realTimeLbpIm(w, h, ctypes.byref(img), ctypes.byref(res))
    for i in range(w):
        for j in range(h):
            res2[i, j] = res[(w * i) + j]
    return res2


def realTimeLbpArr(lbplib, data, w, h):
    """Function calls extern function realTimeLbp 

	Params : lbplib - library pointer (see loadLbpLibrary)
                 data - (ctypes.c_long * (size of the image)) C type array 
                 with Image data
                 w - image width
                 h - image height
    """
    res = (ctypes.c_long * 256)()
    lbplib.realTimeLbp(w, h, ctypes.byref(data), ctypes.byref(res))
    return res


def lbp2Hists(lbplib, IM):
    """Function calls extern function realTimeLbpImNp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 npIM - NumPy matrix
    """
    w = npIM.shape[0]
    h = npIM.shape[1]
    img = (ctypes.c_long * (w * h))()
    res = (ctypes.c_long * (w * h))()
    for i in range(w):
        for j in range(h):
            img[(w * i) + j] = npIM[i, j]
    lbplib.lbp2Hists(w, h, ctypes.byref(img), ctypes.byref(res))
    return res


def lbp2HistsNp(lbplib, npIM):
    """Function calls extern function realTimeLbpImNp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 npIM - NumPy matrix
    """
    w = npIM.shape[0]
    h = npIM.shape[1]
    img = (ctypes.c_long * (w * h))()
    res = (ctypes.c_long * (w * h))()
    for i in range(w):
        for j in range(h):
            img[(w * i) + j] = npIM[i, j]
    lbplib.lbp2Hists(w, h, ctypes.byref(img), ctypes.byref(res))
    for i in range(w):
        for j in range(h):
            res2[i, j] = res[(w * i) + j]
    return res2
