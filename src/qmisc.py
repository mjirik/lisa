#! /usr/bin/python
# -*- coding: utf-8 -*-



import sys
sys.path.append("../extern/pycat/")
sys.path.append("../extern/pycat/extern/py3DSeedEditor/")

import numpy as np



class SparseMatrix():
    def __init__(self, ndarray):
        self.coordinates = ndarray.nonzero()
        self.shape = ndarray.shape
        self.values = ndarray[self.coordinates]
        self.dtype = ndarray.dtype
        self.sparse = True


    def todense(self):
        dense = np.zeros(self.shape, dtype=self.dtype)
        dense[self.coordinates[:]]= self.values
        return dense


def isSparseMatrix(obj):
    if obj.__class__.__name__ == 'SparseMatrix':
        return True
    else:
        return False


import py3DSeedEditor
def manualcrop(data):

    import seed_editor_qt
    pyed = seed_editor_qt.QTSeedEditor(data, mode='crop')
    pyed.exec_()
    #pyed = py3DSeedEditor.py3DSeedEditor(data)
    #pyed.show()
    nzs =  pyed.seeds.nonzero()
    crinfo = [
            [np.min(nzs[0]), np.max(nzs[0])],
            [np.min(nzs[1]), np.max(nzs[1])],
            [np.min(nzs[2]), np.max(nzs[2])],
            ]
    data = crop(data,crinfo)
    return data, crinfo


def crop(data, crinfo):
    return data[
            crinfo[0][0]:crinfo[0][1],
            crinfo[1][0]:crinfo[1][1],
            crinfo[2][0]:crinfo[2][1]
            ]

def uncrop(data, crinfo, orig_shape):
    data_out = np.zeros(orig_shape, dtype=data.dtype)


    print crinfo
    print orig_shape
    print data.shape

    startx = np.round(crinfo[0][0]).astype(int)
    starty = np.round(crinfo[1][0]).astype(int)
    startz = np.round(crinfo[2][0]).astype(int)
    
    data_out [
            #np.round(crinfo[0][0]).astype(int):np.round(crinfo[0][1]).astype(int)+1,
            #np.round(crinfo[1][0]).astype(int):np.round(crinfo[1][1]).astype(int)+1,
            #np.round(crinfo[2][0]).astype(int):np.round(crinfo[2][1]).astype(int)+1
            startx:startx + data.shape[0],
            starty:starty + data.shape[1],
            startz:startz + data.shape[2]
            ] = data

    return data_out

