#!/usr/bin/env python
# Simple program for ITK image read/write in Python
import itk

import SimpleITK as sitk

import numpy as np

import logging
logger = logging.getLogger(__name__)
from sys import argv





class DataWriter:
    def Write3DData(self, data3d, path, filetype='dcm', metadata=None):
        mtd = {'voxelsize_mm':[1,1,1]}
        if metadata != None:
            mtd.update(metadata)


        if filetype in ['dcm', 'DCM', 'dicom']:
            #pixelType = itk.UC
            #imageType = itk.Image[pixelType, 2]
            dim = sitk.GetImageFromArray(data3d)
            vsz = mtd['voxelsize_mm']
            dim.SetSpacing([vsz[0], vsz[2], vsz[1]])
            sitk.WriteImage(dim,path)


#data = np.zeros([100,100,30], dtype=np.uint8)
#data[20:60,60:70, 0:5] = 100
#dw = DataWriter()
#dw.Write3DData(data, 'soubor.dcm')
