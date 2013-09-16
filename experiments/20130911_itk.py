#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src"))

import itk
import SimpleITK as sitk

import datareader

data_path = os.path.join(path_to_script, '../sample_data/jatra_5mm/')




print 'im1: 2D dicom image with SimpleITK'
im1 = sitk.ReadImage(os.path.join(path_to_script,"../sample_data/jatra_5mm/IM-0001-0005.dcm"))
pimsh = sitk.Show(im1)
import pdb; pdb.set_trace()

#--------------------------------------------
print 'im2: 3D dicom image readed with our DataReader and visualization with SimpleITK'
dr = datareader.DataReader()
data3d, metadata = dr.Get3DData(data_path)

im2 = sitk.GetImageFromArray(data3d)
sitk.Show(im2)

import pdb; pdb.set_trace()

#--------------------------------------------
print 'im3: 3D dicom image read and visualization with SimpleITK'
isr = sitk.ImageSeriesReader()
seriesids = isr.GetGDCMSeriesIDs(data_path)
print seriesids
#dcmnames = isr.GetGDCMSeriesFileNames(data_path )
dcmnames = isr.GetGDCMSeriesFileNames(data_path, seriesids[0], True, True, True, True)
isr.SetFileNames(dcmnames)
im3 = isr.Execute()
sitk.Show(im3)


import pdb; pdb.set_trace()
