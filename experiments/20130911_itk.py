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



import pdb; pdb.set_trace()

imraw = sitk.ReadImage(os.path.join(path_to_script,"../sample_data/jatra_5mm/IM-0001-0005.dcm"))
pimsh = sitk.Show(imraw)
import pdb; pdb.set_trace()


dr = datareader.DataReader()
data3d, metadata = dr.Get3DData(data_path)

im3 = sitk.GetImageFromArray(data3d)
sitk.Show(im3)


isr = sitk.ImageSeriesReader()
seriesids = isr.GetGDCMSeriesIDs(data_path)
print seriesids
#dcmnames = isr.GetGDCMSeriesFileNames(data_path )
dcmnames = isr.GetGDCMSeriesFileNames(data_path, seriesids[0]), True, True, True, True

isr.SetFileNames(dcmnames)
import pdb; pdb.set_trace()
