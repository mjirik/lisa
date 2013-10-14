#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
import copy

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
#sys.path.append(os.path.join(path_to_script, "../extern/py3DSeedEditor/"))
#sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest


from PyQt4.QtGui import QFileDialog, QApplication, QMainWindow

import numpy as np
import itk
import SimpleITK as sitk


import pycut


#
import dcmreaddata as dcmr
import seed_editor_qt

print sys.argv
class PycutTest(unittest.TestCase):
    interactivetTest = False
    #interactivetTest = True

    def generate_data(self, shp=[128,128,128] ):
        """ Generating random data with cubic object inside"""

        data = np.zeros(shp)
        data [30:35,40:47,20:80] = 1
        data [28:34, 40:90, 50:60] = 1
        data [20:65, 40:45, 40:48] = 1

# inserting box

        #x_noisy = x + np.random.normal(0, 0.6, size=x.shape)
        return data

    def test_segmentation_with_boundary_penalties(self):
        data = generate_data()
        dataskel_itk = sitk.GetImageFromArray(data)

        itk.BinaryThinningImageFilter
        import pdb; pdb.set_trace()

:
        pixelType = itk.UC
        imageType = itk.Image[pixelType, 3]
        readerType = itk.BinaryThinningImageFilter[imageType]
        writerType = itk.ImageFileWriter[imageType]
        reader = readerType.New()
        writer = writerType.New()
        reader.SetFileName( argv[1] )
        writer.SetFileName( argv[2] )
        writer.SetInput( reader.GetOutput() )
        writer.Update()

    
    @unittest.skipIf(not interactivetTest, 'interactiveTest')
    def test_se(self):
        pass


if __name__ == "__main__":
    unittest.main()
