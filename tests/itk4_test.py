#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
import copy

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
#sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
#sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest


from PyQt4.QtGui import QFileDialog, QApplication, QMainWindow

import numpy as np
# import itk
#import SimpleITK as sitk


# from pysegbase import pycut
# from pysegbase import seed_editor_qt

print sys.argv
class PycutTest(unittest.TestCase):
    interactivetTest = False
    #interactivetTest = True

    def generate_data(self, shp=[100,101,102] ):
        """ Generating random data with cubic object inside"""

        data = np.zeros(shp)
        data [30:35,40:47,20:80] = 1
        data [28:34, 40:90, 50:60] = 1
        data [20:65, 40:45, 40:48] = 1

# inserting box

        #x_noisy = x + np.random.normal(0, 0.6, size=x.shape)
        return data

    @unittest.skip("ITK has only 2D skeleton")
    def test_simple_itk(self):
        import SimpleITK as sitk
        data = self.generate_data()
        data_itk = sitk.GetImageFromArray(data)
        output = sitk.BinaryThinning(data_itk)



    @unittest.skip("ITK has only 2D skeleton")
    def test_itk_skeleton_3d(self):
        """
        This is not working. ITK has 2D thinning algorithm only. :-(
        """
        import SimpleITK as sitk
        data = self.generate_data()


        data = data.astype(np.int16)

        #data_itk = sitk.GetImageFromArray(data)
        #shape = data_itk.GetSize()
        shape = data.shape#[::-1]

        imT = itk.Image.SS3
        im = imT.New(Regions=shape)
        im.Allocate()

        import pdb; pdb.set_trace()
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                for k in range(0,shape[2]):
                    pixel = int(data[i,j,k])
                    #print type(pixel)
                    im.SetPixel([i,j,k], pixel)

        #castFilter = itk.CastImageFilter[itk.Image[itk.F,3], itk.Image[itk.SS,3]]



# skeletonization

        pixelType = itk.SS
        imageType = itk.Image[pixelType, 3]
        thinningFilterType = itk.BinaryThinningImageFilter[imageType, imageType]
        thinningFilter = thinningFilterType.New()
        thinningFilter.New()
        thinningFilter.SetInput(im)
        thinningFilter.Update()
        skelim = thinningFilter.GetOutput()


        import pdb; pdb.set_trace()

        dataout = np.zeros(shape, dtype=np.int8)
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                for k in range(0,shape[2]):
                    pixel = skelim.GetPixel([i,j,k])
                    dataout[i,j,k] = pixel


        import pdb; pdb.set_trace()
        data_itk = sitk.GetImageFromArray(dataout)

        import pdb; pdb.set_trace()
        writerType = itk.ImageFileWriter[imageType, imageType]

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
