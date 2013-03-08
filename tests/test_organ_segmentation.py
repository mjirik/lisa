#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest

import numpy as np


import organ_segmentation
import dcmreaddata1 as dcmr

class OrganSegmentationTest(unittest.TestCase):

    def aaatest_whole_organ_segmentation_interactive(self):
        """
        Function uses organ_segmentation object for segmentation
        """
        dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
        oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4)
        
# manual seeds setting
        print ("with left mouse button select some pixels of the brain")
        print ("with right mouse button select some pixels of other tissues and background")

        oseg.interactivity()

        volume = oseg.get_segmented_volume_size_mm3()
        print volume

        self.assertGreater(volume,50000)
        self.assertLess(volume,100000)


#        roi_mm = [[3,3,3],[150,150,50]]
#        oseg.ni_set_roi()
#        coordinates_mm = [[110,50,30], [10,10,10]]
#        label = [1,2]
#        radius = [5,5]
#        oseg.ni_set_seeds(coordinates_mm, label, radius)
#
#        oseg.make_segmentation()

    def test_box_segmentation(self):
        """
        Function uses organ_segmentation object for segmentation
        """
        #dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
        img3d = np.random.rand(32,32,16) * 5
        img3d[12:22,5:15,4:14] = img3d [12:22,5:15,4:14] + 10
        seeds = np.zeros([32,32,16])
        seeds [13:20,6:7,10] = 1
        seeds [8,1:2,5:20] = 2
#[mm]  10 x 10 x 10
        #voxelsize_mm = [1,4,3]
        voxelsize_mm = [10,10,10]
        metadata = {'voxelsizemm': voxelsize_mm}

        oseg = organ_segmentation.OrganSegmentation(None, data3d=img3d, metadata = metadata, working_voxelsize_mm = 20)
        

# TODO seeedy
        # oseg.seeds = seeds
        #oseg.make_gc()
# manual seeds setting
        print ("with left mouse button select some pixels of the brain")
        print ("with right mouse button select some pixels of other tissues and background")

        oseg.interactivity()

        volume = oseg.get_segmented_volume_size_mm3()
        
        import pdb; pdb.set_trace()

        #mel by to být litr. tedy milion mm3
        self.assertGreater(volume,900000)
        self.assertLess(volume,110000)

    def test_volume_resize(self):

        pass




if __name__ == "__main__":
    unittest.main()
