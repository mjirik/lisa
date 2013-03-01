#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/"))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest
import organ_segmentation
#import dcmreaddir

class OrganSegmentationTest(unittest.TestCase):

    def test_whole_organ_segmentation(self):
        """
        Function uses organ_segmentation object for segmentation
        """
        dcmdir = './../sample_data/matlab/examples/sample_data/DICOM/digest_article/'
        oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4)

        oseg.interactivity()

        roi_mm = [[3,3,3],[150,150,50]]
        oseg.ni_set_roi()
        coordinates_mm = [[110,50,30], [10,10,10]]
        label = [1,2]
        radius = [5,5]
        oseg.ni_set_seeds(coordinates_mm, label, radius)

        oseg.make_segmentation()

    def test_volume_resize(self):

        pass




if __name__ == "__main__":
    unittest.main()
