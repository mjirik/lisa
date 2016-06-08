# ! /usr/bin/python
# -*- coding: utf-8 -*-


# import funkcí z jiného adresáře
import sys
import os.path

# path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest

import numpy as np
from nose.plugins.attrib import attr


from lisa import organ_segmentation
import lisa.dataset
import lisa.lisa_data


# nosetests tests/organ_segmentation_test.py:OrganSegmentationTest.test_create_iparams # noqa


class OrganSegmentationTest(unittest.TestCase):

    def generate_data(self):

        img3d = (np.random.rand(30, 30, 30)*10).astype(np.int16)
        seeds = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation[10:25, 4:24, 2:16] = 1
        img3d = img3d + segmentation*20
        seeds[12:18, 9:16, 3:6] = 1
        seeds[19:22, 21:27, 19:21] = 2

        voxelsize_mm = [5, 5, 5]
        metadata = {'voxelsize_mm': voxelsize_mm}
        return img3d, metadata, seeds, segmentation

    @unittest.skip("in progress")
    def test_sync_paul(self):
        """
        sync with paul account
        """

        # gcparams = {'pairwiseAlpha':10, 'use_boundary_penalties':True}
        oseg = organ_segmentation.OrganSegmentation(None)
        # oseg.add_seeds_mm([120], [120], [400], label=1, radius=30)
        # oseg.add_seeds_mm([170, 220, 250], [250, 280, 200], [400], label=2,
        #                   radius=30)

        # "boundary penalties"
        # oseg.interactivity()
        # oseg.ninteractivity()

        # volume = oseg.get_segmented_volume_size_mm3()

        # misc.obj_to_file(oseg.get_iparams(),'iparams.pkl', filetype='pickle')
        oseg.sync_lisa_data('paul','P4ul')


        file_path = os.path.join(lisa.lisa_data, 'sync','paul', 'from_server', 'test.txt')
        self.assertTrue(os.path.exists(file_path))
    # @unittest.skipIf(not interactiveTest, "interactive test")

if __name__ == "__main__":
    unittest.main()
