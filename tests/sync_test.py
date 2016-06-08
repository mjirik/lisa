# ! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

# import funkcí z jiného adresáře
import sys
import os.path
import shutil

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

    # @unittest.skip("in progress")
    def test_sync_paul(self):
        """
        sync with paul account
        """

        path_to_paul = lisa.lisa_data.path('sync','paul')

        if os.path.exists(path_to_paul):
            shutil.rmtree(path_to_paul)
        oseg = organ_segmentation.OrganSegmentation(None)
        oseg.sync_lisa_data('paul','P4ul')


        file_path = lisa.lisa_data.path('sync','paul', 'from_server', 'test.txt')
        logger.debug('file_path %s', file_path)
        self.assertTrue(os.path.exists(file_path))

if __name__ == "__main__":
    unittest.main()
