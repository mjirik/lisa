#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2014 mjirik <mjirik@mjirik-HP-Compaq-Elite-8300-MT>
#
# Distributed under terms of the MIT license.

"""

"""
import unittest
from nose.plugins.attrib import attr

from lisa import liver_segmentation


class LiverSegmentationTest(unittest.TestCase):

    # test se pouští ze složky lisa
    # nosetests tests/liver_segmentation_test.py -a actual

    @attr('interactive')
    @attr('actual')
    def test_liver_segmentation(self):
        import numpy as np
        import sed3
        img3d = np.random.rand(32, 64, 64) * 4
        img3d[4:24, 12:32, 5:25] = img3d[4:24, 12:32, 5:25] + 25

# seeds
        seeds = np.zeros([32, 64, 64], np.int8)
        seeds[9:12, 13:29, 18:24] = 1
        seeds[9:12, 4:9, 3:32] = 2
# [mm]  10 x 10 x 10        # voxelsize_mm = [1, 4, 3]
        voxelsize_mm = [5, 5, 5]

        ls = liver_segmentation.LiverSegmentation(
            data3d=img3d,
            voxelsize=voxelsize_mm,
            # seeds=seeds
        )
        ls.run()
        volume = np.sum(ls.segmentation == 1) * np.prod(voxelsize_mm)

        ed = sed3.sed3(img3d, contour=ls.segmentation, seeds=seeds)
        ed.show()


        # import pdb; pdb.set_trace()

        # mel by to být litr. tedy milion mm3
        self.assertGreater(volume, 900000)
        self.assertLess(volume, 1100000)

    def test_liver_segmenation_just_run(self):
        """
        Tests only if it run. No strong assert.
        """
        import numpy as np
        img3d = np.random.rand(32, 64, 64) * 4
        img3d[4:24, 12:32, 5:25] = img3d[4:24, 12:32, 5:25] + 25

# seeds
        seeds = np.zeros([32, 64, 64], np.int8)
        seeds[9:12, 13:29, 18:24] = 1
        seeds[9:12, 4:9, 3:32] = 2
# [mm]  10 x 10 x 10        # voxelsize_mm = [1, 4, 3]
        voxelsize_mm = [5, 5, 5]

        ls = liver_segmentation.LiverSegmentation(
            data3d=img3d,
            voxelsize=voxelsize_mm,
            # seeds=seeds
        )
        ls.run()

        # ed = sed3.sed3(img3d, contour=ls.segmentation, seeds=seeds)
        # ed.show()
if __name__ == "__main__":
    unittest.main()
