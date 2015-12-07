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
import io3d
import numpy as np
import sys
import os

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../../pysegbase/src"))

import os
path_to_script = os.path.dirname(os.path.abspath(__file__))
import logging
logger = logging.getLogger(__name__)

import lisa.data
from lisa import organ_segmentation

class LiverSegmentationTest(unittest.TestCase):

    # test se pouští ze složky lisa
    # nosetests tests/liver_segmentation_test.py -a actual

    @attr('interactive')
    @attr('slow')
    def test_automatic(self):
        pass

    def test_train_liver_model(self):



    # @unittest.skipIf(not interactiveTest, "interactive test")
    # @unittest.skip("interactivity params are obsolete")

    @attr('interactive')
    @attr('slow')
    def test_organ_segmentation_with_pretrained_classifier(self):
        """
        Interactivity is stored to file
        :rtype: object
        """

        path_to_data = lisa.data.sample_data_path()
        dcmdir = os.path.join(path_to_data, './liver-orig001.mhd')

        print "Interactive test: with left mouse button select liver, \
            with right mouse button select other tissues"
        # gcparams = {'pairwiseAlpha':10, 'use_boundary_penalties':True}
        fn_mdl = os.path.expanduser("~/lisa_data/liver_intensity.Model.p")
        segparams = {}
        segmodelparams={'mdl_stored_file': fn_mdl}
        # 'pairwise_alpha_per': 3,
        #              'use_boundary_penalties': True,
        #              'boundary_penalties_sigma': 200}
        oseg = organ_segmentation.OrganSegmentation(
            dcmdir, working_voxelsize_mm=4, segparams=segparams, segmodelparams=segmodelparams)
        oseg.add_seeds_mm([110, 150], [110, 100], [200], label=1, radius=30, width=5)
        oseg.add_seeds_mm([250, 230], [210, 260], [200], label=2, radius=50, width=5)

        from PyQt4.QtGui import QApplication
        app = QApplication(sys.argv)
        oseg.interactivity()
        # oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()

        # misc.obj_to_file(oseg.get_iparams(),'iparams.pkl', filetype='pickle')

        self.assertGreater(volume, 1000000)
        self.assertLess(volume, 2000000)


if __name__ == "__main__":
    unittest.main()
