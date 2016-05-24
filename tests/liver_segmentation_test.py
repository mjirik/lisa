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

import lisa.dataset
from lisa import organ_segmentation
from lisa import liver_model

class LiverSegmentationTest(unittest.TestCase):

    # test se pouští ze složky lisa
    # nosetests tests/liver_segmentation_test.py -a actual

    @attr('interactive')
    @attr('slow')
    def test_automatic(self):
        pass

    @attr('slow')
    def test_train_liver_model(self):
        fn_mdl = "test_liver_intensity.Model.p"
        liver_model.train_liver_model_from_sliver_data(
            output_file= fn_mdl,
            sliver_reference_dir="~/lisa_data/sample_data/",
            orig_pattern="*orig*01.mhd",
            ref_pattern="*seg*01.mhd",
        )


    @attr('slow')
    def test_train_liver_model_and_liver_segmentation(self):
        """
        combination of test_train_liver_model and test_organ_model_with_pretrained_classifier
        :return:
        """
        fn_mdl = "test_liver_intensity.Model.p"
        liver_model.train_liver_model_from_sliver_data(
            output_file=fn_mdl,
            sliver_reference_dir="~/lisa_data/sample_data/",
            orig_pattern="*orig*01.mhd",
            ref_pattern="*seg*01.mhd",
        )

        path_to_data = lisa.dataset.sample_data_path()
        dcmdir = os.path.join(path_to_data, './liver-orig001.mhd')

        print "Interactive test: with left mouse button select liver, \
            with right mouse button select other tissues"
        # gcparams = {'pairwiseAlpha':10, 'use_boundary_penalties':True}
        # fn_mdl = os.path.expanduser("~/lisa_data/liver_intensity.Model.p")
        segparams = {}
        segmodelparams={
            'mdl_stored_file': fn_mdl,
            'fv_type':'fv_extern',
            'fv_extern': "intensity_localization_fv"
        }
        # 'pairwise_alpha_per': 3,
        #              'use_boundary_penalties': True,
        #              'boundary_penalties_sigma': 200}
        oseg = organ_segmentation.OrganSegmentation(
            dcmdir, working_voxelsize_mm=4, segparams=segparams, segmodelparams=segmodelparams)
        oseg.add_seeds_mm([200], [110, 150], [110, 100], label=1, radius=30, width=5)
        oseg.add_seeds_mm([200], [250, 230], [210, 260], label=2, radius=50, width=5)

        # from PyQt4.QtGui import QApplication
        # app = QApplication(sys.argv)
        # oseg.interactivity()
        oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()

        # misc.obj_to_file(oseg.get_iparams(),'iparams.pkl', filetype='pickle')

        self.assertGreater(volume, 1300000)
        self.assertLess(volume, 2100000)

    # @unittest.skipIf(not interactiveTest, "interactive test")
    # @unittest.skip("interactivity params are obsolete")

    @attr('interactive')
    @attr('slow')
    def test_organ_segmentation_with_pretrained_classifier(self):
        """
        Interactivity is stored to file
        :rtype: object
        """

        path_to_data = lisa.dataset.sample_data_path()
        dcmdir = os.path.join(path_to_data, './liver-orig001.mhd')

        print "Interactive test: with left mouse button select liver, \
            with right mouse button select other tissues"
        # gcparams = {'pairwiseAlpha':10, 'use_boundary_penalties':True}
        fn_mdl = os.path.expanduser("~/lisa_data/liver_intensity.Model.p")
        segparams = {}
        segmodelparams={
            'mdl_stored_file': fn_mdl,
            'fv_type':'fv_extern',
            'fv_extern': "intensity_localization_fv"
        }
        # 'pairwise_alpha_per': 3,
        #              'use_boundary_penalties': True,
        #              'boundary_penalties_sigma': 200}
        oseg = organ_segmentation.OrganSegmentation(
            dcmdir, working_voxelsize_mm=4, segparams=segparams, segmodelparams=segmodelparams)
        oseg.add_seeds_mm([200], [110, 150], [110, 100], label=1, radius=30, width=5)
        oseg.add_seeds_mm([200], [250, 230], [210, 260], label=2, radius=50, width=5)

        from PyQt4.QtGui import QApplication
        app = QApplication(sys.argv)
        oseg.interactivity()
        # oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()

        # misc.obj_to_file(oseg.get_iparams(),'iparams.pkl', filetype='pickle')

        self.assertGreater(volume, 1000000)
        self.assertLess(volume, 2000000)

    @attr('interactive')
    @attr('slow')
    def test_automatic_liver_seeds(self):
        """
        Interactivity is stored to file
        :rtype: object
        """

        path_to_data = lisa.dataset.sample_data_path()
        dcmdir = os.path.join(path_to_data, './liver-orig001.mhd')

        print "Interactive test: with left mouse button select liver, \
            with right mouse button select other tissues"
        # gcparams = {'pairwiseAlpha':10, 'use_boundary_penalties':True}
        fn_mdl = os.path.expanduser("~/lisa_data/liver_intensity.Model.p")
        segparams = {}
        segmodelparams={
            # 'mdl_stored_file': fn_mdl,
            # 'fv_type':'fv_extern',
            # 'fv_extern': "intensity_localization_fv"
        }
        # 'pairwise_alpha_per': 3,
        #              'use_boundary_penalties': True,
        #              'boundary_penalties_sigma': 200}
        oseg = organ_segmentation.OrganSegmentation(
            dcmdir, working_voxelsize_mm=4, segparams=segparams, segmodelparams=segmodelparams)
        # oseg.add_seeds_mm([110, 150], [110, 100], [200], label=1, radius=30, width=5)
        # oseg.add_seeds_mm([250, 230], [210, 260], [200], label=2, radius=50, width=5)
        print oseg.seeds
        oseg.automatic_liver_seeds()

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
