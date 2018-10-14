#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)
import unittest
import os.path as op
from nose.plugins.attrib import attr
path_to_script = op.dirname(op.abspath(__file__))

import sys
sys.path.insert(0, op.abspath(op.join(path_to_script, "../../io3d")))
import lisa.virtual_resection
import numpy as np

class ResectionTest(unittest.TestCase):

    # @unittest.skipIf(True, "Skipped test")
    # @attr("slow")
    @unittest.skip("demonstrating skipping")
    def test_nothing(self):
        self.fail("shouldn't happen")


    @attr('interactive')
    def test_planar_resection(self):
        """
        Select points for planar virtual resection and check if it works properly
        :return:
        """
        shape = [40, 41, 42]
        from lisa.dataset import generate_sample_data
        datap = generate_sample_data()
        import sys
        from PyQt4.QtGui import QApplication
        app = QApplication(sys.argv)
        print("Select points for planar virtual resection")
        datap = lisa.virtual_resection.resection_planar(datap, interactivity=True, seeds=None)

    def test_planar_resection2(self):
        """
        Check planar virtual resection
        :return:
        """
        shape = [40, 41, 42]
        from lisa.dataset import generate_sample_data
        datap = generate_sample_data(shape=shape)
        seeds = np.zeros(shape)
        inds = np.asarray([[10, 5, 20, 22], [20, 8, 10, 12], [5, 10, 20, 19]])
        seeds [inds[0], inds[1], inds[2]] = 1

        datap = lisa.virtual_resection.resection_planar(datap, interactivity=False, seeds=seeds)

        sonda1 = datap['segmentation'][11, 5, 4]
        sonda2 = datap['segmentation'][22, 22, 14]

        liver_labels = [datap['slab']['liver'], datap['slab']['resected_liver']]
        self.assertIn(sonda1, liver_labels)
        self.assertIn(sonda2, liver_labels)
        self.assertNotEqual(sonda1, sonda2)

    def test_portal_vein_new_resection(self):
        """
        Check planar virtual resection
        :return:
        """
        shape = [40, 41, 42]
        from lisa.dataset import generate_sample_data
        datap = generate_sample_data(shape=shape)
        seeds = np.zeros(shape)
        # points in portal vein where cut will be performed
        inds = np.asarray([[15, 15], [15, 15], [8, 9]])
        seeds [inds[0], inds[1], inds[2]] = 1

        datap = lisa.virtual_resection.resection_portal_vein_new(
            datap, interactivity=False, seeds=seeds)

        sonda1 = datap['segmentation'][11, 5, 4]
        sonda2 = datap['segmentation'][22, 22, 14]

        liver_labels = [datap['slab']['liver'], datap['slab']['resected_liver']]
        # import sed3
        # ed = sed3.sed3(datap['segmentation'])
        # ed.show()
        self.assertIn(sonda1, liver_labels)
        self.assertIn(sonda2, liver_labels)
        self.assertNotEqual(sonda1, sonda2)


    def test_portal_vein_resection(self):
        """
        Check planar virtual resection
        :return:
        """
        shape = [40, 41, 42]
        from lisa.dataset import generate_sample_data
        datap = generate_sample_data(shape=shape)
        seeds = np.zeros(shape)
        # points in portal vein where cut will be performed
        inds = np.asarray([[15, 15], [15, 15], [8, 9]])
        seeds [inds[0], inds[1], inds[2]] = 1

        # import sed3
        # ed = sed3.sed3(datap["data3d"], seeds=seeds, contour=datap['segmentation'])
        # ed.show()
        datap = lisa.virtual_resection.resection_old(datap, interactivity=False, seeds=seeds)

        sonda1 = datap['segmentation'][11, 5, 4]
        sonda2 = datap['segmentation'][22, 22, 14]

        liver_labels = [datap['slab']['liver'], datap['slab']['resected_liver']]
        self.assertIn(sonda1, liver_labels)
        self.assertIn(sonda2, liver_labels)
        self.assertNotEqual(sonda1, sonda2)
        # import sed3
        # ed = sed3.sed3(datap['segmentation'])
        # ed.show()

    def test_branch_labels_just_in_module(self):
        import lisa.organ_segmentation
        import io3d
        # datap = io3d.datasets.generate_abdominal()
        datap = io3d.datasets.generate_synthetic_liver(return_dataplus=True)
        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.import_dataplus(datap)
        bl = lisa.virtual_resection.branch_labels(oseg, "porta")

        import sed3
        ed = sed3.sed3(bl, contour=datap["segmentation"])
        ed.show()

        self.assertEqual(True, True)

    def test_branch_labels_from_oseg(self):
        import lisa.organ_segmentation
        import io3d
        # datap = io3d.datasets.generate_abdominal()
        datap = io3d.datasets.generate_synthetic_liver(return_dataplus=True)
        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.import_dataplus(datap)
        oseg.branch_labels("porta")

        import sed3
        ed = sed3.sed3(oseg.segmentation)
        ed.show()

        self.assertEqual(True, True)

if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stderr)
    logger.setLevel(logging.DEBUG)
    unittest.main()
