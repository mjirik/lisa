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
sys.path.insert(0, op.abspath(op.join(path_to_script, "../../imma")))
# import sys
# import os.path

# imcut_path =  os.path.join(path_to_script, "../../imcut/")
# sys.path.insert(0, imcut_path)
import lisa.virtual_resection
import numpy as np
import io3d


class OrganSegmentationNavigation(unittest.TestCase):

    # def test_relabel_organ_segmentation(self):
    #     import lisa.organ_segmentation
    #     import io3d
    #     # datap = io3d.datasets.generate_abdominal()
    #     datap = io3d.datasets.generate_synthetic_liver(return_dataplus=True)
    #     slab = datap["slab"]
    #     oseg = lisa.organ_segmentation.OrganSegmentation()
    #     oseg.import_dataplus(datap)
    #     self.assertGreater(np.sum(oseg.select_label("porta")), 0, "Generated porta should be bigger than 0 voxels")
    #     oseg.segmentation_relabel("porta", "liver")
    #
    #     self.assertEqual(np.sum(oseg.select_label("porta")), 0)
    #     self.assertGreater(np.sum(oseg.select_label("liver")), 100)
    #
    # def test_relabel_organ_segmentation_from_multiple_labels(self):
    #     import lisa.organ_segmentation
    #     import io3d
    #     # datap = io3d.datasets.generate_abdominal()
    #     datap = io3d.datasets.generate_synthetic_liver(return_dataplus=True)
    #     slab = datap["slab"]
    #     slab["tumor"] = 3
    #     datap["segmentation"][5:10, 5:10, 5:10] = slab["tumor"]
    #     oseg = lisa.organ_segmentation.OrganSegmentation()
    #     oseg.import_dataplus(datap)
    #     self.assertGreater(np.sum(oseg.select_label("porta")), 0, "Generated porta should be bigger than 0 voxels")
    #     oseg.segmentation_relabel(["porta", "tumor"], "liver")
    #
    #     self.assertEqual(np.sum(oseg.select_label("porta")), 0)
    #     self.assertEqual(np.sum(oseg.select_label("tumor")), 0)
    #     self.assertGreater(np.sum(oseg.select_label("liver")), 100)

    def test_bodynavigation(self):

        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.load_data(io3d.datasets.join_path("medical", "orig", "3Dircadb1.1", "PATIENT_DICOM", get_root=True))
        oseg.get_body_navigation_structures()
        self.assertGreater(np.sum(oseg.select_label("spine")), 100)

    def test_bodynavigation_precise(self):

        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.load_data(io3d.datasets.join_path("medical", "orig", "3Dircadb1.1", "PATIENT_DICOM", get_root=True))
        oseg.get_body_navigation_structures_precise()
        self.assertGreater(np.sum(oseg.select_label("bones")), 100)

if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stderr)
    logger.setLevel(logging.DEBUG)
    unittest.main()
