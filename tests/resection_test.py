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

    def test_branch_labels_with_gui_just_in_module_(self):
        import lisa.organ_segmentation
        import io3d
        # datap = io3d.datasets.generate_abdominal()
        datap = io3d.datasets.generate_synthetic_liver(return_dataplus=True)
        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.import_dataplus(datap)
        bl = lisa.virtual_resection.branch_labels(oseg, "porta")
        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        slab = datap["slab"]
        organ_label = "liver"

        seeds = np.zeros_like(data3d, dtype=np.int)
        seeds[40, 125, 166] = 1
        seeds[40, 143, 130] = 2
        seeds[40, 125, 115] = 3

        seglabel1 = bl[seeds == 1][0]
        seglabel2 = bl[seeds == 2][0]
        seglabel3 = bl[seeds == 3][0]
        import imma.measure
        import imma.image_manipulation
        import imma.image_manipulation as ima

        import sed3
        ed = sed3.sed3(bl)  # , contour=datap["segmentation"])
        ed.show()
        organseg = ima.select_labels(segmentation, organ_label, slab)

        # organ_split = lisa.virtual_resection.split_tissue_on_bifurcation(
        #     bl, seglabel1, seglabel2, seglabel3, organseg
        # )

        neighb_labels = imma.measure.neighbors_list(
            bl,
            None,
            # [seglabel1, seglabel2, seglabel3],
            exclude=[0])
        #exclude=[imma.image_manipulation.get_nlabels(slab, ["liver"]), 0])
        # ex
        print(neighb_labels)
        # find whole branche

        connected2 = imma.measure.get_connected_labels(neighb_labels, seglabel2, [seglabel1, seglabel3])
        connected3 = imma.measure.get_connected_labels(neighb_labels, seglabel3, [seglabel1, seglabel2])

        # nl[seglabel2]

        # seg = ima.select_labels(segmentation, organ_label, slab).astype(np.int8)
        seg1 = ima.select_labels(bl, connected2).astype(np.int8)
        seg2 = ima.select_labels(bl, connected3).astype(np.int8)
        seg = seg1 + seg2 * 2
        # seg[ima.select_labels(bl, connected2)] = 2
        # seg[ima.select_labels(bl, connected3)] = 3

        dseg = ima.distance_segmentation(seg)
        dseg[~organseg] = 0

        import sed3
        ed = sed3.sed3(dseg, contour=seg)
        ed.show()

        # imma.measure.
        # coo = imma.measure.CooccurrenceMatrix(segmentation, return_counts=False, dtype=np.int8)
        # coond = coo.to_ndarray()
        # inv_keys = coo.inv_keys()
        # keys = coo.keys()
        #
        # slco1 = keys[seglabel1]
        # slco2 = keys[seglabel2]
        # slco3 = keys[seglabel3]
        # nz1 = np.nonzero(coond[slco1, :])
        # nz2 = np.nonzero(coond[slco2, :])
        # nz3 = np.nonzero(coond[slco3, :])
        #
        # neighb_labels1 = inv_keys[nz1]
        # neighb_labels2 = inv_keys[nz2]
        # neighb_labels3 = inv_keys[nz3]
        # un = np.unique(ed.seeds)

        self.assertEqual(True, True)


if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stderr)
    logger.setLevel(logging.DEBUG)
    unittest.main()
