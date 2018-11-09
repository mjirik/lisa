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
        seeds[inds[0], inds[1], inds[2]] = 1

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
        seeds[inds[0], inds[1], inds[2]] = 1

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
        seeds[inds[0], inds[1], inds[2]] = 1

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
        un0 = np.unique(oseg.segmentation)
        keys0 = list(oseg.slab.keys())
        bl = lisa.virtual_resection.label_volumetric_vessel_tree(oseg, "porta")
        un1 = np.unique(oseg.segmentation)
        keys1 = list(oseg.slab.keys())

        # import sed3
        # ed = sed3.sed3(bl, contour=datap["segmentation"])
        # ed.show()

        self.assertGreater(len(un1), len(un0))
        self.assertGreater(len(keys1), len(keys0))

    def test_branch_labels_from_oseg(self):
        import lisa.organ_segmentation
        import io3d
        # datap = io3d.datasets.generate_abdominal()
        datap = io3d.datasets.generate_synthetic_liver(return_dataplus=True)
        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.import_dataplus(datap)
        un0 = np.unique(oseg.segmentation)
        keys0 = list(oseg.slab.keys())
        oseg.label_volumetric_vessel_tree("porta")
        un1 = np.unique(oseg.segmentation)
        keys1 = list(oseg.slab.keys())
        # import sed3
        # ed = sed3.sed3(oseg.segmentation)
        # ed.show()

        self.assertGreater(len(un1), len(un0))
        self.assertGreater(len(keys1), len(keys0))

    def test_branch_labels_with_gui_just_in_module(self):
        import lisa.organ_segmentation
        import io3d
        # datap = io3d.datasets.generate_abdominal()
        datap = io3d.datasets.generate_synthetic_liver(return_dataplus=True)
        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.import_dataplus(datap)
        labeled_branches = lisa.virtual_resection.label_volumetric_vessel_tree(oseg, "porta")
        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        slab = datap["slab"]
        organ_label = "liver"

        seeds = np.zeros_like(data3d, dtype=np.int)
        seeds[40, 125, 166] = 1
        seeds[40, 143, 130] = 2
        seeds[40, 125, 115] = 3

        seglabel1 = labeled_branches[seeds == 1][0]
        seglabel2 = labeled_branches[seeds == 2][0]
        seglabel3 = labeled_branches[seeds == 3][0]
        import imma.measure
        import imma.image_manipulation
        import imma.image_manipulation as ima

        # import sed3
        # ed = sed3.sed3(labeled_branches)  # , contour=datap["segmentation"])
        # ed.show()
        organseg = ima.select_labels(segmentation, organ_label, slab)

        organ_split, connected = lisa.virtual_resection.split_tissue_on_labeled_tree(
            labeled_branches, seglabel1, [seglabel2, seglabel3], organseg
        )

        # import sed3
        # # ed = sed3.sed3(labeled_branches, contour=organ_split)
        # ed = sed3.sed3(organ_split)
        # ed.show()

        self.assertTrue(np.array_equal(np.unique(organ_split), [0, 1, 2]))

        self.assertGreater(np.sum(organ_split == 0), 1000, "At least some background expected")
        self.assertGreater(np.sum(organ_split == 1), 1000, "At least some object expected")
        self.assertGreater(np.sum(organ_split == 2), 1000, "At least some object expected")

    def test_split_organ_segmentation(self):
        import lisa.organ_segmentation
        import io3d
        # datap = io3d.datasets.generate_abdominal()
        datap = io3d.datasets.generate_synthetic_liver(return_dataplus=True)
        slab = datap["slab"]
        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.import_dataplus(datap)
        oseg.label_volumetric_vessel_tree("porta")
        labeled_branches = oseg.segmentation
        seeds = np.zeros_like(oseg.data3d, dtype=np.int)
        seeds[40, 125, 166] = 1
        seeds[40, 143, 130] = 2
        seeds[40, 125, 115] = 3

        seglabel1 = labeled_branches[seeds == 1][0]
        seglabel2 = labeled_branches[seeds == 2][0]
        seglabel3 = labeled_branches[seeds == 3][0]
        split_labels_out, connected = oseg.split_tissue_with_labeled_volumetric_vessel_tree("liver",
                                                                                            seglabel1, [seglabel2, seglabel3],
                                                                                            split_labels=["split1", "split2"]
                                                                                            )
        # labeled_branches = lisa.virtual_resection.branch_labels(oseg, "porta")
        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        organ_label = "liver"



        # import sed3
        # # ed = sed3.sed3(labeled_branches, contour=organ_split)
        # ed = sed3.sed3(organ_split)
        # ed.show()

        self.assertGreater(np.sum(oseg.select_label("split1")), 1000)
        self.assertGreater(np.sum(oseg.select_label("split2")), 1000)

    def generate_trifurcation(self):
        import io3d.datasets
        datap = io3d.datasets.generate_synthetic_liver(return_dataplus=True)
        slab = datap["slab"]
        # third missed branch
        third_branch_location = [slice(40, 44), slice(90, 120), slice(128, 133)]
        datap["segmentation"][third_branch_location] = slab['porta']
        datap["data3d"][third_branch_location] += 206
        return datap

    def test_split_organ_segmentation_missed_branch(self):
        import lisa.organ_segmentation
        datap = self.generate_trifurcation()
        import io3d
        # datap = io3d.datasets.generate_abdominal()
        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.import_dataplus(datap)
        # import sed3
        # ed = sed3.sed3(oseg.data3d, contour=oseg.segmentation)
        # ed.show()
        oseg.label_volumetric_vessel_tree("porta")
        labeled_branches = oseg.segmentation
        seeds = np.zeros_like(oseg.data3d, dtype=np.int)
        seeds[40, 125, 166] = 1
        seeds[40, 143, 130] = 2
        seeds[40, 125, 115] = 3

        seglabel1 = labeled_branches[seeds == 1][0]
        seglabel2 = labeled_branches[seeds == 2][0]
        seglabel3 = labeled_branches[seeds == 3][0]
        split_labels_out, connected = oseg.split_tissue_with_labeled_volumetric_vessel_tree("liver",
                                                                                            seglabel1, [seglabel2, seglabel3],
                                                                                            split_labels=["split1", "split2"],
                                                                                            on_missed_branch="split"
                                                                                            )

        # import sed3
        # ed = sed3.sed3(oseg.data3d, contour=oseg.select_label("split1"))
        # ed.show()

        self.assertGreater(np.sum(oseg.select_label("split1")), 1000)
        self.assertGreater(np.sum(oseg.select_label("split2")), 1000)

    def test_split_organ_segmentation_missed_branch_neighboor_to_branches_organ_label(self):
        import lisa.organ_segmentation
        datap = self.generate_trifurcation()
        import io3d
        # datap = io3d.datasets.generate_abdominal()
        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.import_dataplus(datap)
        # import sed3
        # ed = sed3.sed3(oseg.data3d, contour=oseg.segmentation)
        # ed.show()
        oseg.label_volumetric_vessel_tree("porta")
        labeled_branches = oseg.segmentation
        # import sed3
        # ed = sed3.sed3(oseg.segmentation, contour=oseg.segmentation)
        # ed.show()
        seeds = np.zeros_like(oseg.data3d, dtype=np.int)
        seeds[40, 125, 166] = 1
        seeds[40, 143, 130] = 2
        seeds[42, 100, 130] = 3

        seglabel1 = labeled_branches[seeds == 1][0]
        seglabel2 = labeled_branches[seeds == 2][0]
        seglabel3 = labeled_branches[seeds == 3][0]
        split_labels_out, connected = oseg.split_tissue_with_labeled_volumetric_vessel_tree(
            "liver", seglabel1, [seglabel2, seglabel3],
            split_labels=["split1", "split2"],
            on_missed_branch="orig"
        )

        # import sed3
        # # ed = sed3.sed3(oseg.data3d, contour=oseg.select_label("split1"))
        # ed = sed3.sed3(oseg.segmentation, contour=oseg.select_label("split1"))
        # ed.show()

        # self.assertEqual(len(split_labels_out), 3)
        self.assertGreater(np.sum(oseg.select_label("split1")), 1000)
        self.assertGreater(np.sum(oseg.select_label("split2")), 1000)
        self.assertGreater(np.sum(oseg.select_label("liver")), 1000)

    def test_split_organ_segmentation_missed_branch_neighboor_to_trunk_organ_label(self):
        import lisa.organ_segmentation
        datap = self.generate_trifurcation()
        import io3d
        # datap = io3d.datasets.generate_abdominal()
        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.import_dataplus(datap)
        # import sed3
        # ed = sed3.sed3(oseg.data3d, contour=oseg.segmentation)
        # ed.show()
        oseg.label_volumetric_vessel_tree("porta")
        labeled_branches = oseg.segmentation
        # import sed3
        # ed = sed3.sed3(oseg.segmentation, contour=oseg.segmentation)
        # ed.show()
        seeds = np.zeros_like(oseg.data3d, dtype=np.int)
        seeds[40, 125, 166] = 1
        seeds[40, 143, 130] = 2
        seeds[40, 125, 115] = 3

        seglabel1 = labeled_branches[seeds == 1][0]
        seglabel2 = labeled_branches[seeds == 2][0]
        seglabel3 = labeled_branches[seeds == 3][0]
        split_labels_out, connected = oseg.split_tissue_with_labeled_volumetric_vessel_tree(
            "liver", seglabel1, [seglabel2, seglabel3],
            split_labels=["split1", "split2"],
            on_missed_branch="orig"
        )

        # import sed3
        # ed = sed3.sed3(oseg.data3d, contour=oseg.select_label("split1"))
        # ed.show()

        # self.assertEqual(len(split_labels_out), 3)
        self.assertGreater(np.sum(oseg.select_label("split1")), 1000)
        self.assertGreater(np.sum(oseg.select_label("split2")), 1000)
        self.assertGreater(np.sum(oseg.select_label("liver")), 1000)

    @unittest.skip("Waiting for implementation of recursive tissue segmentation")
    def test_split_organ_segmentation_recursive(self):
        import lisa.organ_segmentation
        import io3d
        # datap = io3d.datasets.generate_abdominal()
        datap = io3d.datasets.generate_synthetic_liver(return_dataplus=True)
        slab = datap["slab"]
        oseg = lisa.organ_segmentation.OrganSegmentation()
        oseg.import_dataplus(datap)
        oseg.label_volumetric_vessel_tree("porta")
        labeled_branches = oseg.segmentation
        seeds = np.zeros_like(oseg.data3d, dtype=np.int)
        seeds[40, 122:126, 165:168] = 1
        seeds[40, 144:148, 131:133] = 2
        seeds[40, 122:126, 114:117] = 2
        seeds[40, 90:95, 103:106] = 3
        seeds[40, 122:126, 84:88] = 3


        # import sed3
        # ed = sed3.sed3(datap["data3d"], contour=datap["segmentation"], seeds=seeds)
        # ed.show()

        split_labels_out, connected = oseg.split_tissue_recusively_with_labeled_volumetric_vessel_tree("liver", seeds)
        # labeled_branches = lisa.virtual_resection.branch_labels(oseg, "porta")
        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        organ_label = "liver"

        import sed3
        # ed = sed3.sed3(labeled_branches, contour=organ_split)
        ed = sed3.sed3(split_labels_out)
        ed.show()

        self.assertGreater(np.sum(oseg.select_label("split1")), 1000)
        self.assertGreater(np.sum(oseg.select_label("split2")), 1000)


if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stderr)
    logger.setLevel(logging.DEBUG)
    unittest.main()
