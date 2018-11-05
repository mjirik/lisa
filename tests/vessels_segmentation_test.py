#! /usr/bin/python
# -*- coding: utf-8 -*-


# import funkcí z jiného adresáře
import os.path

import sys
path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
# sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
# sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest

from nose.plugins.attrib import attr
import numpy as np

from PyQt4.QtGui import QApplication
import sed3

from lisa import organ_segmentation
from imtools import segmentation


class VesselsSegmentationTest(unittest.TestCase):
    interactiveTest = False

    # @unittest.skip("demonstrating skipping")
    # @unittest.skipIf(not interactiveTest, "interactive test")
    @attr('interactive')
    def test_whole_organ_segmentation_interactive(self):
        pass

    def synthetic_data(self):
        # data
        slab = {'none': 0, 'liver': 1, 'porta': 2, "hepatic_veins": 3}
        voxelsize_mm = np.array([1.0, 1.0, 1.2])

        segm = np.zeros([256, 256, 80], dtype=np.int16)

        # liver
        segm[70:180, 40:190, 30:60] = slab['liver']
        # porta
        segm[120:130, 70:190, 40:45] = slab['porta']
        segm[80:130, 100:110, 40:45] = slab['porta']
        segm[120:170, 130:135, 40:44] = slab['porta']
        segm[150:160, 70:190, 51:55] = slab['hepatic_veins']

        data3d = np.zeros(segm.shape)
        data3d[segm == slab['liver']] = 156
        data3d[segm == slab['porta']] = 206
        noise = (np.random.normal(0, 30, segm.shape))  # .astype(np.int16)
        data3d = (data3d + noise).astype(np.int16)
        return data3d, segm, voxelsize_mm, slab

    def test_synthetic_data_segmentation(self):
        """
        Function uses organ_segmentation for synthetic box object
        segmentation.
        """

        data3d, segm, voxelsize_mm, slab = self.synthetic_data()
# @TODO je tam bug, prohlížeč neumí korektně pracovat s doubly
#        app = QApplication(sys.argv)
#        #pyed = QTSeedEditor(noise )
#        pyed = QTSeedEditor(data3d)
#        pyed.exec_()
#        #img3d = np.zeros([256,256,80], dtype=np.int16)

        # pyed = sed3.sed3(data3d)
        # pyed.show()

        outputTmp = segmentation.vesselSegmentation(
            data3d,  # .astype(np.uint8),
            segmentation=(segm == slab['liver']),  # .astype(np.uint8),
            # segmentation = oseg.orig_scale_segmentation,
            voxelsize_mm=voxelsize_mm,
            threshold=180,
            inputSigma=0.15,
            aoi_dilation_iterations=2,
            nObj=1,
            interactivity=False,
            biggestObjects=True,
            binaryClosingIterations=5,
            binaryOpeningIterations=1)

# ověření výsledku
        # pyed = sed3.sed3(outputTmp, contour=segm==slab['porta'])
        # pyed.show()

# @TODO opravit chybu v vesselSegmentation
        outputTmp = (outputTmp == 2)
        errim = np.abs(
            outputTmp.astype(np.int) - (segm == slab['porta']).astype(np.int)
        )

# ověření výsledku
        # pyed = sed3.sed3(errim, contour=segm==slab['porta'])
        # pyed.show()
# evaluation
        sum_of_wrong_voxels = np.sum(errim)
        sum_of_voxels = np.prod(segm.shape)

        # print "wrong ", sum_of_wrong_voxels
        # print "voxels", sum_of_voxels

        errorrate = sum_of_wrong_voxels/sum_of_voxels

        # import pdb; pdb.set_trace()

        self.assertLess(errorrate, 0.1)

    def test_virtual_resection(self):
        """
        Make virtual resection on synthetic data
        """
        import lisa.virtual_resection as vr
        data3d, segm, voxelsize_mm, slab = self.synthetic_data()
        seeds = np.zeros([256, 256, 80], dtype=np.int16)
        seeds[125, 160, 44] = 1
        datap = {'data3d': data3d, 'segmentation': segm,
                 'voxelsize_mm': voxelsize_mm, 'slab': slab}

        datap = vr.resection(
            datap, method='PV',
            interactivity=False, seeds=seeds
        )

        resected = np.sum(
            datap['segmentation'] == datap['slab']['resected_liver'])
        remaining = np.sum(
            datap['segmentation'] == datap['slab']['liver'])
        ratio = np.double(resected)/(resected+remaining)

        self.assertGreater(ratio, 0.17)
        self.assertLess(ratio, 0.19)

    def test_split_by_two_vessels(self):
        """
        Make virtual resection on synthetic data
        """
        import lisa.virtual_resection as vr
        data3d, segm, voxelsize_mm, slab = self.synthetic_data()
        seeds = np.zeros([256, 256, 80], dtype=np.int16)
        seeds[125, 160, 44] = 1
        datap = {'data3d': data3d, 'segmentation': segm,
                 'voxelsize_mm': voxelsize_mm, 'slab': slab}

        oseg = organ_segmentation.OrganSegmentation()
        oseg.import_dataplus(datap)

        oseg.split_organ_by_two_vessels(
            seed_label1="porta",
            seed_label2="hepatic_veins",
            organ_label="liver",
            output_label1="liver",
            output_label2="liver_remnant"
        )

        resected = np.sum(
            datap['segmentation'] == datap['slab']['liver'])
        remaining = np.sum(
            datap['segmentation'] == datap['slab']['liver_remnant'])
        ratio = np.double(resected)/(resected+remaining)
        # ed = sed3.sed3(datap["segmentation"])
        # ed.show()
        # print(ratio)

        self.assertGreater(ratio, 0.6)
        self.assertLess(ratio, 0.8)

    @attr('slow')
    def test_real_data_segmentation(self):
        import lisa.dataset
        app = QApplication(sys.argv)

        dcmdir = os.path.join(lisa.dataset.sample_data_path(), 'jatra_5mm')

        oseg = organ_segmentation.OrganSegmentation(dcmdir,
                                                    working_voxelsize_mm=4)
        oseg.add_seeds_mm([81], [120], [120],
                          label=1, radius=30)
        oseg.add_seeds_mm([81], [170, 220, 250], [250, 250, 200],
                          label=2, radius=30)
        oseg.add_seeds_mm([100], [170], [140],
                          label=10, radius=5)
        # oseg.interactivity(min_val=-200, max_val=200)
        oseg.ninteractivity()

        ed = sed3.sed3(oseg.data3d, contour=oseg.segmentation, seeds=oseg.seeds)
        ed.show()

        oseg.portalVeinSegmentation(seeds=(oseg.seeds==10))

#
#
#        outputTmp = segmentation.vesselSegmentation(
#            data3d,
#            segmentation = segm==slab['liver'],
#            #segmentation = oseg.orig_scale_segmentation,
#            voxelsize_mm = voxelsize_mm,
#            threshold = 1204,
#            inputSigma = 0.15,
#            dilationIterations = 2,
#            nObj = 1,
#            interactivity = False,
#            binaryClosingIterations = 5,
#            binaryOpeningIterations = 1)
#
#

    @attr('interactive')
    def test_ui_threshold_same_params_as_portal_segmentation(self):

        params = {
            'threshold': -1,
            'inputSigma': 0.15,
            'dilationIterations': 10,
            'nObj': 1,
            'biggestObjects': False,
            'useSeedsOfCompactObjects': True,
            'interactivity': True,
            'binaryClosingIterations': 2,
            'binaryOpeningIterations': 0
        }
        import imtools
        import imtools.sample_data
        datap = imtools.sample_data.generate()
        app = QApplication(sys.argv)
        outputTmp = segmentation.vesselSegmentation(
            datap['data3d'],
            datap['segmentation'],
            # datap['voxelsize_mm'],
            **params)
        # uit.run()
        # plt.show()
if __name__ == "__main__":
    unittest.main()
