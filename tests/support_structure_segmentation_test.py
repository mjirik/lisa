#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest

import numpy as np


#
import imcut.dcmreaddata as dcmr
import lisa.support_structure_segmentation


class SupportStructureSegmentationTest(unittest.TestCase):

    def setUp(self):
        # self.dcmdir = os.path.join(
        #     path_to_script, '../sample_data/jatra_06mm_jenjatraplus/')
        self.dcmdir = os.path.join(path_to_script, '../sample_data/jatra_5mm')
        # self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
        reader = dcmr.DicomReader(self.dcmdir)
        self.data3d = reader.get_3Ddata()
        self.metadata = reader.get_metaData()

    @unittest.skip("comment after implementation")
    def test_bone_segmentation(self):
        """
        Check values in two areas.

        Area1: total number of voxels segmented as bones in probebox in spine.

        Area2: total number of voexel segmented as none in upper left corner.
        """

        slab = {'none': 0, 'bone': 8, 'lungs': 9, 'heart': 10}
        # import pdb; pdb.set_trace()
#            SupportStructureSegmentation
        sss = lisa.support_structure_segmentation.SupportStructureSegmentation(
            data3d=self.data3d,
            voxelsize_mm=self.metadata['voxelsize_mm'],
            modality='CT',
            slab=slab

        )

        sss.bone_segmentation()

        # total number of voxels segmented as bones in spine
        probebox1 = sss.segmentation[360:370, 260:270, 5:15] == slab['bone']
        self.assertGreater(np.sum(probebox1), 20)

        # total number of voexel segmented as none in upper left corner
        probebox1 = sss.segmentation[10:20, 10:20, 5:15] == slab['none']
        self.assertGreater(np.sum(probebox1), 900)

        # sss.visualization()
        # import pdb; pdb.set_trace()

    @unittest.skip("comment after implementation")
    def test_lungs_segmentation(self):
        """
        Check values in two areas.

        Area1: total number of voxels segmented as bones in probebox in lungs.

        Area2: total number of voexel segmented as none in upper left corner.
        """

        slab = {'none': 0, 'bone': 8, 'lungs': 9, 'heart': 10}
        # import pdb; pdb.set_trace()
#            SupportStructureSegmentation
        sss = lisa.support_structure_segmentation.SupportStructureSegmentation(
            data3d=self.data3d,
            voxelsize_mm=self.metadata['voxelsize_mm'],
            modality='CT',
            slab=slab

        )

        sss.lungs_segmentation()
        # sss.segmentation[260:270,160:170,1:10] = 2
        # sss.visualization()
        # total number of voxels segmented as bones in spine
        probebox1 = sss.segmentation[260:270, 160:170, 1:10] == slab['lungs']
        self.assertGreater(np.sum(probebox1), 20)

        # total number of voexel segmented as none in upper left corner
        probebox1 = sss.segmentation[10:20, 10:20, 5:15] == slab['none']
        self.assertGreater(np.sum(probebox1), 900)

        # import pdb; pdb.set_trace()


#    def test_synthetic_data_lesions_automatic_localization(self):
#        """
#        Function uses lesions  automatic localization in synthetic data.
#        """
# dcmdir = os.path.join(
#   path_to_script,
#   './../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
# data
#        slab = {'none':0, 'liver':1, 'porta':2, 'lesions':6}
#        voxelsize_mm = np.array([1.0,1.0,1.2])
#
#        segm = np.zeros([256,256,80], dtype=np.int16)
#
# liver
#        segm[70:190,40:220,30:60] = slab['liver']
# port
#        segm[120:130,70:220,40:45] = slab['porta']
#        segm[80:130,100:110,40:45] = slab['porta']
#        segm[120:170,130:135,40:44] = slab['porta']
#
# vytvoření kopie segmentace - před určením lézí
#        segm_pre = copy.copy(segm)
#
#        segm[150:180,70:105,42:55] = slab['lesions']
#
#
#        data3d = np.zeros(segm.shape)
#        data3d[segm== slab['none']] = 680
#        data3d[segm== slab['liver']] = 1180
#        data3d[segm== slab['porta']] = 1230
#        data3d[segm== slab['lesions']] = 1110
# noise = (np.random.rand(segm.shape[0], segm.shape[1],
#                         segm.shape[2])*30).astype(np.int16)
# noise = (np.random.normal(0,30,segm.shape))#.astype(np.int16)
#        data3d = (data3d + noise  ).astype(np.int16)
#
#
#        data={'data3d':data3d,
#                'slab':slab,
#                'voxelsize_mm':voxelsize_mm,
#                'segmentation':segm_pre
#                }
#
# @TODO je tam bug, prohlížeč neumí korektně pracovat s doubly
# app = QApplication(sys.argv)
# pyed = QTSeedEditor(noise )
# pyed = QTSeedEditor(data3d)
# pyed.exec_()
# img3d = np.zeros([256,256,80], dtype=np.int16)
#
# pyed = sed3.sed3(data3d)
# pyed.show()
#
#        tumory = lesions.Lesions()
#
#        tumory.import_data(data)
#        tumory.automatic_localization()
# tumory.visualization()
#
#
#
# ověření výsledku
# pyed = sed3.sed3(outputTmp, contour=segm==slab['porta'])
# pyed.show()
#
#        errim = np.abs(
#                (tumory.segmentation == slab['lesions']).astype(np.int) -
#                (segm == slab['lesions']).astype(np.int))
#
# ověření výsledku
# pyed = sed3.sed3(errim, contour=segm==slab['porta'])
# pyed.show()
# evaluation
#        sum_of_wrong_voxels = np.sum(errim)
#        sum_of_voxels = np.prod(segm.shape)
#
# print "wrong ", sum_of_wrong_voxels
# print "voxels", sum_of_voxels
#
#        errorrate = sum_of_wrong_voxels/sum_of_voxels
#
#
#        self.assertLess(errorrate,0.1)
#
#
#
if __name__ == "__main__":
    unittest.main()
