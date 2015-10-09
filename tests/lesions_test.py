#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
import copy

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest

import numpy as np


import pysegbase.dcmreaddata as dcmr

class LesionsTest(unittest.TestCase):

    def test_import_lesion_editor(self):
        import lesioneditor
        import lesioneditor.Lession_editor_slim

    @unittest.skip("Cekame, az to Tomas opravi")
    def test_lesion_editor(self):
        """
        little more complex try to create editor object but do not run show 
        function

        """
        from PyQt4 import QtGui
        import lesioneditor
        import lesioneditor.Lession_editor_slim
        data3d = np.zeros([10, 11, 12], dtype=np.int16)
        segmentation = np.zeros(data3d.shape, dtype=np.int16)
        slab = {"liver": 1, "porta": 2}
        voxelsize_mm = [1.0, 1.2, 1.0]

        datap1={
            'data3d': data3d,
            'segmentation': segmentation,
            'slab': slab,
            'voxelsize_mm': voxelsize_mm
        }
        app = QtGui.QApplication(sys.argv)
        le = lesioneditor.Lession_editor_slim.LessionEditor(datap1=datap1)
    # @TODO znovu zprovoznit test
    @unittest.skip("Cekame, az to Tomas opravi")

    def test_synthetic_data_lesions_automatic_localization(self):
        """
        Function uses lesions  automatic localization in synthetic data.
        """
        import lesions
        #dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
# data
        slab = {'none':0, 'liver':1, 'porta':2, 'lesions':6}
        voxelsize_mm = np.array([1.0,1.0,1.2])

        segm = np.zeros([256,256,80], dtype=np.int16)

        # liver
        segm[70:190,40:220,30:60] = slab['liver']
# port
        segm[120:130,70:220,40:45] = slab['porta']
        segm[80:130,100:110,40:45] = slab['porta']
        segm[120:170,130:135,40:44] = slab['porta']

        # vytvoření kopie segmentace - před určením lézí
        segm_pre = copy.copy(segm)

        segm[150:180,70:105,42:55] = slab['lesions']


        data3d = np.zeros(segm.shape)
        data3d[segm== slab['none']] = 680
        data3d[segm== slab['liver']] = 1180
        data3d[segm== slab['porta']] = 1230
        data3d[segm== slab['lesions']] = 1110
        #noise = (np.random.rand(segm.shape[0], segm.shape[1], segm.shape[2])*30).astype(np.int16)
        noise = (np.random.normal(0,30,segm.shape))#.astype(np.int16)
        data3d = (data3d + noise  ).astype(np.int16)


        data={'data3d':data3d,
                'slab':slab,
                'voxelsize_mm':voxelsize_mm,
                'segmentation':segm_pre
                }

# @TODO je tam bug, prohlížeč neumí korektně pracovat s doubly
#        app = QApplication(sys.argv)
#        #pyed = QTSeedEditor(noise )
#        pyed = QTSeedEditor(data3d)
#        pyed.exec_()
#        #img3d = np.zeros([256,256,80], dtype=np.int16)

       # pyed = sed3.sed3(data3d)
       # pyed.show()

        tumory = lesions.Lesions()

        tumory.import_data(data)
        tumory.automatic_localization()
        #tumory.visualization()



# ověření výsledku
        #pyed = sed3.sed3(outputTmp, contour=segm==slab['porta'])
        #pyed.show()

        errim = np.abs(
                (tumory.segmentation == slab['lesions']).astype(np.int) -
                (segm == slab['lesions']).astype(np.int))

# ověření výsledku
        #pyed = sed3.sed3(errim, contour=segm==slab['porta'])
        #pyed.show()
#evaluation
        sum_of_wrong_voxels = np.sum(errim)
        sum_of_voxels = np.prod(segm.shape)

        #print "wrong ", sum_of_wrong_voxels
        #print "voxels", sum_of_voxels

        errorrate = sum_of_wrong_voxels/sum_of_voxels

        #import pdb; pdb.set_trace()

        self.assertLess(errorrate,0.1)




if __name__ == "__main__":
    unittest.main()
