#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest

import numpy as np

#from PyQt4.QtGui import QApplication
import py3DSeedEditor

import organ_segmentation
import segmentation
import dcmreaddata as dcmr

class VesselsSegmentationTest(unittest.TestCase):
    interactiveTest = False


    #@unittest.skip("demonstrating skipping")
    @unittest.skipIf(not interactiveTest, "interactive test")
    def test_whole_organ_segmentation_interactive(self):
        pass

    def test_synthetic_data_segmentation(self):
        """
        Function uses organ_segmentation  for synthetic box object 
        segmentation.
        """
        #dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
# data
        slab = {'none':0, 'liver':1, 'porta':2}
        voxelsize_mm = np.array([1.0,1.0,1.2])

        segm = np.zeros([256,256,80], dtype=np.int16)

        # liver
        segm[70:180,40:190,30:60] = slab['liver']
# port
        segm[120:130,70:190,40:45] = slab['porta']
        segm[80:130,100:110,40:45] = slab['porta']
        segm[120:170,130:135,40:44] = slab['porta']



        data3d = np.zeros(segm.shape)
        data3d[segm== slab['liver']] = 156
        data3d[segm== slab['porta']] = 206
        #noise = (np.random.rand(segm.shape[0], segm.shape[1], segm.shape[2])*30).astype(np.int16)
        noise = (np.random.normal(0,30,segm.shape))#.astype(np.int16)
        data3d = (data3d + noise  ).astype(np.int16)

# @TODO je tam bug, prohlížeč neumí korektně pracovat s doubly 
#        app = QApplication(sys.argv)
#        #pyed = QTSeedEditor(noise )
#        pyed = QTSeedEditor(data3d)
#        pyed.exec_()
#        #img3d = np.zeros([256,256,80], dtype=np.int16)
        
       # pyed = py3DSeedEditor.py3DSeedEditor(data3d)
       # pyed.show()

        outputTmp = segmentation.vesselSegmentation(
            data3d, #.astype(np.uint8),
            segmentation = (segm==slab['liver']), #.astype(np.uint8),
            #segmentation = oseg.orig_scale_segmentation,
            voxelsize_mm = voxelsize_mm,
            threshold = 180,
            inputSigma = 0.15,
            dilationIterations = 2,
            nObj = 1,
            interactivity = False,
            biggestObjects=True,
            binaryClosingIterations = 5,
            binaryOpeningIterations = 1)

# ověření výsledku
        #pyed = py3DSeedEditor.py3DSeedEditor(outputTmp, contour=segm==slab['porta'])
        #pyed.show()

# @TODO opravit chybu v vesselSegmentation
        outputTmp = (outputTmp == 2)
        errim = np.abs(outputTmp.astype(np.int) - (segm == slab['porta']).astype(np.int))

# ověření výsledku
        #pyed = py3DSeedEditor.py3DSeedEditor(errim, contour=segm==slab['porta'])
        #pyed.show()
#evaluation
        sum_of_wrong_voxels = np.sum(errim)
        sum_of_voxels = np.prod(segm.shape)

        #print "wrong ", sum_of_wrong_voxels
        #print "voxels", sum_of_voxels

        errorrate = sum_of_wrong_voxels/sum_of_voxels

        #import pdb; pdb.set_trace()

        self.assertLess(errorrate,0.1)

    def test_real_data_segmentation(self):

        import misc
        dcmdir = os.path.join(path_to_script,'./../sample_data/jatra_5mm')
        
        oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4)
        oseg.add_seeds_mm([120],[120],[-81], label=1, radius=30)
        oseg.add_seeds_mm([170,220,250],[250,250,200],[-81], label=2, radius=30)
        #oseg.interactivity(min_val=-200, max_val=200)
        oseg.ninteractivity()

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

if __name__ == "__main__":
    unittest.main()
