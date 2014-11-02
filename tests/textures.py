#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest

import numpy as np


import organ_segmentation
import pysegbase.dcmreaddata as dcmr


#  nosetests tests/organ_segmentation_test.py:OrganSegmentationTest.test_create_iparams


class TexturesTest(unittest.TestCase):
    interactiveTest = False
    verbose = False

    def test_texture_features(self):
        """
        Interactivity is stored to file
        """
        try:
            from pysegbase.seed_editor_qt import QTSeedEditor
        except:
            logger.warning("Deprecated of pyseg_base as submodule")
            from seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication
        from skimage.feature import greycomatrix, greycoprops
        import misc
        dcmdir = os.path.join(path_to_script,'./../sample_data/jatra_5mm')
        
        #gcparams = {'pairwiseAlpha':10, 'use_boundary_penalties':True}
        #segparams = {'pairwise_alpha_per':3, 'use_boundary_penalties':True,'boundary_penalties_sigma':200}
        #oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4, segparams=segparams)
        #oseg.add_seeds_mm([120],[120],[70], label=1, radius=30)
        #oseg.add_seeds_mm([170,220,250],[250,280,200],[70], label=2, radius=30)


        reader = dcmr.DicomReader(dcmdir) # , qt_app=qt_app)
        data3d = reader.get_3Ddata()
# normalizace na čísla od nuly do 255
        data3d = data3d/(2**4)
        metadata = reader.get_metaData()
        iparams = {}
        iparams['series_number'] = reader.series_number
        iparams['datadir'] = dcmdir

        working_voxelsize_mm = 2

        voxelsize_mm = np.array(metadata['voxelsize_mm'])
        zoom = voxelsize_mm / working_voxelsize_mm

        PATCH_SIZE = 21
        PATCH_DIST = 15
        shp = data3d.shape
        vx, vy, vz = np.mgrid[0:shp[0] - PATCH_SIZE:PATCH_DIST, 
                0:shp[1] - PATCH_SIZE:PATCH_DIST,
                0:shp[2] - PATCH_SIZE:8]

        
        import pdb; pdb.set_trace()

        feat = np.zeros(vx.shape)
        feat2 = np.zeros(vx.shape)

        #vx = vx.reshape(-1)
        #vy = vy.reshape(-1)
        #vz = vz.reshape(-1)


        

        #for i in range(0,len(vx)):
        it = np.nditer(vx, flags=['multi_index'])
        while not it.finished:
            vxi = vx[it.multi_index]
            vyi = vy[it.multi_index]
            vzi = vz[it.multi_index]

            

            patch = data3d[
                vxi:vxi + PATCH_SIZE,
                vyi:vyi + PATCH_SIZE,
                vzi
                ]
            patch = np.squeeze(patch)
            print  it.iterindex, ' - ', vxi, ' ',  vyi, ' ',vzi , ' - ',it.multi_index
            #import pdb; pdb.set_trace()
            glcm = greycomatrix(patch,
                [5],[0],256,
                symmetric=True, normed=True)
            dissimilarity = greycoprops(glcm, 'dissimilarity')
            feat[it.multi_index] = dissimilarity
            feat2[it.multi_index] = greycoprops(glcm,'correlation')
            it.iternext()





        locations = [(474, 291), (440, 433), (466, 18), (462, 236)]




        qt_app = QApplication(sys.argv)
        pyed = QTSeedEditor(feat)
        qt_app.exec_()
        pyed = QTSeedEditor(feat2)
        qt_app.exec_()

        pdb


if __name__ == "__main__":
    unittest.main()
