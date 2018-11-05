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

from nose.plugins.attrib import attr
# import numpy as np


# import imcut.dcmreaddata as dcmr

from seededitorqt import QTSeedEditor

class QTSeedEditorTest(unittest.TestCase):
    interactive_tests = False

    @unittest.skipIf(not interactive_tests,"interactive test")
    def test_data_editor(self):
        """
        Funkce provádí změnu vstupních dat - data3d
        """
        #pyed = sed3.sed3(self.data3d, contour = oseg.segmentation)
        #pyed.show()

        from PyQt4.QtGui import QApplication
        import numpy as np
        im3d = np.random.rand(15,15,15)
        print("Select pixels for deletion (it will be setted to 0)")
#, QMainWindow
        app = QApplication(sys.argv)
        pyed = QTSeedEditor(im3d, mode='draw')
        pyed.exec_()


        deletemask = pyed.getSeeds()


        print("If it is ok, press 'Return'. If it is wrong, click into image and press 'Return'")
        # rewrite input data
        im3d [deletemask != 0] = 0
        pyed = QTSeedEditor(im3d)
        app.exec_()
        sds  = pyed.getSeeds()
# if usere select pixel, there will be 1 or 2 in sds
        self.assertLess(np.max(sds), 1)

    ##@unittest.skipIf(True,"interactive test")
    #def test_vincentka_06_slice_thickness_interactive(self):
    #    """
    #    Interactive test. SliceThickness is not voxel depth. If it is, this
    #    test will fail.
    #    """
    #    #dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
    #    dcmdir = os.path.expanduser('~/data/medical/data_orig/vincentka/13021610/10200000/')
    #    dcmdir = os.path.expanduser('~/data/medical/data_orig/vincentka/13021610/12460000/')
    #    oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4)
    #
# ma#nual seeds setting
    #    print("with left mouse button select some pixels of the bottle content")
    #    print("with right mouse button select some pixels of background")

    #    oseg.interactivity()

    #    volume = oseg.get_segmented_volume_size_mm3()
    #    #print volume

    #    self.assertGreater(volume,600000)
    #    self.assertLess(volume,850000)

    @attr("interactive")
    def test_data_editor_tree(self):
        """
        Just for visual check of seed edito
        """
        #pyed = sed3.sed3(self.data3d, contour = oseg.segmentation)
        #pyed.show()

        from PyQt4.QtGui import QApplication
        import numpy as np
        im3d = np.random.rand(15,15,15)
        print("Select pixels for deletion (it will be setted to 0)")
        #, QMainWindow
        app = QApplication(sys.argv)
        pyed = QTSeedEditor(im3d, mode='draw')
        pyed.exec_()


        deletemask = pyed.getSeeds()

if __name__ == "__main__":
    unittest.main()
