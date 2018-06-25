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

from PyQt4.QtGui import QFileDialog, QApplication
try:
    from imcut.seed_editor_qt import QTSeedEditor
except:
    logger.warning("Deprecated of pyseg_base as submodule")
    try:
        from imcut.seed_editor_qt import QTSeedEditor
    except:
        logger.warning("Deprecated of pyseg_base as submodule")
        from seed_editor_qt import QTSeedEditor


import numpy as np


# import imcut.dcmreaddata as dcmr


#  nosetests tests/organ_segmentation_test.py:OrganSegmentationTest.test_create_iparams


class QtAppTest(unittest.TestCase):
    interactiveTest = False


    @unittest.skipIf(not interactiveTest, "interactive test")
    def test_viewer_seeds(self):


        app = QApplication(sys.argv)
        dcmdir = QFileDialog.getExistingDirectory(caption='Select DICOM Folder',
                                                  options=QFileDialog.ShowDirsOnly)
        print('ahoj')
        #app.exec_()
        #app.exit(0)

        #del(app)

        #import pdb; pdb.set_trace()
        img = np.random.rand(30,30,30)
        print('1')
        app = QApplication(sys.argv)
        print('2')
        pyed = QTSeedEditor(img)
        print('3')
        app.exec_()
        del(app)

        app = QApplication(sys.argv)
        print('2')
        pyed = QTSeedEditor(img)
        print('3')
        app.exec_()
        del(app)


if __name__ == "__main__":
    unittest.main()
