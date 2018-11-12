#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik
#
# Distributed under terms of the MIT license.

"""

"""
import unittest
from nose.plugins.attrib import attr
import sys

from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt
import lisa.lisaWindow
import lisa.organ_segmentation
from lisa.lisaWindow import OrganSegmentationWindow
import io3d.datasets


class LisaGUITest(unittest.TestCase):

    def setUp(self):
        '''Create the GUI'''
        self.app = QApplication(sys.argv)
        oseg = lisa.organ_segmentation.OrganSegmentation()
        self.oseg_w = OrganSegmentationWindow(oseg) # noqa
        # self.form = MargaritaMixer.MargaritaMixer()

    @attr('interactive')
    def test_click(self):
        # i5 = self.oseg_w.grid.itemAt(5)
        dcmdirw = self.oseg_w.uiw['dcmdir']
        QTest.mouseClick(dcmdirw, Qt.LeftButton)

    @attr('interactive')
    def test_lisa_run(self):
        # i5 = self.oseg_w.grid.itemAt(5)
        # dcmdirw = self.oseg_w.uiw['dcmdir']
        # QTest.mouseClick(dcmdirw, Qt.LeftButton)
        pass
        self.app.exec_()

    def test_zz_quit_gui(self):
        """
        Tests event of quit
        """
        self.assertTrue(self.oseg_w.quit(event=None))

    @attr('interactive')
    def test_split(self):
        self.oseg_w.oseg.load_data(r"C:\Users\miros\lisa_data\P09_cropped_portal_tree_labeled.pklz")
        self.app.exec_()
        # self.oseg_w.loadDataFile()

    @attr('interactive')
    def test_split_on_ircad(self):
        self.oseg_w.oseg.load_data(io3d.datasets.join_path("medical", "orig", "3Dircadb1.1", "PATIENT_DICOM", get_root=True))
        self.app.exec_()
        # self.oseg_w.loadDataFile()

    @attr('interactive')
    def test_relabel(self):
        self.oseg_w.oseg.load_data(io3d.datasets.join_path("medical", "orig", "3Dircadb1.1", "PATIENT_DICOM", get_root=True))
        self.oseg_w.ui_select_label("hura")
        self.app.exec_()

    def test_bodynavigation(self):
        self.oseg_w.oseg.load_data(io3d.datasets.join_path("medical", "orig", "3Dircadb1.1", "PATIENT_DICOM", get_root=True))
        self.oseg_w.ui_select_label("hura")

if __name__ == "__main__":
    unittest.main()
