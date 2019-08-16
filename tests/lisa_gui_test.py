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
import pytest
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
import lisa.lisaWindow
import lisa.organ_segmentation
from lisa.lisaWindow import OrganSegmentationWindow
import io3d.datasets


class LisaGUITest(object):

    # def setUp(self):
    @classmethod
    def setup_class(self):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        '''Create the GUI'''
        self.app = QApplication(sys.argv)
        oseg = lisa.organ_segmentation.OrganSegmentation()
        self.oseg_w = OrganSegmentationWindow(oseg) # noqa
        # self.form = MargaritaMixer.MargaritaMixer()

    @classmethod
    def teardown_class(cls):
        """ teardown any state that was previously setup with a call to
        setup_class.
        """


    @pytest.mark.interactive
    def test_click(self):
        # i5 = self.oseg_w.grid.itemAt(5)
        dcmdirw = self.oseg_w.uiw['dcmdir']
        QTest.mouseClick(dcmdirw, Qt.LeftButton)

    @pytest.mark.interactive
    def test_lisa_run(self):
        # i5 = self.oseg_w.grid.itemAt(5)
        # dcmdirw = self.oseg_w.uiw['dcmdir']
        # QTest.mouseClick(dcmdirw, Qt.LeftButton)
        pass
        self.app.exec_()


    @pytest.mark.interactive
    def test_split(self):
        self.oseg_w.oseg.load_data(r"C:\Users\miros\lisa_data\P09_cropped_portal_tree_labeled.pklz")
        self.app.exec_()
        # self.oseg_w.loadDataFile()

    @pytest.mark.interactive
    def test_split_on_ircad(self):
        self.oseg_w.oseg.load_data(io3d.datasets.join_path("medical", "orig", "3Dircadb1.1", "PATIENT_DICOM", get_root=True))
        self.app.exec_()
        # self.oseg_w.loadDataFile()

    @pytest.mark.interactive
    def test_relabel(self):
        self.oseg_w.oseg.load_data(io3d.datasets.join_path("medical", "orig", "3Dircadb1.1", "PATIENT_DICOM", get_root=True))
        self.oseg_w.ui_select_label("hura")
        self.app.exec_()

    @pytest.mark.interactive
    def test_bodynavigation(self):
        self.oseg_w.oseg.load_data(io3d.datasets.join_path("medical", "orig", "3Dircadb1.1", "PATIENT_DICOM", get_root=True))
        self.oseg_w.ui_select_label("hura")

    def test_zz_quit_gui(self):
        """
        Tests event of quit
        """
        self.assertTrue(self.oseg_w.quit(event=None))

if __name__ == "__main__":
    unittest.main()
