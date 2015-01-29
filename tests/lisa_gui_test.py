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

    def test_quit_gui(self):
        """
        Tests event of quit
        """
        self.assertTrue(self.oseg_w.quit(event=None))


if __name__ == "__main__":
    unittest.main()
