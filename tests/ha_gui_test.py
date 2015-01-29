#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@mjirik-Latitude-E6520>
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
import lisa.histology_analyser_gui as HA_GUI


class HAGUITest(unittest.TestCase):

    def setUp(self):
        '''Create the GUI'''
        self.app = QApplication(sys.argv)

    @attr('interactive')
    def test_click(self):
        pass
        # i5 = self.oseg_w.grid.itemAt(5)
        # dcmdirw = self.oseg_w.uiw['dcmdir']
        # QTest.mouseClick(dcmdirw, Qt.LeftButton)

    @attr('interactive')
    def test_run_and_quit_gui(self):
        """
        Tests event of quit
        """
        self.form = HA_GUI.HistologyAnalyserWindow(
            inputfile='sample_data/biodur_sample',
            crop=[0, 50, 300, 401, 400, 502]
        )

        # TODO remove
        import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        QTest.mouseClick(self.form.btn_process,
                         Qt.LeftButton)
        # self.assertTrue(self.oseg_w.quit(event=None))


if __name__ == "__main__":
    unittest.main()
