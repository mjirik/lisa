#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © %YEAR%  <>
#
# Distributed under terms of the %LICENSE% license.

"""

"""

import logging

logger = logging.getLogger(__name__)
import argparse
from PyQt4.QtGui import QGridLayout, QLabel, QPushButton, QLineEdit
from PyQt4 import QtGui
import sys
import virtual_resection


class SegmentationWidget(QtGui.QWidget):
    def __init__(self, oseg, lisa_window):
        super(SegmentationWidget, self).__init__()
        self.oseg = oseg
        self.lisa_window = lisa_window
        self.initUI()

    def initUI(self):
        self.mainLayout = QGridLayout(self)
        self.groupA = None

        lblSegConfig = QLabel('Select label')
        self.mainLayout.addWidget(lblSegConfig, 1, 1, 1, 6)
        self.initLabels()

        btnLabel = QPushButton("New label")
        btnLabel.setCheckable(True)
        btnLabel.clicked.connect(self.btnNewLabel)
        self.mainLayout.addWidget(btnLabel, 4, 1, 1, 6)

        lblSegType = QLabel('Choose type of segmentation')
        self.mainLayout.addWidget(lblSegType, 5, 1, 1, 6)
        lblSegType = QLabel('Choose virtual resection')
        self.mainLayout.addWidget(lblSegType, 7, 1, 1, 6)
        self.initConfigs()

        self.lblSegData = QLabel()
        self.mainLayout.addWidget(self.lblSegData, 9, 1, 1, 6)

        self.lblSegError = QLabel()
        self.lblSegError.setStyleSheet("color: red;");
        self.mainLayout.addWidget(self.lblSegError, 10, 1, 1, 6)

    def btnNewLabel(self):
        self.lisa_window.ui_select_label("Write new label")
        self.reinitLabels()

    def initLabels(self):
        column = 1
        row = 2
        if self.groupA is None:
            self.groupA = QtGui.QButtonGroup()
        id = 0

        for key, value in self.oseg.slab.items():
            id += 1
            if key == "none":
                continue
            else:
                btnLabel = QPushButton(key)
                btnLabel.setCheckable(True)
                btnLabel.clicked.connect(self.configEvent)
                self.mainLayout.addWidget(btnLabel, row, column)
                self.groupA.addButton(btnLabel)
                self.groupA.setId(btnLabel, id)
                column += 1
                if column == 7:
                    column = 1
                    row += 1

    def reinitLabels(self):
        for btn in self.groupA.buttons():
            btn.deleteLater()
            self.groupA.removeButton(btn)
        self.initLabels()


    def configEvent(self):
        id = self.groupA.checkedId()
        selected_label = self.oseg.slab.keys()[id - 1]
        alt_seg_params = {
            "output_label": selected_label,
            'clean_seeds_after_update_parameters': True,
        }
         #alt_seg_params['output_label'] = selected_label
        self.oseg.update_parameters(alt_seg_params)

        self.enableSegType()

    def initConfigs(self):
        self.btnSegManual = QPushButton("Manual", self)
        # btnSegManual.clicked.connect(self.btnManualSeg)
        self.mainLayout.addWidget(self.btnSegManual, 6, 1)

        self.btnSegSemiAuto = QPushButton("Semi-automatic", self)
        # btnSegSemiAuto.clicked.connect(self.btnSemiautoSeg)
        self.mainLayout.addWidget(self.btnSegSemiAuto, 6, 2)

        self.btnSegMask = QPushButton("Mask", self)
        # btnSegMask.clicked.connect(self.maskRegion)
        self.mainLayout.addWidget(self.btnSegMask, 6, 3)

        self.btnSegPV = QPushButton("Portal Vein", self)
        # btnSegPV.clicked.connect(self.btnPortalVeinSegmentation)
        self.mainLayout.addWidget(self.btnSegPV, 6, 4)

        self.btnSegHV = QPushButton("Hepatic Vein", self)
        # btnSegHV.clicked.connect(self.btnHepaticVeinsSegmentation)
        self.mainLayout.addWidget(self.btnSegHV, 6, 5)

        #dalsi radek
        self.btnVirtualResectionPV = QPushButton("Portal Vein", self)
        # btnVirtualResectionPV.clicked.connect(self.btnVirtualResectionPV)
        self.mainLayout.addWidget(self.btnVirtualResectionPV, 8, 1)

        self.btnVirtualResectionPlanar = QPushButton("Planar", self)
        # btnVirtualResectionPlanar.clicked.connect(self.btnVirtualResectionPlanar)
        self.mainLayout.addWidget(self.btnVirtualResectionPlanar, 8, 2)

        self.btnVirtualResectionPV_testing = QPushButton("PV testing", self)
        # self.btnVirtualResectionPV_testing.clicked.connect(    )
        self.mainLayout.addWidget(self.btnVirtualResectionPV_testing, 8, 4)

        self.btnSegSmoo = QPushButton("Segmentation smoothing", self)
        self.btnSegSmoo.clicked.connect(self.btn_segmentation_smoothing)
        self.btnSegSmoo.setDisabled(False)
        self.btnSegSmoo.setCheckable(True)
        self.mainLayout.addWidget(self.btnSegSmoo, 10, 1)
        # self.disableSegType()
        self.enableSegType()

    def enableSegType(self):
        self.btnSegManual.setDisabled(False)
        self.btnSegSemiAuto.setDisabled(False)
        self.btnSegMask.setDisabled(False)
        self.btnSegPV.setDisabled(False)
        self.btnSegHV.setDisabled(False)
        self.btnVirtualResectionPV.setDisabled(False)
        self.btnVirtualResectionPlanar.setDisabled(False)

    def disableSegType(self):
        self.btnSegManual.setDisabled(True)
        self.btnSegSemiAuto.setDisabled(True)
        self.btnSegMask.setDisabled(True)
        self.btnSegPV.setDisabled(True)
        self.btnSegHV.setDisabled(True)
        self.btnVirtualResectionPV.setDisabled(True)
        self.btnVirtualResectionPlanar.setDisabled(True)

    def btn_segmentation_smoothing(self):
        self.lisa_window.statusBar().showMessage('Segmentation smoothing')
        val = self.lisa_window.ui_get_double("Smoothing sigma in mm", value=1.)
        self.oseg.segmentation_smooting(sigma_mm=val, labels=self.oseg.output_label)

        self.lisa_window.statusBar().showMessage('Ready')

def main():
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    # fh = logging.FileHandler('log.txt')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.debug('start')

    # input parser
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    # parser.add_argument(
    #     '-i', '--inputfile',
    #     default=None,
    #     required=True,
    #     help='input file'
    # )
    parser.add_argument(
        '--dict',
        default="{'jatra':2, 'ledviny':7}",
        # required=True,
        help='input dict'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)

    app = QtGui.QApplication(sys.argv)

    # w = QtGui.QWidget()
    # w = DictEdit(dictionary={'jatra':2, 'ledviny':7})
    w = SegmentationWidget()
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
