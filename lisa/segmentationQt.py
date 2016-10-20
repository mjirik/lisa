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


class SegmentationWidget(QtGui.QWidget):
    def __init__(self, oseg):
        super(SegmentationWidget, self).__init__()
        self.oseg = oseg
        self.initUI()

    def initUI(self):
        self.mainLayout = QGridLayout(self)

        lblSegConfig = QLabel('Choose configure')
        self.mainLayout.addWidget(lblSegConfig, 1, 1, 1, 6)
        self.initLabels()

        lblSegType = QLabel('Choose type of segmentation')
        self.mainLayout.addWidget(lblSegType, 5, 1, 1, 6)
        self.initConfigs()

        self.lblSegError = QLabel()
        self.lblSegError.setStyleSheet("color: red;");
        self.mainLayout.addWidget(self.lblSegError, 10, 1, 1, 6)

        lblSegConfigBETA = QLabel('Choose configure (beta)')
        self.mainLayout.addWidget(lblSegConfigBETA, 11, 1, 1, 6)
        self.initLabelsAuto()

    def initLabels(self):
        btnHearth = QPushButton("Hearth", self)
        btnHearth.setCheckable(True)
        btnHearth.clicked.connect(self.configEvent)
        self.mainLayout.addWidget(btnHearth, 2, 1)

        btnKidneyL = QPushButton("Kidney Left", self)
        btnKidneyL.setCheckable(True)
        btnKidneyL.clicked.connect(self.configEvent)
        self.mainLayout.addWidget(btnKidneyL, 2, 2)

        btnKidneyR = QPushButton("Kidney Right", self)
        btnKidneyR.setCheckable(True)
        btnKidneyR.clicked.connect(self.configEvent)
        self.mainLayout.addWidget(btnKidneyR, 2, 3)

        btnLiver = QPushButton("Liver", self)
        btnLiver.setCheckable(True)
        btnLiver.clicked.connect(self.configEvent)
        self.mainLayout.addWidget(btnLiver, 2, 4)

        self.group = QtGui.QButtonGroup()
        self.group.addButton(btnHearth)
        self.group.addButton(btnKidneyL)
        self.group.addButton(btnKidneyR)
        self.group.addButton(btnLiver)
        self.group.setId(btnHearth, 1)
        self.group.setId(btnKidneyL, 2)
        self.group.setId(btnKidneyR, 3)
        self.group.setId(btnLiver, 4)

    def initLabelsAuto(self):
        position = 1
        self.groupA = QtGui.QButtonGroup()
        for key, value in self.oseg.slab.items():
            btnLabel = QPushButton(key)
            btnLabel.setCheckable(True)
            btnLabel.clicked.connect(self.configAutoEvent)
            self.mainLayout.addWidget(btnLabel, 12, position)
            self.groupA.addButton(btnLabel)
            self.groupA.setId(btnLabel, position)
            position += 1

    def configAutoEvent(self):
        alt_seg_params = {
            "output_label": 'left kidney',
            'clean_seeds_after_update_parameters': True,
        }
        id = self.groupA.checkedId()
        print id
        selected_label = self.oseg.slab.keys()[id - 1]
        alt_seg_params['output_label'] = selected_label
        self.oseg.update_parameters(alt_seg_params)

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

        self.disableSegType()

    def configEvent(self, event):
        id = self.group.checkedId()
        self.lblSegError.setText("")
        if id == 1:
            self.oseg.update_parameters_based_on_label("label hearth")
            self.enableSegType()
        elif id == 2:
            self.oseg.update_parameters_based_on_label("label kidney L")
            self.enableSegType()
        elif id == 3:
            self.oseg.update_parameters_based_on_label("label kidney R")
            self.enableSegType()
        elif id == 4:
            self.oseg.update_parameters_based_on_label("label liver")
            self.enableSegType()
        else:
            self.lblSegError.setText("Unknown error: Config have not been set.")
            self.disableSegType()

    def enableSegType(self):
        self.btnSegManual.setDisabled(False)
        self.btnSegSemiAuto.setDisabled(False)
        self.btnSegMask.setDisabled(False)
        self.btnSegPV.setDisabled(False)
        self.btnSegHV.setDisabled(False)

    def disableSegType(self):
        self.btnSegManual.setDisabled(True)
        self.btnSegSemiAuto.setDisabled(True)
        self.btnSegMask.setDisabled(True)
        self.btnSegPV.setDisabled(True)
        self.btnSegHV.setDisabled(True)


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
