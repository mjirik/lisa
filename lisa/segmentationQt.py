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
        self.initLabelsAuto()

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

    def initLabelsAuto(self):
        position = 1
        self.groupA = QtGui.QButtonGroup()
        id = 1
        for key, value in self.oseg.slab.items():
            if key == "none":
                continue
            else:
                btnLabel = QPushButton(key)
                btnLabel.setCheckable(True)
                btnLabel.clicked.connect(self.configAutoEvent)
                self.mainLayout.addWidget(btnLabel, 2, position)
                self.groupA.addButton(btnLabel)
                self.groupA.setId(btnLabel, id)
                position += 1
            id += 1

    def configAutoEvent(self):
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

        self.disableSegType()

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
