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
from PyQt4.QtGui import QGridLayout, QLabel, QPushButton, QLineEdit, QComboBox
from PyQt4 import QtGui
import sys
from . import virtual_resection


class SegmentationWidget(QtGui.QWidget):
    def __init__(self, oseg, lisa_window, use_ui_label_dropdown=True):
        super(SegmentationWidget, self).__init__()
        self.oseg = oseg
        self.lisa_window = lisa_window
        self.use_ui_label_dropdown = use_ui_label_dropdown
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
        self.mainLayout.addWidget(btnLabel, 1, 3, 1, 2)

        lblSegType = QLabel('Choose type of segmentation')
        self.mainLayout.addWidget(lblSegType, 4, 1, 1, 6)
        lblSegType = QLabel('Choose virtual resection')
        self.mainLayout.addWidget(lblSegType, 6, 1, 1, 6)
        self.initConfigs()

        self.lblSegData = QLabel()
        self.mainLayout.addWidget(self.lblSegData, 8, 1, 1, 6)

        self.lblSegError = QLabel()
        self.lblSegError.setStyleSheet("color: red;");
        self.mainLayout.addWidget(self.lblSegError, 9, 1, 1, 6)

    def btnNewLabel(self):
        self.lisa_window.ui_select_label("Write new label")
        self.reinitLabels()

    def initLabels(self):
        if self.use_ui_label_dropdown:
            self.ui_label_dropdown = QComboBox()
            self.ui_label_dropdown.addItems(
                list(self.oseg.slab.keys())
            )
            # self.label_combo_box.clicked.connect(self.configEvent)
            self.ui_label_dropdown.activated[str].connect(self.action_select_label)
            column = 1
            row = 2

            self.mainLayout.addWidget(self.ui_label_dropdown, row, column)


        else:
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
        if self.use_ui_label_dropdown:
            self.mainLayout.removeWidget(self.ui_label_dropdown)
            self.ui_label_dropdown.deleteLater()
            self.ui_label_dropdown = None
        else:
            for btn in self.groupA.buttons():
                btn.deleteLater()
                self.groupA.removeButton(btn)
        self.initLabels()

    def action_select_label(self, text):
        selected_label = self.oseg.nlabels(text)
        print("selected label", selected_label)
        alt_seg_params = {
            "output_label": selected_label,
            'clean_seeds_after_update_parameters': True,
        }
        #alt_seg_params['output_label'] = selected_label
        self.oseg.update_parameters(alt_seg_params)

    def configEvent(self):
        logger.warning("Deprecated function configEvent. It will be removed in the future.")
        id = self.groupA.checkedId()
        selected_label = list(self.oseg.slab.keys())[id - 1]
        alt_seg_params = {
            "output_label": selected_label,
            'clean_seeds_after_update_parameters': True,
        }
         #alt_seg_params['output_label'] = selected_label
        self.oseg.update_parameters(alt_seg_params)

        self.enableSegType()

    def initConfigs(self):
        baserow = 5
        self.btnSegManual = QPushButton("Manual", self)
        # btnSegManual.clicked.connect(self.btnManualSeg)
        self.mainLayout.addWidget(self.btnSegManual, baserow, 1)

        self.btnSegSemiAuto = QPushButton("Semi-automatic", self)
        # btnSegSemiAuto.clicked.connect(self.btnSemiautoSeg)
        self.mainLayout.addWidget(self.btnSegSemiAuto, baserow, 2)

        self.btnSegMask = QPushButton("Mask", self)
        # btnSegMask.clicked.connect(self.maskRegion)
        self.mainLayout.addWidget(self.btnSegMask, baserow, 3)

        self.btnSegPV = QPushButton("Vessel", self)
        # btnSegPV.clicked.connect(self.btnPortalVeinSegmentation)
        self.mainLayout.addWidget(self.btnSegPV, baserow, 4)

        self.btnSegHV = QPushButton("Hepatic Vein", self)
        # btnSegHV.clicked.connect(self.btnHepaticVeinsSegmentation)
        self.mainLayout.addWidget(self.btnSegHV, baserow, 5)

        #dalsi radek
        self.btnVirtualResectionPV = QPushButton("Portal Vein", self)
        # btnVirtualResectionPV.clicked.connect(self.btnVirtualResectionPV)
        self.mainLayout.addWidget(self.btnVirtualResectionPV, baserow + 2, 1)

        self.btnVirtualResectionPlanar = QPushButton("Planar", self)
        # btnVirtualResectionPlanar.clicked.connect(self.btnVirtualResectionPlanar)
        self.mainLayout.addWidget(self.btnVirtualResectionPlanar, baserow + 2, 2)

        self.btnVirtualResectionPV_testing = QPushButton("PV testing", self)
        # self.btnVirtualResectionPV_testing.clicked.connect(    )
        self.mainLayout.addWidget(self.btnVirtualResectionPV_testing, baserow + 2, 4)

        self.btn_fill_segmentation = QPushButton("Fill holes", self)
        self.btn_fill_segmentation.setEnabled(True)
        self.btn_fill_segmentation.clicked.connect(self.action_fill_holes_in_segmentation)
        self.mainLayout.addWidget(self.btn_fill_segmentation, baserow + 2 , 5)

        self.btnSegSmoo = QPushButton("Segmentation smoothing", self)
        self.btnSegSmoo.clicked.connect(self.action_segmentation_smoothing)
        self.mainLayout.addWidget(self.btnSegSmoo, baserow + 2, 6)

        self.btnSegSmoo = QPushButton("Segmentation relabel", self)
        self.btnSegSmoo.clicked.connect(self.action_segmentation_relabel)
        self.mainLayout.addWidget(self.btnSegSmoo, baserow + 2, 7)
        # self.disableSegType()
        self.enableSegType()

    def action_fill_holes_in_segmentation(self):
        self.oseg.fill_holes_in_segmentation()

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

    def action_segmentation_smoothing(self):
        self.lisa_window.statusBar().showMessage('Segmentation smoothing...')

        val, ok = self.lisa_window.ui_get_double("Smoothing sigma in mm", value=1.)
        logger.debug(ok)
        if ok is False:
            return
        self.oseg.segm_smoothing(sigma_mm=val, labels=self.oseg.output_label)

        self.lisa_window.statusBar().showMessage('Ready')

    def action_segmentation_relabel(self):
        self.lisa_window.statusBar().showMessage('Segmentation relabelling...')
        strlabel = self.oseg.nlabels(self.oseg.output_label, return_mode="str")
        val = self.lisa_window.ui_select_label("Rename from " + strlabel + "to to fallowing label")
        self.oseg.segmentation_relabel(from_label=self.oseg.output_label, to_label=val[0])
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
