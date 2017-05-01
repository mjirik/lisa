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
import copy

class DictEdit(QtGui.QWidget):
    def __init__(self, dictionary):
        super(DictEdit, self).__init__()

        self.dictionary = dictionary
        self.autoValue = 1
        self.initUI()


    def initUI(self):
        self.mainLayout = QGridLayout(self)
        self.mainLayout.addWidget(QLabel("Key"), 0, 1)
        self.mainLayout.addWidget(QLabel("Value"), 0, 2)

        self.slabKeys = []
        self.slabValues = []
        self.pos = 1
        self.ui_label_lines = []

        self.initLines()

        self.lblSlabError = QLabel()
        self.lblSlabError.setStyleSheet("color: red;");
        self.mainLayout.addWidget(self.lblSlabError, 20, 1, 1, 2)

        btnAddLabel = QPushButton("+", self)
        btnAddLabel.setFixedWidth(30)
        btnAddLabel.setStyleSheet('QPushButton {background-color: green; color: #FFFFFF}')
        btnAddLabel.clicked.connect(self.addLabel)
        self.mainLayout.addWidget(btnAddLabel, 21, 3)

        btnSaveSlab = QPushButton("Save", self)
        btnSaveSlab.clicked.connect(self.saveSlab)
        self.mainLayout.addWidget(btnSaveSlab, 21, 1)
        self.btnSaveSlab = btnSaveSlab

        self.btnDiscard = QPushButton("Discard", self)
        self.btnDiscard.clicked.connect(self.discardChanges)
        self.mainLayout.addWidget(self.btnDiscard, 21, 2)

    def initLines(self):
        for key, value in self.dictionary.slab.items(): #self.oseg.slab.items():
            if key == "none":
                continue
            keyW = QLineEdit(key)
            valueW = QLineEdit(str(value))

            btnDlt = QPushButton(u"\u00D7")
            btnDlt.setFixedWidth(30)
            btnDlt.setStyleSheet('QPushButton {background-color: red; color: #FFFFFF}')

            ui_label_line = [self.pos, keyW, valueW, btnDlt]
            self.ui_label_lines.append(ui_label_line)
            btnDlt.clicked.connect(lambda state, x=ui_label_line: self.deleteLine(x))

            self.mainLayout.addWidget(keyW, self.pos, 1)
            self.mainLayout.addWidget(valueW, self.pos, 2)
            self.mainLayout.addWidget(btnDlt, self.pos, 3)


            self.slabKeys.append(keyW)
            self.slabValues.append(valueW)
            self.pos += 1

    def deleteLines(self):
        for i in range(0, len(self.ui_label_lines)):
            self.ui_label_lines[i][1].deleteLater()
            self.ui_label_lines[i][2].deleteLater()
            self.ui_label_lines[i][3].deleteLater()

        self.slabKeys = []
        self.slabValues = []
        self.pos = 1
        self.ui_label_lines = []

    def deleteLine(self, event):
        self.slabKeys.remove(event[1])
        self.slabValues.remove(event[2])

        event[1].deleteLater()
        event[2].deleteLater()
        event[3].deleteLater()

        self.sortLines(event[0])
        self.ui_label_lines.remove(event)
        self.pos -= 1

    def sortLines(self, whence):
        for i in range(whence, len(self.ui_label_lines)):
            self.ui_label_lines[i][0] = i
            self.mainLayout.addWidget(self.ui_label_lines[i][1], i, 1)
            self.mainLayout.addWidget(self.ui_label_lines[i][2], i, 2)
            self.mainLayout.addWidget(self.ui_label_lines[i][3], i, 3)

    def addLabel(self):
        if self.pos < 12:
            actualValues = []
            for i in range(0, len(self.slabValues)):
                actualValues.append(int(self.slabValues[i].text()))
            while self.autoValue in actualValues:
                self.autoValue += 1

            keyW = QLineEdit()
            valueW = QLineEdit(str(self.autoValue))
            btnDlt = QPushButton(u"\u00D7")
            btnDlt.setFixedWidth(30)
            btnDlt.setStyleSheet('QPushButton {background-color: red; color: #FFFFFF}')
            ui_label_line = [self.pos, keyW, valueW, btnDlt]
            self.ui_label_lines.append(ui_label_line)
            btnDlt.clicked.connect(lambda state, x=ui_label_line: self.deleteLine(x))

            self.mainLayout.addWidget(keyW, self.pos, 1)
            self.mainLayout.addWidget(valueW, self.pos, 2)
            self.mainLayout.addWidget(btnDlt, self.pos, 3)
            self.slabKeys.append(keyW)
            self.slabValues.append(valueW)

            self.pos += 1
        else:
            self.lblSlabError.setText("You cannot add new label")

    def saveSlab(self):
        self.lblSlabError.setText('')
        newSlab = {}
        for i in range(0, len(self.slabKeys)):
            wk = str(self.slabKeys[i].text())
            wv = str(self.slabValues[i].text())
            if wk != '':
                newSlab[wk] = wv
            elif wv != '':
                self.lblSlabError.setText("You have to name key!")
        self.dictionary.slab = newSlab

    def discardChanges(self):
        self.deleteLines()
        self.initLines()

    def getDict(self):
        return self.dictionary




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
    w = DictEdit(dictionary=eval(args.dict))
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()