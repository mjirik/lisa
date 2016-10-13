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
        self.initUI()


    def initUI(self):
        self.mainLayout = QGridLayout(self)
        self.mainLayout.addWidget(QLabel("Key"), 0, 1)
        self.mainLayout.addWidget(QLabel("Value"), 0, 2)

        self.slabKeys = []
        self.slabValues = []
        self.pos = 1
        self.ui_label_lines = []
        for key, value in self.dictionary.slab.items(): #self.oseg.slab.items():
            keyW = QLineEdit(key)
            valueW = QLineEdit(str(value))
            btnDlt = QPushButton("X")
            pos = copy.copy(self.pos)
            ui_label_line = [pos, keyW, valueW, btnDlt]
            self.ui_label_lines.append(ui_label_line)
            btnDlt.setFixedWidth(30)
            #btnDlt.clicked.connect(lambda: self.deleteLine(str(pos)))
            def f1():
                self.deleteLine(pos)
            btnDlt.clicked.connect(lambda state, x=ui_label_line: self.deleteLine(x))
            self.mainLayout.addWidget(keyW, pos, 1)
            self.mainLayout.addWidget(valueW, pos, 2)
            self.mainLayout.addWidget(btnDlt, pos, 3)
            self.slabKeys.append(keyW)
            self.slabValues.append(valueW)
            self.pos += 1
            #smazat prvek: keyW.deleteLater() nebo skryt: keyW.setParent(None)


        self.lblSlabError = QLabel()
        self.lblSlabError.setStyleSheet("color: red;");
        self.mainLayout.addWidget(self.lblSlabError, 20, 1, 1, 2)

        btnAddLabel = QPushButton("Add label", self)
        btnAddLabel.clicked.connect(self.addLabel)
        self.mainLayout.addWidget(btnAddLabel, 21, 2)

        self.mainLayout.addWidget(QLabel("             "), 0, 3)

        btnSaveSlab = QPushButton("Save", self)
        btnSaveSlab.clicked.connect(self.saveSlab)
        self.mainLayout.addWidget(btnSaveSlab, 1, 4)

        self.btnBack = QPushButton("Back", self)
        self.mainLayout.addWidget(self.btnBack, 2, 4)

    def deleteLine(self, event):
        print event

    def addLabel(self):
        if self.pos < 13:
            keyW = QLineEdit()
            valueW = QLineEdit()
            self.mainLayout.addWidget(keyW, self.pos, 1)
            self.mainLayout.addWidget(valueW, self.pos, 2)
            self.slabKeys.append(keyW)
            self.slabValues.append(valueW)
            self.pos += 1
        else:
            self.lblSlabError.setText("You cannot add new label")

    def saveSlab(self, event):
        self.lblSlabError.setText('')
        newSlab = {}
        for i in range(0, len(self.slabKeys)):
            wk = str(self.slabKeys[i].text())
            wv = str(self.slabValues[i].text())
            if wk != '':
                newSlab[wk] = wv
            elif wv != '':
                self.lblSlabError.setText("You have to name key!")
        self.dictionary = newSlab
        print self.dictionary

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