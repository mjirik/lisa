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
from PyQt4 import QtGui
from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QLineEdit, QFrame, \
    QFont, QPixmap, QFileDialog, QStyle
from PyQt4 import QtGui
from PyQt4.Qt import QString
import sys

class DictEdit(QtGui.QWidget):

    def __init__(self, dictionary={}):
        super(DictEdit, self).__init__()

        self.dictionary={}
        self.initUI()
        self.text.setText(str(dictionary))


    def initUI(self):
        mainLayout = QHBoxLayout(self)

        #self.setGeometry(300, 300, 250, 150)
        #self.setWindowTitle('Icon')
        #self.setWindowIcon(QtGui.QIcon('web.png'))

        btn = QtGui.QPushButton('Button', self)
        mainLayout.addWidget(btn)

        self.text = QtGui.QTextEdit()
        mainLayout.addWidget(self.text)


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