#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © %YEAR% %USER% <%MAIL%>
#
# Distributed under terms of the %LICENSE% license.

"""
%HERE%
"""

import logging

logger = logging.getLogger(__name__)
import argparse
import PyQt4 #import QtGui, QtCore
import PyQt4.QtCore
import PyQt4.QtGui
import sys
import time

from . import lisa_data

def splash_screen(qapp):
    """
    create lisa splash screen
    :param qapp:
    :return:
    """
   # Create and display the splash screen
    lisa_data.create_lisa_data_dir_tree()
    splash_pix = PyQt4.QtGui.QPixmap(lisa_data.path('.lisa/LISA256.png'))
    splash = PyQt4.QtGui.QSplashScreen(splash_pix, PyQt4.QtCore.Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    qapp.processEvents()
    return splash


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
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        # required=True,
        help='input file'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)
 # Simulate something that takes time

    app = PyQt4.QtGui.QApplication(sys.argv)
    splash = splash_screen(app)
    time.sleep(2)
    w = PyQt4.QtGui.QWidget()
    b = PyQt4.QtGui.QLabel(w)
    b.setText("Hello World!")
    w.setGeometry(100,100,200,50)
    b.move(50,20)
    w.setWindowTitle("PyQt")
    w.show()
    splash.finish(w)
    app.exec_()
if __name__ == "__main__":
    main()