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

from PyQt4 import QtGui
# from mainwindow import Ui_MainWindow

class Login(QtGui.QDialog):
    def __init__(self, parent=None, checkLoginFcn=None):
        super(Login, self).__init__(parent)
        self.textName = QtGui.QLineEdit(self)
        self.textPass = QtGui.QLineEdit(self)
        self.textPass.setEchoMode(QtGui.QLineEdit.Password)
        self.buttonLogin = QtGui.QPushButton('Login', self)
        self.buttonLogin.clicked.connect(self.handleLogin)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.textName)
        layout.addWidget(self.textPass)
        layout.addWidget(self.buttonLogin)
        if checkLoginFcn is None:
            self.checkLoginFcn = self.checkLogin
        else:
            self.checkLoginFcn = checkLoginFcn

    def checkLogin(self, text_name, text_pass):
        if (text_name == 'foo' and
                    text_pass == 'bar'):
            return True
        else:
            return False

    def handleLogin(self):
        if self.checkLoginFcn(str(self.textName.text()), str(self.textPass.text())):
            self.accept()
        else:
            QtGui.QMessageBox.warning(
                self, 'Error', 'Bad user or password')

class Window(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        # self.ui = Ui_MainWindow()
        # self.ui.setupUi(self)



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

    import sys
    app = QtGui.QApplication(sys.argv)
    login = Login()

    if login.exec_() == QtGui.QDialog.Accepted:
        window = Window()
        window.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()