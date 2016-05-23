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

import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *


class LogEntryModel(QAbstractListModel):
    def __init__(self, logfile, parent=None):
        super(LogEntryModel, self).__init__(parent)
        print logfile
        self.parent = parent
        self.entries = None
        self.slurp(logfile)
        self.logfile = logfile

    def rowCount(self, parent=QModelIndex()):
        return len(self.entries)

    def data(self, index, role):
        # print "data is called ", index.row()
        if index.isValid() and role == Qt.DisplayRole:
            return QVariant(self.entries[index.row()])
        else:
            return QVariant()

    def slurp(self, logfile=None):

        if self.entries is None:
            self.entries = []
        if logfile is None:
            logfile=self.logfile
        with open(logfile, 'rb') as fp:
            for line in fp.readlines():
                # tokens = line.strip().split(' : ')
                # sender = tokens[2]
                # message = tokens[4]
                # entry = "%s %s" % (sender, message)
                entry = line
                self.entries.append(entry)
            # print str(self.entries[-1]), len(self.entries)
            # self.parent.addItem("adf")
            # self.parent.update()
            # self.appendRow(self.enteries[-1])



class LogViewerForm(QDialog):
    def __init__(self, logfile, qapp=None, parent=None):
        super(LogViewerForm, self).__init__(parent)

        self.watcher = QFileSystemWatcher([logfile], parent=None)
        self.connect(self.watcher, SIGNAL('fileChanged(const QString&)'), self.update_log)
        self.qapp = qapp

        # build the list widget
        list_label = QLabel(QString("<strong>MoMo</strong> Log Viewer"))
        list_model = LogEntryModel(logfile)
        self.list_model = list_model
        self.list_view = QListView()
        self.list_view.setModel(self.list_model)
        list_label.setBuddy(self.list_view)

        # define the layout
        layout = QVBoxLayout()
        layout.addWidget(list_label)
        layout.addWidget(self.list_view)
        self.setLayout(layout)

    def update_log(self):
        print 'file changed'
        self.list_model.slurp(self.list_model.logfile)
        self.list_view.updateGeometries()
        self.list_view.update()
        self.list_view.repaint()

        self.qapp.processEvents()
        # from PyQt4.QtCore import pyqtRemoveInputHook; pyqtRemoveInputHook()
        # import ipdb; ipdb.set_trace()
        from PyQt4 import QtCore
        # QTCore

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
        required=True,
        help='input file'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)

    app = QApplication(sys.argv)
    # form = LogViewerForm(args.inputfile, qapp=app)
    # form.show()




    list = QListView()
    list.setWindowTitle('nazev okna')
    list.setMinimumSize(600,400)
    model = LogEntryModel(args.inputfile, parent=list)
    list.setModel(model)
    watcher = QFileSystemWatcher([args.inputfile], parent=None)

    list.connect(watcher, SIGNAL('fileChanged(const QString&)'), model.slurp)

    list.show()
    app.exec_()

if __name__ == "__main__":
    main()

