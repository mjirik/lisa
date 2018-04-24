#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.

"""

"""

import logging
logger = logging.getLogger(__name__)
import argparse
import os

import PyQt4
import PyQt4.QtCore
import PyQt4.QtGui
from PyQt4.QtGui import QWidget, QGridLayout, QSpinBox, QLineEdit, QCheckBox,\
        QComboBox, QTextEdit, QDialog, QMainWindow, QDoubleSpinBox, QLabel, \
        QPushButton
import sys
if sys.version_info.major == 2:
    from PyQt4.QtCore import QString
else:
    QString = str

from pyqtconfig import ConfigManager

class DictGui(QDialog):
# class LisaConfigWindow(QMainWindow):

    def __init__(self, config_in, ncols=2):
        super(DictGui, self).__init__()

        self.setWindowTitle('Lisa Config')
        self.config = ConfigManager()
        self.ncols = ncols
# used with other module. If it is set, lisa config is deleted
        self.reset_config = False

        CHOICE_A = "{pairwise_alpha_per_mm2: 45, return_only_object_with_seeds: true}"
        # CHOICE_A = 23

        CHOICE_B = 2
        CHOICE_C = 3
        CHOICE_D = 4

        map_dict = {
            'Graph-Cut': CHOICE_A,
            'Choice B': CHOICE_B,
            'Choice C': CHOICE_C,
            'Choice D': CHOICE_D,
        }
        config_def = {
            'working_voxelsize_mm': 2.0,
            'save_filetype': 'pklz',
            # 'manualroi': True,
            'segparams': CHOICE_A,
        }
        
        config_def.update(config_in)
        
        self.config.set_defaults(config_def)

        gd = QGridLayout()
        gd_max_i = 0
        for key, value in config_in.items():
            if type(value) is int:
                sb = QSpinBox()
                sb.setRange(-100000, 100000)
            elif type(value) is float:
                sb = QDoubleSpinBox()
            elif type(value) is str:
                sb = QLineEdit()
            elif type(value) is bool:
                sb = QCheckBox()
            else:
                logger.error("Unexpected type in config dictionary")

            row = gd_max_i / self.ncols
            col = (gd_max_i % self.ncols) * 2

            gd.addWidget(QLabel(key),row, col +1) 
            gd.addWidget(sb, row, col + 2)
            self.config.add_handler(key, sb)
            gd_max_i += 1




        # gd.addWidget(QLabel("save filetype"), 1, 1)
        # te = QLineEdit()
        # gd.addWidget(te, 1, 2)
        # self.config.add_handler('save_filetype', te)
        #
        # gd.addWidget(QLabel("segmentation parameters"), 2, 1)
        # te = QLineEdit()
        # gd.addWidget(te, 2, 2)
        # self.config.add_handler('segparams', te)

        # cb = QCheckBox()
        # gd.addWidget(cb, 2, 2)
        # self.config.add_handler('active', cb)

        # gd.addWidget(QLabel("segmentation parameters"), 3, 1)
        # cmb = QComboBox()
        # cmb.addItems(map_dict.keys())
        # gd.addWidget(cmb, 3, 2)
        # self.config.add_handler('segparams', cmb, mapper=map_dict)

        self.current_config_output = QTextEdit()
        # rid.setColumnMinimumWidth(3, logo.width()/2)
        text_col = (self.ncols * 2) + 3
        gd.setColumnMinimumWidth(text_col, 500)
        gd.addWidget(self.current_config_output, 1, text_col, (gd_max_i/2)-1, 1)

        btn_reset_config = QPushButton("Default", self)
        btn_reset_config.clicked.connect(self.btnResetConfig)
        gd.addWidget(btn_reset_config, 0, text_col)

        btn_save_config = QPushButton("Ok", self)
        btn_save_config.clicked.connect(self.btnSaveConfig)
        gd.addWidget(btn_save_config, (gd_max_i / 2), text_col)

        self.config.updated.connect(self.show_config)

        self.show_config()

        # my line
        self.setLayout(gd)

        # self.window = QWidget()
        # self.window.setLayout(gd)
        # self.setCentralWidget(self.window)
    def add_grid_line(self, key, value):
        pass

    def get_config_as_dict(self):
        dictionary = self.config.as_dict()
        for key, value in dictionary.items():
            from PyQt4.QtCore import pyqtRemoveInputHook
            pyqtRemoveInputHook()
            # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
            if type(value) == QString:
                value = str(value)
            dictionary[key] = value

        return dictionary

    def btnResetConfig(self, event=None):
        self.reset_config = True
        self.close()

    def btnSaveConfig(self, event=None):
        self.reset_config = False
        self.close()


    def show_config(self):
        text = str(self.get_config_as_dict())
        self.current_config_output.setText(text)

def dictGui(cfg):
    import yaml
    # convert values to json
    isconverted = {}
    for key, value in cfg.items():
        if type(value) in (str, int, float, bool):

            isconverted[key] = False
            if type(value) is str:
                pass

        else:
            isconverted[key] = True
            cfg[key] = yaml.dump(value, default_flow_style=True)

    w = DictGui(cfg)
    w.exec_()
    if w.reset_config:
        return None
    
    newcfg = w.get_config_as_dict()
    for key, value in newcfg.items():
        if isconverted[key]:
            newcfg[key] = yaml.load(value)


    return newcfg

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



if __name__ == "__main__":
    main()
