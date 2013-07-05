#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/"))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))
#import featurevector

import logging
logger = logging.getLogger(__name__)

import numpy as np
#import scipy.ndimage
import argparse

#from PyQt4.QtCore import Qt
#from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
#     QGridLayout, QLabel, QPushButton, QFrame, QFileDialog,\
#     QFont, QInputDialog, QComboBox, QRadioButton, QButtonGroup

# ----------------- my scripts --------
import misc
import py3DSeedEditor




class SimpleSegmentation:
    def simple_segmentation(self, data3d, voxelsize_mm):
        simple_seg = np.zeros(data3d.shape )

        # ukázka zápisu do dat
        simple_seg[10:-10, 20:30, 2:5] = 1


        return simple_seg


        
def main():

    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(
            description='Module for segmentation of simple anatomical structures')
    parser.add_argument('-i', '--inputfile',
            default='organ.pkl',
            help='path to data dir')

    args = parser.parse_args()

    data = misc.obj_from_file(args.inputfile, filetype = 'pickle')
    data3d = data['data3d']
    voxelsize_mm = data['voxelsize_mm']

    ss = SimpleSegmentation()
    simple_seg = ss.simple_segmentation(data3d, voxelsize_mm)

    #visualization
    pyed = py3DSeedEditor.py3DSeedEditor(data['data3d'], seeds=simple_seg)
    pyed.show()

    # save
    savestring = raw_input('Save output data? (y/n): ')
    if savestring in ['Y', 'y']:
        misc.obj_to_file(data, "resection.pkl", filetype='pickle')


if __name__ == "__main__":
    main()

