#! /usr/bin/python
# -*- coding: utf-8 -*-
#definice konstant
BONES = 1290
SPINE = 105
INSIDE_BODY = 5
LUNGS_UP = 400
LUNGS_DOWN = 150	
SPINE_ID = 2
LUNGS_ID = 3

# import funkcí z jiného adresáře
import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/"))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/extern/sed3/"))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))
#import featurevector

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.ndimage
import argparse

#from PyQt4.QtCore import Qt
#from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
#     QGridLayout, QLabel, QPushButton, QFrame, QFileDialog,\
#     QFont, QInputDialog, QComboBox, QRadioButton, QButtonGroup

# ----------------- my scripts --------
import misc
import sed3




class SimpleSegmentation:
    
    def simple_segmentation(self, data3d, voxelsize_mm):
        simple_seg = np.zeros(data3d.shape )
	#definice konvoluční masky
	KONV_MASK = np.ones([10,10,10], float)
	KONV_MASK = KONV_MASK/9.0;	
	

	#definice konvoluční funkce - param a - matice m*n kterou budeme přenásobovat konvoluční maticí, b - Konvoluční maska m*n
     
	
	# nalezení kostí
	simple_seg = data3d > BONES 
	#simple_seg[(simple_seg.shape[0]/5)*4:simple_seg.shape[0]] = 0
	
	#nalzení páteře
	spine_finder = scipy.ndimage.filters.convolve((simple_seg).astype(np.int), KONV_MASK)
	#pyed = sed3.sed3(simple_seg)
	#
				
	simple_seg += ((spine_finder>25)*SPINE_ID)
	pyed = sed3.sed3(simple_seg)
	pyed.show()	

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
    pyed = sed3.sed3(data['data3d'], seeds=simple_seg)
    pyed.show()

    # save
    savestring = raw_input('Save output data? (y/n): ')
    if savestring in ['Y', 'y']:
        misc.obj_to_file(data, "resection.pkl", filetype='pickle')


if __name__ == "__main__":
    main()

