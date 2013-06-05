#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))
import logging
logger = logging.getLogger(__name__)

from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
     QGridLayout, QLabel, QPushButton, QFrame, QFileDialog,\
     QFont, QInputDialog, QComboBox, QRadioButton, QButtonGroup
import argparse


import numpy as np
import seg2fem
import misc
import viewer

def showSegmentation(segmentation, voxelsize_mm = np.ones([3,1]), degrad = 4, label = 1):
    """
    Funkce vrací trojrozměrné porobné jako data['segmentation'] 
    v data['slab'] je popsáno, co která hodnota znamená
    """
    labels = []

    segmentation = segmentation[::degrad,::degrad,::degrad]
    
    #import pdb; pdb.set_trace()
    mesh_data = seg2fem.gen_mesh_from_voxels_mc(segmentation, voxelsize_mm*degrad)
    if False:
        mesh_data.coors = seg2fem.smooth_mesh(mesh_data)
    else:
        mesh_data = seg2fem.gen_mesh_from_voxels_mc(segmentation, voxelsize_mm * 1.0e-2)
        #mesh_data.coors += 
    vtk_file = "mesh_geom.vtk"
    mesh_data.write(vtk_file)
    app = QApplication(sys.argv)
    view = viewer.QVTKViewer(vtk_file)
    view.exec_()

    return labels

if __name__ == "__main__":
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(
            description='Segment vessels from liver \n\
                    \npython organ_segmentation.py\n\
                    \npython organ_segmentation.py -mroi -vs 0.6')
    parser.add_argument('-i', '--inputfile',
            default='organ.pkl',
            help='input file')
    parser.add_argument('-d', '--degrad', type=int,
            default=4,
            help='data degradation, default 4')
    parser.add_argument('-l', '--label', type=int, metavar='N', nargs='+',
            default=[4],
            help='segmentation labels, default 1')
    args = parser.parse_args()

    data = misc.obj_from_file(args.inputfile, filetype = 'pickle')
    #args.label = np.array(eval(args.label))
    #print args.label
    #import pdb; pdb.set_trace()
    ds = np.zeros(data['segmentation'].shape, np.bool)
    for i in range(0,len(args.label)):
        ds = ds | (data['segmentation'] == args.label[i])

    #print ds
    #print "sjdf ", ds.shape
    #ds = data['segmentation'] == args.label[0]
    #pyed = py3DSeedEditor.py3DSeedEditor(data['segmentation'])
    #pyed.show()
    #seg = np.zeros([100,100,100])
    #seg [50:80, 50:80, 60:75] = 1
    #seg[58:60, 56:72, 66:68]=2
    #dat = np.random.rand(100,100,100) 
    #dat [50:80, 50:80, 60:75] =  dat [50:80, 50:80, 60:75] + 1 
    #dat [58:60, 56:72, 66:68] =  dat  [58:60, 56:72, 66:68] + 1
    #slab = {'liver':1, 'porta':2, 'portaa':3, 'portab':4}
    #data = {'segmentation':seg, 'data3d':dat, 'slab':slab}

    showSegmentation(ds, degrad=args.degrad)
