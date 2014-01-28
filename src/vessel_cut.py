#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.ndimage
import seg2fem
import viewer3
#import vtk
import argparse

from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
     QGridLayout, QLabel, QPushButton, QFrame, QFileDialog,\
     QFont, QInputDialog, QComboBox, QRadioButton, QButtonGroup

# ----------------- my scripts --------
import misc
import py3DSeedEditor
import show3
import qmisc
normal = 0
coordinates = None



def cut_editor_old(data):
    
    pyed = py3DSeedEditor.py3DSeedEditor(data['segmentation'])
    pyed.show()
    split_obj0 = pyed.seeds
    split_obj = split_obj0.copy()
    vessels = data['segmentation'] == data['slab']['porta']
    vesselstmp = vessels

    sumall = np.sum(vessels==1)

    #split_obj = scipy.ndimage.binary_dilation(split_obj, iterations = 5 )
    #vesselstmp = vessels * (1 - split_obj)

    lab, n_obj = scipy.ndimage.label(vesselstmp)

    print 'sumall ', sumall
    #while n_obj < 2 :
# dokud neni z celkoveho objektu ustipnuto alespon 80 procent
    while np.sum(lab == qmisc.max_area_index(lab,n_obj)) > (0.95*sumall) :

        split_obj = scipy.ndimage.binary_dilation(split_obj, iterations=3)
        vesselstmp = vessels * (1 - split_obj)
    
        lab, n_obj = scipy.ndimage.label(vesselstmp)
        print 'sum biggest ', np.sum(lab == qmisc.max_area_index(lab,n_obj))
        #print "n_obj  ",  n_obj
        #import pdb; pdb.set_trace()
        #print 'max ', np.sum(lab == max_area_index(lab,n_obj))
    
#    print ("Zjistete si, ktere objekty jsou nejvets a nastavte l1 a l2")

#    print ("np.sum(lab==3)")

    # všechny objekty, na které se to rozpadlo
    #pyed = py3DSeedEditor.py3DSeedEditor(lab)
    #pyed.show()
    obj1 = get_biggest_object(lab)

# vymaz nejvetsiho
    lab[obj1==1] = 0
    obj2 = get_biggest_object(lab)

    lab = obj1 + 2*obj2
    #print "baf"
    spl_vis = split_obj*2
    spl_vis[split_obj0] = 1
    #spl_vis[]
    pyed = py3DSeedEditor.py3DSeedEditor(lab, seeds=spl_vis)
    pyed.show()
    cut_by_user = split_obj0
    return lab, cut_by_user


def cut_editor(data,inputfile):
    #global normal,coordinates
    viewer = viewer3.Viewer(inputfile)
    # zobrazovani jater v kodu 
    viewer.prohlizej(data,'View','liver')
    
    #mesh = viewer.generate_mesh(segmentation,voxelsize_mm,degrad)
    #viewer.View(mesh,False)
    #viewer.buttons(window,grid)
    #print(viewer.normal)
    #print(viewer.coordinates)
    
    '''
    Funkce vrací trojrozměrné porobné jako data['segmentation'] 
    v data['slab'] je popsáno, co která hodnota znamená
    labels = []
    segmentation = segmentation[::degrad,::degrad,::degrad]
    print("Generuji data...")
    segmentation = segmentation[:,::-1,:]
    mesh_data = seg2fem.gen_mesh_from_voxels_mc(segmentation, voxelsize_mm*degrad)
    print("Done")
    if True:
        mesh_data.coors = seg2fem.smooth_mesh(mesh_data)
    vtk_file = "mesh_geom.vtk"
    mesh_data.write(vtk_file)
    app = QApplication(sys.argv)
    #view = viewer3.QVTKViewer(vtk_file,'Cut')
    '''

    
    
    
    #normal = viewer3.normal_and_coordinates().set_normal()
    #coordinates = viewer3.normal_and_coordinates().set_coordinates()
    #return normal,coordinates
    pass


def change(data,name):
    #vessels = get_biggest_object(data['segmentation'] == data['slab']['porta'])
    #data['segmentation'][vessels == 2] = data['slab']['porta']
    segmentation = data['segmentation']
    cut_editor(segmentation == data['slab'][name])
    
    
    
def resection(data,name, use_old_editor = False):
    if use_old_editor:
        return resection_old(data)
    else:
        return resection_new(data, name)

def resection_old(data):
    vessels = get_biggest_object(data['segmentation'] == data['slab']['porta'])
# ostranění porty z více kusů, nastaví se jim hodnota liver
    #data['segmentation'][data['segmentation'] == data['slab']['porta']] = data['slab']['liver']
    #show3.show3(data['segmentation'])

    #data['segmentation'][vessels == 1] = data['slab']['porta']
    segmentation = data['segmentation']
    print ("Select cut")
    
    print data["slab"]
    #lab = cut_editor(data)#== data['slab']['porta'])

    lab, cut = cut_editor_old(data)#['segmentation'] == data['slab']['porta'])
    

    l1 = 1
    l2 = 2
    

    # dist se tady počítá od nul jenom v jedničkách
    dist1 = scipy.ndimage.distance_transform_edt(lab != l1)
    dist2 = scipy.ndimage.distance_transform_edt(lab != l2)


    #segm = (dist1 < dist2) * (data['segmentation'] != data['slab']['none'])
    segm = (((data['segmentation'] != 0) * (dist1 < dist2)).astype('int8') + (data['segmentation'] != 0).astype('int8'))

    v1, v2 = liver_spit_volume_mm3(segm, data['voxelsize_mm'])
    print "Liver volume: %.4g l" % ((v1+v2)*1e-6)
    print "volume1: %.4g l  (%.3g %%)" % ((v1)*1e-6, 100*v1/(v1+v2))
    print "volume2: %.4g l  (%.3g %%)" % ((v2)*1e-6, 100*v2/(v1+v2))

    #pyed = py3DSeedEditor.py3DSeedEditor(segm)
    #pyed.show()
    #import pdb; pdb.set_trace()
    linie = (((data['segmentation'] != 0) * (np.abs(dist1 - dist2) < 1))).astype(np.int8)
    linie_vis = 2 * linie
    linie_vis[cut] = 1
    linie_vis= linie_vis.astype(np.int8)
    pyed = py3DSeedEditor.py3DSeedEditor(data['data3d'], seeds=linie_vis, contour=(data['segmentation'] != 0))
    pyed.show()

    #import pdb; pdb.set_trace()

    #show3.show3(data['segmentation'])

    
    data['slab']['resected_liver'] = 3
    data['slab']['resected_porta'] = 4

    mask_resected_liver = ((segm == 1) &
            (data['segmentation'] == data['slab']['liver']))
    mask_resected_porta = ((segm == 1) &
            (data['segmentation'] == data['slab']['porta']))

    data['segmentation'][mask_resected_liver] = \
            data['slab']['resected_liver']
    data['segmentation'][mask_resected_porta] = \
            data['slab']['resected_porta']
    
    return data

def resection_new(data, name):
    #vessels = get_biggest_object(data['segmentation'] == data['slab']['porta'])
    # ostranění porty z více kusů, nastaví se jim hodnota liver
    #data['segmentation'][data['segmentation'] == data['slab']['porta']] = data['slab']['liver']
    #show3.show3(data['segmentation'])

    #data['segmentation'][vessels == 2] = data['slab']['porta']
    segmentation = data['segmentation']
    print(data['slab'])
    change(data,name)
    
    #print data["slab"]
    #change(segmentation == data['slab']['porta'])
    #lab = cut_editor(segmentation == data['slab']['porta'])

    

def get_biggest_object(data):
    return qmisc.get_one_biggest_object(data)


def liver_spit_volume_mm3(segm, voxelsize_mm):
    """
    segm: 0 - nothing, 1 - remaining tissue, 2 - resected tissue
    """
    voxelsize_mm3 = np.prod(voxelsize_mm)
    v1 = np.sum(segm == 1) * voxelsize_mm3
    v2 = np.sum(segm == 2) * voxelsize_mm3

    return v1, v2

def View(name):
    data = misc.obj_from_file("out", filetype = 'pickle')
    resection(data,name)


        

if __name__ == "__main__":
    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)


#    SectorDisplay2__()



    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(description='Segment vessels from liver')
    parser.add_argument('-i', '--inputfile', 
            help='input file from organ_segmentation')
    parser.add_argument('-ii', '--defaultinputfile',  action='store_true',
            help='"organ.pkl" as input file from organ_segmentation')
    parser.add_argument('-oe', '--use_old_editor',  action='store_true',
            help='use an old editor for vessel cut')
    parser.add_argument('-o', '--outputfile',  default=None,
            help='output file')
    parser.add_argument('-oo', '--defaultoutputfile',  action='store_true',
            help='"vessels.pickle" as output file')
    args = parser.parse_args()


    data = misc.obj_from_file(args.inputfile, filetype = 'pickle')
    ds = data['segmentation'] == data['slab']['liver']
    print "vs ", data['voxelsize_mm']
    #seg = np.zeros([100,100,100])
    #seg [50:80, 50:80, 60:75] = 1
    #seg[58:60, 56:72, 66:68]=2
    #dat = np.random.rand(100,100,100) 
    #dat [50:80, 50:80, 60:75] =  dat [50:80, 50:80, 60:75] + 1 
    #dat [58:60, 56:72, 66:68] =  dat  [58:60, 56:72, 66:68] + 1
    #slab = {'liver':1, 'porta':2, 'portaa':3, 'portab':4}
    #data = {'segmentation':seg, 'data3d':dat, 'slab':slab}
    name = 'porta'

    #cut_editor(data,args.inputfile)
    if args.use_old_editor:
        resection(data,name, use_old_editor=args.use_old_editor)
    else:
        cut_editor(data,args.inputfile)
    #print normal
    #print coordinates



    defaultoutputfile =  "05-resection.pkl"
    if args.defaultoutputfile:
        args.outputfile = defaultoutputfile

    if args.defaultinputfile:
        args.inputfile = "out"

    if args.outputfile == None:

        savestring = raw_input ('Save output data? (y/n): ')
        if savestring in ['Y','y']:

            misc.obj_to_file(data, defaultoutputfile, filetype = 'pickle')
    else:
        misc.obj_to_file(data, args.outputfile, filetype = 'pickle')

