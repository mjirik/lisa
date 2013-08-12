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
import unittest

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.ndimage
import seg2fem
import viewer3
import vtk

from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
     QGridLayout, QLabel, QPushButton, QFrame, QFileDialog,\
     QFont, QInputDialog, QComboBox, QRadioButton, QButtonGroup

# ----------------- my scripts --------
import misc
import py3DSeedEditor
import show3
normal = 0
coordinates = None



def cut_editor_old(data):
    pyed = py3DSeedEditor.py3DSeedEditor(data['segmentation'])
    pyed.show()
    split_obj = pyed.seeds
    vessels = data['segmentation'] == data['slab']['porta']
    vesselstmp = vessels

    sumall = np.sum(vessels==1)

    split_obj = scipy.ndimage.binary_dilation(split_obj, iterations = 5 )
    vesselstmp = vessels * (1 - split_obj)

    lab, n_obj = scipy.ndimage.label(vesselstmp)

    #while n_obj < 2 :
# dokud neni z celkoveho objektu ustipnuto alespon 80 procent
    while np.sum(lab == max_area_index(lab,n_obj)) > (0.8*sumall) :

        split_obj = scipy.ndimage.binary_dilation(split_obj, iterations = 5 )
        vesselstmp = vessels * (1 - split_obj)
    
        lab, n_obj = scipy.ndimage.label(vesselstmp)
    
#    print ("Zjistete si, ktere objekty jsou nejvets a nastavte l1 a l2")
#    print (str(n_obj))

#    print ("np.sum(lab==3)")

    obj1 = get_biggest_object(lab)
   # pyed = py3DSeedEditor.py3DSeedEditor(lab)
    #pyed.show()
# vymaz nejvetsiho
    lab[obj1==1] = 0
    obj2 = get_biggest_object(lab)

    lab = obj1 + 2*obj2
   # pyed = py3DSeedEditor.py3DSeedEditor(lab)
   # pyed.show()
    return lab
    pass

def cut_editor(segmentation, voxelsize_mm = np.ones([3,1]), degrad = 3):
    global normal,coordinates
    """
    Funkce vrací trojrozměrné porobné jako data['segmentation'] 
    v data['slab'] je popsáno, co která hodnota znamená
    """
    labels = []
    segmentation = segmentation[::degrad,::degrad,::degrad]
    print("Generuji data...")
    mesh_data = seg2fem.gen_mesh_from_voxels_mc(segmentation, voxelsize_mm*degrad)
    print("Done")
    if True:
        mesh_data.coors = seg2fem.smooth_mesh(mesh_data)
    vtk_file = "mesh_geom.vtk"
    mesh_data.write(vtk_file)
    app = QApplication(sys.argv)
    #view = viewer3.QVTKViewer(vtk_file)
    viewer3.QVTKViewer(vtk_file)
    normal = viewer3.normal_and_coordinates().set_normal()
    coordinates = viewer3.normal_and_coordinates().set_coordinates()
    return normal,coordinates
    pass


def change(data,name):
    vessels = get_biggest_object(data['segmentation'] == data['slab']['porta'])
    data['segmentation'][vessels == 2] = data['slab']['porta']
    segmentation = data['segmentation']
    cut_editor(segmentation == data['slab'][name])
    
    
    
def resection(data,name):
    vessels = get_biggest_object(data['segmentation'] == data['slab']['porta'])
    # ostranění porty z více kusů, nastaví se jim hodnota liver
    #data['segmentation'][data['segmentation'] == data['slab']['porta']] = data['slab']['liver']
    #show3.show3(data['segmentation'])

    data['segmentation'][vessels == 2] = data['slab']['porta']
    segmentation = data['segmentation']
    print(data['slab'])
    change(data,name)
    
    #print data["slab"]
    #change(segmentation == data['slab']['porta'])
    #lab = cut_editor(segmentation == data['slab']['porta'])

    #lab = cut_editor_old(segmentation)
    

    l1 = 1
    l2 = 2
    

    # dist se tady počítá od nul jenom v jedničkách
    #dist1 = scipy.ndimage.distance_transform_edt(lab != l1)
    #dist2 = scipy.ndimage.distance_transform_edt(lab != l2)


    #segm = (dist1 < dist2) * (data['segmentation'] != data['slab']['none'])
    #segm = (((data['segmentation'] != 0) * (dist1 < dist2)).astype('int8') + (data['segmentation'] != 0).astype('int8'))

    #pyed = py3DSeedEditor.py3DSeedEditor(segm)
   # pyed.show()
   # import pdb; pdb.set_trace()
   # pyed = py3DSeedEditor.py3DSeedEditor(data['data3d'], contour=segm)
    #pyed.show()
    #import pdb; pdb.set_trace()

    #show3.show3(data['segmentation'])
    

    

def get_biggest_object(data):
    """ Return biggest object """
    lab, num = scipy.ndimage.label(data)
    #print ("bum = "+str(num))
    
    maxlab = max_area_index(lab, num)

    data = (lab == maxlab)
    return data


def max_area_index(labels, num):
    """
    Return index of maxmum labeled area
    """
    mx = 0
    mxi = -1
    for l in range(1,num):
        mxtmp = np.sum(labels == l)
        if mxtmp > mx:
            mx = mxtmp
            mxi = l

    return mxi

def View(name):
    data = misc.obj_from_file("out", filetype = 'pickle')
    resection(data,name)


        

if __name__ == "__main__":
    data = misc.obj_from_file("out", filetype = 'pickle')
    ds = data['segmentation'] == data['slab']['liver']
    #seg = np.zeros([100,100,100])
    #seg [50:80, 50:80, 60:75] = 1
    #seg[58:60, 56:72, 66:68]=2
    #dat = np.random.rand(100,100,100) 
    #dat [50:80, 50:80, 60:75] =  dat [50:80, 50:80, 60:75] + 1 
    #dat [58:60, 56:72, 66:68] =  dat  [58:60, 56:72, 66:68] + 1
    #slab = {'liver':1, 'porta':2, 'portaa':3, 'portab':4}
    #data = {'segmentation':seg, 'data3d':dat, 'slab':slab}
    name = 'porta'
    resection(data,name)
    print normal
    print coordinates

#    SectorDisplay2__()

