#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/"))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/extern/py3DSeedEditor/"))
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.ndimage

# ----------------- my scripts --------
import misc
import py3DSeedEditor
import show3


def cut_editor(img3d):
    pass

def cut_editor_old(img3d):
    pass

def resection(data):
    vessels = get_biggest_object(data['segmentation'] == data['slab']['porta'])
# ostranění porty z více kusů, nastaví se jim hodnota liver
    data['segmentation'][data['segmentation'] == data['slab']['porta']] = data['slab']['liver']
    show3.show3(data['segmentation'])

    data['segmentation'][vessels == 1] = data['slab']['porta']
    img3d = data['segmentation']
    print ("Select cut")
    #cut_editor_old(img3d)
    pyed = py3DSeedEditor.py3DSeedEditor(data['segmentation'])
    pyed.show()
    split_obj = pyed.seeds
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
    
    print ("Zjistete si, ktere objekty jsou nejvets a nastavte l1 a l2")
    print (str(n_obj))

    print ("np.sum(lab==3)")
    

    l1 = 1
    l2 = 2
    import pdb; pdb.set_trace()

    # dist se tady počítá od nul jenom v jedničkách
    dist1 = scipy.ndimage.distance_transform_edt(lab != l1)
    dist2 = scipy.ndimage.distance_transform_edt(lab != l2)


    #segm = (dist1 < dist2) * (data['segmentation'] != data['slab']['none'])
    segm = (((data['segmentation'] != 0) * (dist1 < dist2)).astype('int8') + (data['segmentation'] != 0).astype('int8'))

    pyed = py3DSeedEditor.py3DSeedEditor(segm)
    pyed.show()
    import pdb; pdb.set_trace()
    pyed = py3DSeedEditor.py3DSeedEditor(data['data3d'], contour=segm)
    pyed.show()
    import pdb; pdb.set_trace()

    #show3.show3(data['segmentation'])
    

    





def get_biggest_object(data):
    """ Return biggest object """
    lab, num = scipy.ndimage.label(data)
    print ("bum = "+str(num))
    
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


import gtk 
import numpy as np 
from matplotlib.patches import Polygon, PathPatch 
import mpl_toolkits.mplot3d.art3d as art3d 
from matplotlib.figure import Figure 
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas 
        

if __name__ == "__main__":
    data = misc.obj_from_file("out", filetype = 'pickle')
    ds = data['segmentation'] == data['slab']['liver']
    pyed = py3DSeedEditor.py3DSeedEditor(data['segmentation'])
    pyed.show()
    resection(data)

#    SectorDisplay2__()

