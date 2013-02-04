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


def resection(data):
    #vessels = get_biggest_object(data['segmentation'] == data['slab']['porta'])

    #pyed = py3DSeedEditor.py3DSeedEditor(vessels)
    #pyed.show()
    show3.show3(data['segmentation'])
    

    import pdb; pdb.set_trace()
    



def get_biggest_object(data):
    """ Return biggest object """
    lab, num = scipy.ndimage.label(data)
    
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
        
class SectorDisplay2__: 
    def __init__(self): 
        win = gtk.Window() 
        win.set_default_size(800,800) 
        vbox = gtk.VBox() 
        win.add(vbox) 

        fig = Figure() 
        canvas = FigureCanvas(fig)  # a gtk.DrawingArea
        ax = fig.add_subplot(111, projection='3d') 

        a = np.array([[0,0],[10,0],[10,10],[0,10]]) 
        p = Polygon(a,fill=True) 
        ax.add_patch(p) 
        art3d.pathpatch_2d_to_3d(p, z=3) 

        ax.set_xlim3d(0, 20) 
        ax.set_ylim3d(0, 20) 
        ax.set_zlim3d(0, 20) 

        vbox.pack_start(canvas) 
        win.show_all() 
    
# Run the Gtk mainloop 
        gtk.main()

if __name__ == "__main__":
    data = misc.obj_from_file("out", filetype = 'pickle')
    ds = data['segmentation'] == data['slab']['liver']
    pyed = py3DSeedEditor.py3DSeedEditor(data['segmentation'])
    pyed.show()
    resection(data)

#    SectorDisplay2__()

