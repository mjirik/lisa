#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.ndimage

# ----------------- my scripts --------
import misc
import sed3
import virtual_resection


def resection(data):
    #pyed = sed3.sed3(data['segmentation'])
    #pyed.show()
    # vessels = get_biggest_object(data['segmentation'] == data['slab']['porta'])
    vessels = data['segmentation'] == data['slab']['porta']
# ostranění porty z více kusů, nastaví se jim hodnota liver
    #data['segmentation'][data['segmentation'] == data['slab']['porta']] = data['slab']['liver']
    #show3.show3(data['segmentation'])
    import pdb; pdb.set_trace()

    #data['segmentation'][vessels == 1] = data['slab']['porta']
    #segm = data['segmentation']
    #pyed = sed3.sed3(vessels)
    #pyed.show()
    print("Select cut")
    lab = virtual_resection.cut_editor_old(data)
    pyed = sed3.sed3(lab )#, contour=segm)
    pyed.show()
    l1 = 1
    l2 = 2
    #import pdb; pdb.set_trace()

    # dist se tady počítá od nul jenom v jedničkách
    dist1 = scipy.ndimage.distance_transform_edt(lab != l1)
    dist2 = scipy.ndimage.distance_transform_edt(lab != l2)




    #segm = (dist1 < dist2) * (data['segmentation'] != data['slab']['none'])
    segm = (((data['segmentation'] != 0) * (dist1 < dist2)).astype('int8') + (data['segmentation'] != 0).astype('int8'))

# vizualizace 1
#    pyed = sed3.sed3(segm)
#    pyed.show()
#    import pdb; pdb.set_trace()
#    pyed = sed3.sed3(data['data3d'], contour=segm)
#    pyed.show()
#    import pdb; pdb.set_trace()

# vizualizace 2
    linie = np.abs(dist1 - dist2) < 1
    pyed = sed3.sed3(data['data3d'], contour = data['segmentation']==data['slab']['liver'] ,seeds = linie)
    pyed.show()

    #show3.show3(data['segmentation'])








def get_biggest_object(data):
    """ Return biggest object """
    lab, num = scipy.ndimage.label(data)
    #print("bum = "+str(num))

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
# import numpy as np
from matplotlib.patches import Polygon  # , PathPatch
from mpl_toolkits.mplot3d import art3d
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
    data = misc.obj_from_file("vessels.pickle", filetype = 'pickle')
    #ds = data['segmentation'] == data['slab']['liver']
    #pyed = sed3.sed3(data['segmentation'])
    #pyed.show()
    resection(data)

#    SectorDisplay2__()

