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

#import numpy as np
#import scipy.ndimage


import os
import os.path

# ----------------- my scripts --------
import misc
import py3DSeedEditor
import show3
import vessel_cut
import glob

def get_subdirs(dirpath, wildcard = '*'):

    dirlist = []
    if os.path.exists(dirpath):
        logger.info('dirpath = '  + dirpath )
        #print completedirpath
    else:
        logger.error('Wrong path: '  + dirpath )
        raise Exception('Wrong path : ' + dirpath )

    #print 'copmpletedirpath = ', completedirpath

    dirlist = [o for o in os.listdir(dirpath) if os.path.isdir(o)]
    #for infile in glob.glob( os.path.join(dirpath, wildcard) ):
    #    dirlist.append(infile)
    #    #print "current file is: " + infile

    misc.obj_to_file(dirlist, 'experiment_data.yaml','yaml')
    return dirlist


class Lesions:
    """

    lesions = Lesions(data3d, segmentation, slab)
    lesions.automatic_localization()

    or

    lesions = Lesions()
    lesions.import_data(data)
    lesions.automatic_localization()




    """
    def __init__(self, data3d=None, voxelsize_mm=None, segmentation=None, slab=None):
        self.data3d = data3d
        self.segmentation = segmentation
        self.slab = slab
        self.voxelsize_mm = voxelsize_mm
    
    def import_data(self, data):
        self.data = data
        self.data3d = data['data3d']
        self.segmentation = data['segmentation']
        self.slab = data['slab']
        self.voxelsize_mm = data['voxelsize_mm']
    
    def automatic_localization(self):
        """ 
        Automatic localization of lesions. Params from constructor 
        or import_data() function.
        """
        self.segmentation, self.slab = self._automatic_localization(
                self.data3d,
                self.voxelsize_mm,
                self.segmentation,
                self.slab
                )

    def export_data(self):
        self.data['segmentation'] = self.segmentation
        pass

    def _automatic_localization(self, data3d, voxelsize_mm, segmentation, slab):
        """
        Automatic localization made by Tomas Ryba
        """

        #vessels = data3d[segmentation==slab['porta']]
        #pyed = py3DSeedEditor.py3DSeedEditor(vessels)
        #pyed.show()

        segmentation[153:180,70:106,42:55] = slab['lesions']

        return segmentation, slab

    def visualization(self):

        pyed = py3DSeedEditor.py3DSeedEditor(self.data['data3d'], contour = self.data['segmentation']==self.data['slab']['lesions'])
        pyed.show()





if __name__ == "__main__":
    data = misc.obj_from_file("vessels.pickle", filetype = 'pickle')
    #ds = data['segmentation'] == data['slab']['liver']
    #pyed = py3DSeedEditor.py3DSeedEditor(data['segmentation'])
    #pyed.show()
    tumory = Lesions()

    tumory.import_data(data)
    tumory.automatic_localization()
    tumory.visualization()

#    SectorDisplay2__()

