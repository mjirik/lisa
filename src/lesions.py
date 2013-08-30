#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import skimage.exposure as skexp

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
import vessel_cut


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

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
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

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def export_data(self):
        self.data['segmentation'] = self.segmentation
        pass

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def _automatic_localization(self, data3d, voxelsize_mm, segmentation, slab):
        """
        Automatic localization made by Tomas Ryba
        """

        #vessels = data3d[segmentation==slab['porta']]
        # print slab
        # pyed = py3DSeedEditor.py3DSeedEditor(data3d, contour=segmentation, seeds=(segmentation==2))
        # pyed.show()
        #
        # segmentation[153:180,70:106,42:55] = slab['lesions']
        seeds = self.analyseHistogram()

        return segmentation, slab

    def visualization(self):

        pyed = py3DSeedEditor.py3DSeedEditor(self.data['data3d'], contour = self.data['segmentation']==self.data['slab']['lesions'])
        pyed.show()

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def analyseHistogram(self):
        voxels = self.data3d[np.nonzero(self.segmentation)]
        # hist, bin_edges = np.histogram(voxels, bins=256)
        # peakind = signal.find_peaks_cwt(hist, np.arange(1,10))

        hist, bins = skexp.histogram(voxels)

        max_peak = hist.max()

        plt.figure()
        plt.plot(bins, hist)
        plt.hold(True)

        # plt.plot(peakind, hist[peakind], 'ro')

        plt.show()


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def windowing(self, im, level=50, width=300):
        #srovnani na standardni skalu = odecteni 1024HU
        im -= 1024
        maxHU = level + width
        minHU = level - width

        # #oriznuti cisel pod a nad oknem
        # imw = np.where(im > maxHU, maxHU, im)
        # imw = np.where(im < minHU, minHU, im)
        #
        # #posunuti rozsahu k nule
        # imw -= minHU
        #
        # #uprava rozsahu na interval [0, 255]
        # inw = imw / width * 255

        imw = skexp.rescale_intensity(im, in_range=(minHU, maxHU), out_range=(0, 255))

        return imw



#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    datapath = os.path.join(path_to_script, "../vessels.pkl")
    data = misc.obj_from_file(datapath, filetype = 'pickle')
    #ds = data['segmentation'] == data['slab']['liver']
    #pyed = py3DSeedEditor.py3DSeedEditor(data['segmentation'])
    #pyed.show()
    tumory = Lesions()

    tumory.import_data(data)
    tumory.automatic_localization()
    # tumory.visualization()

#    SectorDisplay2__()

