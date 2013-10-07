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
sys.path.append(os.path.join(path_to_script, "../extern/py3DSeedEditor/"))
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
from skimage.segmentation import random_walker

# version-dependent imports ---------------------------------
from pkg_resources import parse_version

#importing distance transform
if parse_version(scipy.__version__) > parse_version('0.9'):
    from scipy.ndimage.morphology import distance_transform_edt
else:
    from scipy.ndimage import distance_transform_edt
# -----------------------------------------------------------

class Lesions:
    """

    lesions = Lesions(data3d, segmentation, slab)
    lesions.automatic_localization()

    or

    lesions = Lesions()
    lesions.import_data(data)
    lesions.automatic_localization()




    """


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def __init__(self, data3d=None, voxelsize_mm=None, segmentation=None, slab=None):
        self.data3d = data3d
        self.segmentation = segmentation
        self.slab = slab
        self.voxelsize_mm = voxelsize_mm


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
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
        class1 = self.analyseHistogram( debug=True )
        seeds = self.getSeedsUsingClass1(class1)

        liver = self.data3d * (self.segmentation != 0)
        rw = random_walker(liver, seeds, mode='cg_mg')

        #self.segmentation = np.where(class1, self.data['slab']['lesions'], self.segmentation)
        self.segmentation = np.where(rw==2, self.data['slab']['lesions'], self.segmentation)

        #py3DSeedEditor.py3DSeedEditor(self.data3d, contour=(rw==2))
        py3DSeedEditor.py3DSeedEditor(self.data3d, contour=(self.segmentation==self.data['slab']['lesions']))
        plt.show()



        return segmentation, slab


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
#    def getObjects(self):



#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def visualization(self):
        #pyed = py3DSeedEditor.py3DSeedEditor(self.data['data3d'], seeds = self.data['segmentation']==self.data['slab']['lesions'])
        pyed = py3DSeedEditor.py3DSeedEditor(self.segmentation==self.data['slab']['lesions'])
        pyed.show()


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def analyseHistogram(self, debug=False):
        voxels = self.data3d[np.nonzero(self.segmentation)]
        hist, bins = skexp.histogram(voxels)
        max_peakIdx = hist.argmax()

        minT = 0.95 * hist[max_peakIdx]
        maxT = 1.05 * hist[max_peakIdx]
        histTIdxs = (hist >= minT) * (hist <= maxT)
        histTIdxs = np.nonzero(histTIdxs)[0]
        histTIdxs = histTIdxs.astype(np.int)

        class1TMin = bins[histTIdxs[0]]
        class1TMax = bins[histTIdxs[-1]]

        liver = self.data3d * (self.segmentation > 0)
        class1 = np.where( (liver >= class1TMin) * (liver <= class1TMax), 1, 0)

        if debug:
            plt.figure()
            plt.plot(bins, hist)
            plt.hold(True)

            plt.plot(bins[max_peakIdx], hist[max_peakIdx], 'ro')
            plt.plot(bins[histTIdxs], hist[histTIdxs], 'r')
            plt.plot(bins[histTIdxs[0]], hist[histTIdxs[0]], 'rx')
            plt.plot(bins[histTIdxs[-1]], hist[histTIdxs[-1]], 'rx')
            plt.title('Histogram of liver density and its class1 = maximal peak (red dot) +-5% of its density (red line).')
            plt.show()

        return class1


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def getSeedsUsingClass1(self, class1):
        distData = np.where(class1 == 1, False, True)
        distData *= self.segmentation > 0
        dists = distance_transform_edt(distData)

        seeds = dists > 0.5 * dists.max()

        allSeeds = np.zeros(self.data3d.shape, dtype=np.int)
        # allSeeds[np.nonzero(self.segmentation)] = 80
        allSeeds[np.nonzero(class1)] = 1 #zdrava tkan
        allSeeds[np.nonzero(seeds)] = 2 #outliers
        #kvuli segmentaci pomoci random walkera se omezi obraz pouze na segmentovana jatra a cevy
        allSeeds = np.where(self.segmentation == 0, -1, allSeeds)

        return allSeeds


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def windowing(self, im, level=50, width=300):
        #srovnani na standardni skalu = odecteni 1024HU
        im -= 1024

        #zjisteni minimalni a maximalni density
        minHU = level - width
        maxHU = level + width

        #rescalovani intenzity tak, aby skala <minHU, maxHU> odpovidala intervalu <0,255>
        imw = skexp.rescale_intensity(im, in_range=(minHU, maxHU), out_range=(0, 255))

        return imw


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    #datapath = os.path.join(path_to_script, "../vessels1.pkl") #horsi kvalita segmentace
    datapath = os.path.join(path_to_script, "../vessels.pkl") #hypodenzni meta
    # datapath = os.path.join(path_to_script, "../organ.pkl") #horsi kvalita segmentace
    data = misc.obj_from_file(datapath, filetype = 'pickle')
    #ds = data['segmentation'] == data['slab']['liver']
    #pyed = py3DSeedEditor.py3DSeedEditor(data['segmentation'])
    #pyed.show()
    tumory = Lesions()

    tumory.import_data(data)
    tumory.automatic_localization()
    # tumory.visualization()

#    SectorDisplay2__()

