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

import logging
logger = logging.getLogger(__name__)
import argparse

import numpy as np
import scipy

# ----------------- my scripts --------
import misc
import py3DSeedEditor
import show3
import vessel_cut
import scipy.ndimage.measurements as scimeas
from skimage.morphology import remove_small_objects
from skimage.segmentation import random_walker
from skimage import measure as skmeasure
import scipy.ndimage.morphology as scimorph
from scipy.ndimage import generate_binary_structure
import cv2
from mayavi import mlab

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
        self.min_size_of_comp = 200 #minimal size of object to be considered as lession


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
        class1 = self.analyseHistogram( debug=False )
        seeds = self.getSeedsUsingClass1(class1)

        liver = self.data3d * (self.segmentation != 0)
        print('Starting random walker...')
        rw = random_walker(liver, seeds, mode='cg_mg')
        print('...finished.')

        #self.segmentation = np.where(class1, self.data['slab']['lesions'], self.segmentation)
        label_l = self.data['slab']['lesions']
        label_v = self.data['slab']['porta']

        #self.segmentation = np.where(np.logical_and(rw==2, self.segmentation!=label_v), label_l, self.segmentation)
        #les1 = np.where(np.logical_and(rw==2, self.segmentation!=label_v), label_l, self.segmentation)

        #py3DSeedEditor.py3DSeedEditor(self.data3d, contour=(rw==2))
        #py3DSeedEditor.py3DSeedEditor(self.data3d, contour=(self.segmentation==self.data['slab']['lesions']))
        #plt.show()

        lessions = self.filterObjects(rw==2)

        self.segmentation = np.where(lessions, label_l, self.segmentation)

        #py3DSeedEditor.py3DSeedEditor(self.data3d, contour=lessions, windowW=350, windowC=50).show()
        self.mayavi3dVisualization()

        return segmentation, slab


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def removeObjectsWithLabel(self, objects, label):
        """
        This function removes all objects from <objects> that contains a voxel that is labeled as <label>.
        """
        labels, nlabels = self.getObjects(objects)
        labeledObjs = self.segmentation == label
        notVesselObjects = np.zeros_like(self.segmentation)

        #iterates through all founded objects
        for l in range(1, nlabels+1):
            obj = labels == l
            #count number of voxels that are also labeled ass vessels
            sameVoxs = np.logical_and(labeledObjs, obj).sum()
            #if there are no vessel voxels, then this object is taken into account
            if sameVoxs == 0:
                notVesselObjects += obj

        return notVesselObjects


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def getObjects(self, objects = np.zeros((1),dtype=np.bool)):
        if not objects.any():
            label_l = self.data['slab']['lesions']
            objects = self.segmentation==label_l

        only_big = remove_small_objects(objects, min_size=self.min_size_of_comp, connectivity=1, in_place=False)
        labels, nlabels = scimeas.label(only_big)

        return labels, nlabels


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def getFeatures(self, labels, nlabels):
        features = np.zeros((nlabels, 2))
        for lab in range(1, nlabels+1):
            obj = (labels==lab)
            size = obj.sum()
            strel = np.ones((3,3,3), dtype=np.bool)
            #obj = scimorph.binary_closing(obj, generate_binary_structure(3,3))
            obj = scimorph.binary_closing(obj, strel)
            compactness = self.getZunicsCompatness(obj)
            features[lab-1,0] = size
            features[lab-1,1] = compactness

            print 'size = %i'%(size)
            print 'compactness = %.3f'%(compactness)
            #py3DSeedEditor.py3DSeedEditor(self.data3d, contour=(labels==lab)).show()
        print features[:,1]
        return features


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def filterObjects(self, objects, minComp=0.3):
        """
        This function removes objects that aren't lessions.
        Firstly, it removes objects, that contains voxels labeled as vessels. Then it filters out to small objects and
        finally it calculates object features and removes objects with bad features (e.g. to non-compact etc.).
        """
        #TODO: If lession is near a vessel, it is possible that the lession will have a voxel labeled as vessel and will be filtered out.
        #removing objects that contains voxels labeled as porta
        objects = self.removeObjectsWithLabel(objects, label=self.data['slab']['porta'])
        #removing to small objects
        objects = remove_small_objects(objects, min_size=self.min_size_of_comp, connectivity=1, in_place=False)
        #computing object features
        labels, nlabels = scimeas.label(objects)
        features = self.getFeatures(labels, nlabels)
        #filtering objects with respect to their features
        featuresOK = features[:,1] >= minComp
        objsOK = np.zeros_like(objects)
        for i in np.argwhere(featuresOK>0):
            objsOK += labels == (i+1)

        return objsOK


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def getBounds(self, obj):
        obj_er = scimorph.binary_erosion(obj)
        bounds = obj - obj_er
        return bounds


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def getMoment(self, obj, p, q, r):
        elems = np.argwhere(obj)
        mom = 0
        for el in elems:
            mom += el[0]**p + el[1]**q + el[2]**r
        return mom


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def getCentralMoment(self, obj, p, q, r):
        elems = np.argwhere(obj)
        m000 = obj.sum()
        m100 = (elems[:,0]).sum()
        m010 = (elems[:,1]).sum()
        m001 = (elems[:,2]).sum()
        xc = m100 / m000
        yc = m010 / m000
        zc = m001 / m000

        mom = 0
        for el in elems:
            mom += (el[0] - xc)**p + (el[1] - yc)**q + (el[2] - zc)**r

        #mxyz = elems.sum(axis=0)
        #cent = mxyz / m000
        #mom = 0
        #for el in elems:
        #    mom += (el[0] - cent[0])**p + (el[1] - cent[1])**q + (el[2] - cent[2])**r
        return mom


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def getZunicsCompatness(self, obj):
        m000 = obj.sum()
        m200 = self.getCentralMoment(obj, 2, 0, 0)
        m020 = self.getCentralMoment(obj, 0, 2, 0)
        m002 = self.getCentralMoment(obj, 0, 0, 2)
        term1 = (3**(5./3)) / (5 * (4*np.pi)**(2./3))
        term2 = m000**(5./3) / (m200 + m020 + m002)
        K = term1 * term2
        return K


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def mayavi3dVisualization(self, objs = 'pvl'):
        data = np.zeros_like(self.data3d)
        srcs = list()
        colors = list()
        if 'p' in objs:
            #liver = self.segmentation == self.data['slab']['liver']
            #parenchym = np.logical_or(liver, vessels)
            parenchym = self.segmentation > 0
            data = np.where(parenchym, 1, 0)
            data = data.T
            srcs.append(mlab.pipeline.scalar_field(data))
            colors.append((0,1,0))
        if 'v' in objs:
            vessels = self.segmentation == self.data['slab']['porta']
            data = np.where(vessels, 1, 0)
            data = data.T
            srcs.append(mlab.pipeline.scalar_field(data))
            colors.append((1,0,0))
        if 'l' in objs:
            lessions = self.segmentation == self.data['slab']['lesions']
            data = np.where(lessions, 1, 0)
            data = data.T
            srcs.append(mlab.pipeline.scalar_field(data))
            colors.append((0,0,1))

        for src, col in zip(srcs, colors):
            src.spacing = [.62, .62, 5]
            mlab.pipeline.iso_surface(src, contours=2, opacity=0.1, color=col)

        mlab.show()


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def visualization(self):
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
            default='vessels.pkl',
            help='path to data dir')

    args = parser.parse_args()
    #datapath = os.path.join(path_to_script, "../vessels1.pkl") #horsi kvalita segmentace
    #datapath = os.path.join(path_to_script, args.inputfile) #hypodenzni meta
    #datapath = os.path.join(path_to_script, "../organ.pkl") #horsi kvalita segmentace
    data = misc.obj_from_file(args.inputfile, filetype = 'pickle')
    #ds = data['segmentation'] == data['slab']['liver']
    #pyed = py3DSeedEditor.py3DSeedEditor(data['segmentation'])
    #pyed.show()
    tumory = Lesions()

    tumory.import_data(data)
    tumory.automatic_localization()
    # tumory.visualization()

#    SectorDisplay2__()

if __name__ == "__main__":
    main()
