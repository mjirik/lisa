#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path
import numpy as np
# from scipy import signal
import matplotlib.pyplot as plt
# import skimage.exposure as skexp

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
# import featurevector

import logging
logger = logging.getLogger(__name__)
import argparse

import scipy

# ----------------- my scripts --------
import misc
import sed3

try:
    import dcmreaddata
except:
    from imcut import dcmreaddata


# import show3
# import vessel_cut
import scipy.ndimage.measurements as scimeas
from skimage.morphology import remove_small_objects
import skimage.morphology as skimor
from skimage.segmentation import random_walker
# import skimage.filter as skifil
# from skimage import measure as skmeasure
import scipy.ndimage.morphology as scimorph
# from scipy.ndimage import generate_binary_structure
# import cv2
import tools
# from mayavi import mlab
# import pymorph as pm

# version-dependent imports ---------------------------------
from pkg_resources import parse_version

# importing distance transform
if parse_version(scipy.__version__) > parse_version('0.9'):
    from scipy.ndimage.morphology import distance_transform_edt
else:
    from scipy.ndimage import distance_transform_edt
# -----------------------------------------------------------


class Lesions:
    """

    lesions = Lesions(data3d, voxelsize_mm, segmentation, slab)
    lesions.automatic_localization()

    or

    lesions = Lesions()
    lesions.import_data(data)
    lesions.automatic_localization()




    """

# ----------------------------------------------------------------------------
# ---------------------------------------------------------------------------


    def __init__(self, data3d=None, voxelsize_mm=None, segmentation=None,
                 slab=None):


        self.data3d = data3d
        self.data = None
        self.segmentation = segmentation
        self.slab = slab
        self.voxelsize_mm = voxelsize_mm
        # min size of object to be considered as lession
        # self.min_size_of_comp = 200
        self.min_size_of_comp = 50

# ---------------------------------------------------------------------------
    def run_gui(self):
        import lesioneditor
        import lesioneditor.Lession_editor_slim
        datap1={
            'data3d': self.data3d,
            'segmentation': self.segmentation,
            'slab': self.slab,
            'voxelsize_mm': self.voxelsize_mm
        }
        le = lesioneditor.Lession_editor_slim.LessionEditor(datap1=datap1)
        le.show()
        # get data
        # self.segmentation = le.segmentation
        # self.slab = le.slab

       
# ---------------------------------------------------------------------------
    def import_data(self, data):
        self.data = data
        self.data3d = data['data3d']
        self.segmentation = data['segmentation']
        self.slab = data['slab']
        self.voxelsize_mm = data['voxelsize_mm']

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def automatic_localization(self):
        """
        Automatic localization of lesions. Params from constructor
        or import_data() function.
        """
        self._automatic_localization()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def export_data(self):
        self.data['segmentation'] = self.segmentation
        pass

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def _automatic_localization(self):
        """
        Automatic localization made by Tomas Ryba.
        """

        # seeds = self.get_seeds_using_class_1(class1)
        liver = self.data3d * (self.segmentation != 0)
        print('analyzing histogram...')
        class1 = tools.analyse_histogram(self.data3d,
                                         roi=self.segmentation != 0)
        # sed3.sed3(self.data3d, seeds=class1).show()
        print('getting seeds...')
        seeds = self.get_seeds_using_prob_class1(
            liver,
            class1,
            thresholdType='percOfMaxDist',
            percT=0.3)

        # sed3.sed3(self.data3d, seeds=seeds).show()

        print('Starting random walker...')
        rw = random_walker(liver, seeds, mode='cg_mg')
        print('...finished.')

        label_l = self.data['slab']['lesions']

        lessions = rw == 2
        sed3.sed3(self.data3d, contour=lessions).show()
        lessions = self.filter_objects(lessions)

        self.segmentation = np.where(lessions, label_l, self.segmentation)

        # self.mayavi3dVisualization()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def remove_objects_with_label(self, objects, label):
        """
        This function removes all objects from <objects> that contains a
        voxel that is labeled as <label>.
        """
        labels, nlabels = self.get_objects(objects)
        labeled_objs = self.segmentation == label
        not_vessel_objects = np.zeros_like(self.segmentation)

        # iterates through all founded objects
        for l in range(1, nlabels+1):
            obj = labels == l
            # count number of voxels that are also labeled ass vessels
            same_voxs = np.logical_and(labeled_objs, obj).sum()
            # if there are no vessel voxels, then this object is taken into
            # account
            if same_voxs == 0:
                not_vessel_objects += obj

        return not_vessel_objects

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def get_objects(self, objects=np.zeros(1, dtype=np.bool)):
        if not objects.any():
            label_l = self.data['slab']['l esions']
            objects = self.segmentation == label_l

        only_big = remove_small_objects(objects,
                                        min_size=self.min_size_of_comp,
                                        connectivity=1, in_place=False)
        labels, nlabels = scimeas.label(only_big)

        return labels, nlabels

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def get_features(self, labels, nlabels):
        features = np.zeros((nlabels, 2))
        for lab in range(1, nlabels+1):
            obj = labels == lab
            size = obj.sum()
            strel = np.ones((3, 3, 3), dtype=np.bool)
            # obj =scimorph.binary_closing(obj, generate_binary_structure(3,3))
            obj = scimorph.binary_closing(obj, strel)
            compactness = self.get_zunics_compatness(obj)
            features[lab-1, 0] = size
            features[lab-1, 1] = compactness

            print('size = %i' % size)
            print('compactness = %.3f' % compactness)
            # sed3.sed3(self.data3d, contour=(labels==lab)).show() # noqa
        print(features[:, 1])
        return features

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def filter_objects(self, objects, min_comp=0.3):
        """
        This function removes objects that aren't lessions.
        Firstly, it removes objects, that contains voxels labeled as vessels.
        Then it filters out to small objects and finally it calculates object
        features and removes objects with bad features (e.g. to non-compact
        etc.).
        """
        # TODO: If a lession is near a vessel, it is possible that the lession
        # will have a voxel labeled as vessel and will be filtered out.
        # removing objects that contains voxels labeled as porta
        if 'porta' in self.data['slab']:
            objects = self.remove_objects_with_label(
                objects,
                label=self.data['slab']['porta'])

        # removing to small objects
        print('Small objects removal:')
        _, nlabels = scimeas.label(objects)
        print('    before: nlabels = %i' % nlabels)
        objects = skimor.remove_small_objects(
            objects, min_size=self.min_size_of_comp,
            connectivity=1, in_place=False)
        _, n_labels = scimeas.label(objects)
        print('    after: nlabels = %i' % nlabels)

        # computing object features
        labels, n_labels = scimeas.label(objects)
        features = self.get_features(labels, n_labels)

        print('Non-compact objects removal:')
        print('    before: nlabels = %i' % n_labels)
        # filtering objects with respect to their features
        features_ok = features[:, 1] >= min_comp
        objs_ok = np.zeros_like(objects)
        for i in np.argwhere(features_ok > 0):
            objs_ok += labels == (i + 1)
        print('\tafter: nlabels = %i' % features_ok.sum())

        return objs_ok

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def get_bounds(self, obj):
        obj_er = scimorph.binary_erosion(obj)
        bounds = obj - obj_er
        return bounds

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def get_moment(self, obj, p, q, r):
        elems = np.argwhere(obj)
        mom = 0
        for el in elems:
            mom += el[0]**p + el[1]**q + el[2]**r
        return mom

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def get_central_moment(self, obj, p, q, r):
        elems = np.argwhere(obj)
        m000 = obj.sum()
        m100 = (elems[:, 0]).sum()
        m010 = (elems[:, 1]).sum()
        m001 = (elems[:, 2]).sum()
        xc = m100 / m000
        yc = m010 / m000
        zc = m001 / m000

        mom = 0
        for el in elems:
            mom += (el[0] - xc)**p + (el[1] - yc)**q + (el[2] - zc)**r

        # mxyz = elems.sum(axis=0)
        # cent = mxyz / m000
        # mom = 0
        # for el in elems:
        #     mom += (el[0] - cent[0])**p + (el[1] - cent[1])**q +
        #                                    (el[2] - cent[2])**r
        return mom

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def get_zunics_compatness(self, obj):
        m000 = obj.sum()
        m200 = self.get_central_moment(obj, 2, 0, 0)
        m020 = self.get_central_moment(obj, 0, 2, 0)
        m002 = self.get_central_moment(obj, 0, 0, 2)
        term_1 = (3**(5./3)) / (5 * (4*np.pi)**(2./3))
        term_2 = m000**(5./3) / (m200 + m020 + m002)
        k = term_1 * term_2
        return k

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    # def mayavi3dVisualization(self, objs = 'pvl'):
    #     data = np.zeros_like(self.data3d)
    #     srcs = list()
    #     colors = list()
    #     if 'p' in objs:
    #         #liver = self.segmentation == self.data['slab']['liver']
    #         #parenchym = np.logical_or(liver, vessels)
    #         parenchym = self.segmentation > 0
    #         data = np.where(parenchym, 1, 0)
    #         data = data.T
    #         srcs.append(mlab.pipeline.scalar_field(data))
    #         colors.append((0,1,0))
    #     if 'v' in objs:
    #         vessels = self.segmentation == self.data['slab']['porta']
    #         data = np.where(vessels, 1, 0)
    #         data = data.T
    #         srcs.append(mlab.pipeline.scalar_field(data))
    #         colors.append((1,0,0))
    #     if 'l' in objs:
    #         lessions = self.segmentation == self.data['slab']['lesions']
    #         data = np.where(lessions, 1, 0)
    #         data = data.T
    #         srcs.append(mlab.pipeline.scalar_field(data))
    #         colors.append((0,0,1))
    #
    #     for src, col in zip(srcs, colors):
    #         src.spacing = [.62, .62, 5]
    #         mlab.pipeline.iso_surface(src, contours=2, opacity=0.1,color=col)
    #
    #     mlab.show()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def visualization(self):
        pyed = sed3.sed3(
            self.segmentation == self.data['slab']['lesions'])
        pyed.show()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def get_seeds_using_class_1(self, class1):
        dist_data = np.where(class1 == 1, False, True)
        dist_data *= self.segmentation > 0
        dists = distance_transform_edt(dist_data)

        seeds = dists > 0.5 * dists.max()

        all_seeds = np.zeros(self.data3d.shape, dtype=np.int)
        # allSeeds[np.nonzero(self.segmentation)] = 80
        all_seeds[np.nonzero(class1)] = 1  # zdrava tkan
        all_seeds[np.nonzero(seeds)] = 2  # outliers
        # kvuli segmentaci pomoci random walkera se omezi obraz pouze na
        # segmentovana jatra a cevy
        all_seeds = np.where(self.segmentation == 0, -1, all_seeds)

        return all_seeds

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def get_seeds_using_prob_class1(self, data, class1, roi=None, dens_min=20,
                                    dens_max=255,
                                    thresholdType='percOfMaxDist', percT=0.5):
        # calculates probability based on similarity of intensities
        probs, mu = tools.intensity_probability(data, std=10)
        # sed3.sed3(data).show()
        # sed3.sed3(probs).show()
        # normalizing and calculating reciprocal values
        # weights_ints = skexp.rescale_intensity(probs,
        # in_range=(0,probs.max()), out_range=(1,0))
        weights_ints = np.exp(-probs)

        weights_h = np.where(data > mu, 1 - probs, 0)
        weights_l = np.where(data < mu, 1 - probs, 0)
        # sed3.sed3(1 - probs).show()
        sed3.sed3(weights_h).show()
        sed3.sed3(weights_l).show()

        if roi is None:
            roi = np.logical_and(data >= dens_min, data <= dens_max)
        dist_data = np.where(class1 == 1, False, True)
        dist_data *= roi > 0
        # dists = distance_transform_edt(dist_data)
        # sed3.sed3(dists).show()

        # print 'dists max = %i' % dists.max()
        # print 'dists min = %i' % dists.min()
        # print 'weights_ints max = %.4f' % weights_ints.max()
        # print 'weights_ints min = %.4f' % weights_ints.min()
        # print 'probs max = %.4f' % probs.max()
        # print 'probs min = %.4f' % probs.min()

        # energy = dists * weights_ints
        energy = weights_ints
        # sed3.sed3(energy).show()

        seeds = np.zeros(data.shape, dtype=np.bool)
        if thresholdType == 'percOfMaxDist':
            seeds = energy > (percT * energy.max())
        elif thresholdType == 'mean':
            seeds = energy > 2 * (energy[np.nonzero(energy)]).mean()

        # TODO: tady je problem, ze energy je v intervalu <0.961, 1> - hrozne
        # maly rozsah
        print('energy max = %.4f' % energy.max())
        print('energy min = %.4f' % energy.min())
        print('thresh = %.4f' % (percT * energy.max()))
        print(seeds.min())
        print(seeds.max())
        print('seed perc = %.2f' % (
            (energy > percT * energy.max()).sum()/np.float(energy.nbytes)))
        sed3.sed3(seeds).show()

        # removing to small objects
        min_size_of_seed_area = 60
        print('before removing: %i' % seeds.sum())
        seeds = skimor.remove_small_objects(
            seeds, min_size=min_size_of_seed_area, connectivity=1,
            in_place=False)
        print('after removing: %i' % seeds.sum())

        all_seeds = np.zeros(data.shape, dtype=np.int)
        # allSeeds[np.nonzero(self.segmentation)] = 80
        all_seeds[np.nonzero(class1)] = 1  # zdrava tkan
        all_seeds[np.nonzero(seeds)] = 2  # outliers
        # kvuli segmentaci pomoci random walkera se omezi obraz pouze na
        # segmentovana jatra a cevy
        all_seeds = np.where(roi == 0, -1, all_seeds)

        sed3.sed3(all_seeds).show()

        return all_seeds

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    def overlay_test(self):
        # filename =
        # r'c:\Data\kky\pro mechaniku 21.12.2012\53009707\Export0000\SR0000'
        # filename = r'c:\Data\kky\pro mechaniku
        # 21.12.2012\52496602\Export0001\SR0000'
        # filename = r'c:\Data\kky\pro mechaniku
        # 21.12.2012\54559730\Export0000\SR0000'
        filename = r'c:\Data\kky\pro mechaniku 21.12.2012\53596059\Export0000\SR0000' # noqa
        # dcr = dcmreaddata.DicomReader(
        # r'c:\Data\kky\pro mechaniku 21.12.2012\52496602\Export0001\SR0000')
        dcr = dcmreaddata.DicomReader(filename)
        data3d = dcr.get_3Ddata()
        overlay = dcr.get_overlay()
        win_w = 350
        win_c = 50
        min_w = win_c - (win_w / 2.)
        max_w = win_c + (win_w / 2.)

        for key in overlay.keys():
            overlay = overlay[key]

            for i in range(overlay.shape[0]):
                plt.figure(), plt.gray()
                plt.subplot(121), plt.imshow(data3d[i, :, :], vmin=min_w,
                                             vmax=max_w)
                plt.subplot(122), plt.imshow(overlay[i, :, :])
                cv2.imwrite(filename + '/overlay_%i.png' % i, data3d[i, :, :])

        plt.show()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def main():

    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(
        description='Module for segmentation of simple anatomical structures')
    parser.add_argument('-i', '--inputfile', default='vessels.pkl',
                        help='path to data dir')

    # args = parser.parse_args()
    # horsi kvalita segmentace
    # datapath = os.path.join(path_to_script, "../vessels1.pkl")
    # hypodenzni meta
    # datapath = os.path.join(path_to_script, args.inputfile)
    # horsi kvalita segmentace
    # datapath = os.path.join(path_to_script, "../organ.pkl")

    # data = misc.obj_from_file(args.inputfile, filetype = 'pickle')

    dcmdir = '/home/tomas/Dropbox/Work/Data/medical/org-38289898-export1.pklz'
    data = misc.obj_from_file(dcmdir, filetype='pickle')

    # windowing
    data['data3d'] = tools.windowing(data['data3d'], level=50, width=350)
    # data['data3d'] = smoothing(data['data3d'], sliceId=0)

    # smoothing ----------------
    # bilateral
    # data['data3d'] = tools.smoothing_bilateral(data['data3d'],
    # sigma_space=15, sigma_color=0.05, sliceId=0)
    # more bilateral
    # data['data3d'] = tools.smoothing_bilateral(data['data3d'],
    # sigma_space=15, sigma_color=0.1, sliceId=0)
    # total-variation
    data['data3d'] = tools.smoothing_tv(data['data3d'], weight=0.05, sliceId=0)
    # more total-variation
    # data['data3d'] = tools.smoothing_tv(data['data3d'], weight=0.2,
    # multichannel=False, sliceId=0)
    # sed3.sed3(data['data3d']).show()

    tumory = Lesions()
    # tumory.overlay_test()
    tumory.import_data(data)
    tumory.automatic_localization()

    tumors = tumory.segmentation == tumory.data['slab']['lesions']
    sed3.sed3(tumory.data3d, contour=tumors).show()

#    SectorDisplay2__()

if __name__ == "__main__":
    main()
