# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
LISA - organ segmentation tool.

'Liver Surgery Analyser

    npython organ_segmentation.py
    npython organ_segmentation.py -mroi -vs 0.6
"""

import logging
logger = logging.getLogger(__name__)

import sys
import os
import os.path as op

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))

import exceptionProcessing

# from scipy.io import loadmat, savemat
import scipy
import scipy.ndimage
import numpy as np

# import dcmreaddata as dcmreader
import pycut
# from seg2fem import gen_mesh_from_voxels, gen_mesh_from_voxels_mc
# from viewer import QVTKViewer
import qmisc
import misc
import config
import datareader
import datawriter

import time
# import audiosupport
import argparse
# import skimage
# import skimage.transform

scaling_modes = {
    'original': (None, None, None),
    'double': (None, 'x2', 'x2'),
    '3mm': (None, '3', '3')
}

#  Defaultparameters for segmentation

# version comparison
from pkg_resources import parse_version
import sklearn
if parse_version(sklearn.__version__) > parse_version('0.10'):
    # new versions
    cvtype_name = 'covariance_type'
else:
    cvtype_name = 'cvtype'

default_segmodelparams = {
    'type': 'gmmsame',
    'params': {cvtype_name: 'full', 'n_components': 3}
}

default_segparams = {'pairwise_alpha_per_mm2': 40,
                     'use_boundary_penalties': False,
                     'boundary_penalties_sigma': 50}

config_version = [1, 0, 0]


def import_gui():
    # from lisaWindow import OrganSegmentationWindow
    # from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
    #     QGridLayout, QLabel, QPushButton, QFrame, \
    #     QFont, QPixmap
    # from PyQt4.Qt import QString
    # from seed_editor_qt import QTSeedEditor
    pass


class OrganSegmentation():
    def __init__(
        self,
        datapath=None,
        working_voxelsize_mm=3,
        viewermax=None,
        viewermin=None,
        series_number=None,
        autocrop=True,
        autocrop_margin_mm=[10, 10, 10],
        manualroi=False,
        texture_analysis=None,
        segmentation_smoothing=False,
        smoothing_mm=4,
        volume_blowup=1.00,
        data3d=None,
        metadata=None,
        seeds=None,
        edit_data=False,
        segparams={},
        segmodelparams=default_segmodelparams,
        roi=None,
        output_label=1,
        slab={},
        output_datapath=None,
        input_datapath_start='',
        experiment_caption='',
        lisa_operator_identifier='',
        volume_unit='ml',
        save_filetype='pklz',

        #           iparams=None,
    ):
        """ Segmentation of objects from CT data.

        datapath: path to directory with dicom files
        manualroi: manual set of ROI before data processing, there is a
             problem with correct coordinates
        data3d, metadata: it can be used for data loading not from directory.
            If both are setted, datapath is ignored
        output_label: label for output segmented volume
        slab: aditional label system for description segmented data
        {'none':0, 'liver':1, 'lesions':6}
        roi: region of interest. [[startx, stopx], [sty, spy], [stz, spz]]
        seeds: ndimage array with size same as data3d
        experiment_caption = this caption is used for naming of outputs
        lisa_operator_identifier: used for logging
        input_datapath_start: Path where user directory selection dialog
            starts.
        volume_blowup: Blow up volume is computed in smoothing so it is working
            only if smoothing is turned on.

        """

        self.iparams = {}
        self.datapath = datapath
        self.output_datapath = output_datapath
        self.input_datapath_start = input_datapath_start
        self.crinfo = [[0, -1], [0, -1], [0, -1]]
        self.slab = slab
        self.output_label = output_label
        self.working_voxelsize_mm = None
        self.input_wvx_size = working_voxelsize_mm

        # print segparams
# @TODO each axis independent alpha
        self.segparams = default_segparams
        self.segparams.update(segparams)
        self.segmodelparams = default_segmodelparams
        self.segmodelparams.update(segmodelparams)
        self.series_number = series_number

        self.autocrop = autocrop
        self.autocrop_margin_mm = np.array(autocrop_margin_mm)
        self.texture_analysis = texture_analysis
        self.segmentation_smoothing = segmentation_smoothing
        self.smoothing_mm = smoothing_mm
        self.volume_blowup = volume_blowup
        self.edit_data = edit_data
        self.roi = roi
        self.data3d = data3d
        self.seeds = seeds
        self.segmentation = None
        self.processing_time = None
        self.experiment_caption = experiment_caption
        self.lisa_operator_identifier = lisa_operator_identifier
        self.version = qmisc.getVersionString()
        self.viewermax = viewermax
        self.viewermin = viewermin
        self.volume_unit = volume_unit
        self.organ_interactivity_counter = 0
        self.dcmfilelist = None
        self.save_filetype = save_filetype
        self.vessel_tree = {}

#
        oseg_input_params = locals()
        oseg_input_params = self.__clean_oseg_input_params(oseg_input_params)

        logger.debug("oseg_input_params")
        logger.debug(str(oseg_input_params))
        self.oseg_input_params = oseg_input_params

        if data3d is None or metadata is None:
            # if 'datapath' in self.iparams:
            #     datapath = self.iparams['datapath']

            if datapath is not None:
                reader = datareader.DataReader()
                datap = reader.Get3DData(datapath, dataplus_format=True)
                # self.iparams['series_number'] = metadata['series_number']
                # self.iparams['datapath'] = datapath
                self.import_dataplus(datap)
            else:
                # data will be selected from gui
                pass
                # logger.error('No input path or 3d data')

        else:
            # self.data3d = data3d
            # default values are updated in next line
            mindatap = {'series_number': -1,
                        'voxelsize_mm': 1,
                        'datapath': None,
                        'data3d': data3d
                        }

            mindatap.update(metadata)
            self.import_dataplus(mindatap)

            # self.iparams['series_number'] = self.metadata['series_number']
            # self.iparams['datapath'] = self.metadata['datapath']
        # self.import_dataplus()

    # def importDataPlus(self, datap):
    #    """
    #    Function for input data
    #    """
    #    self.data3d = datap['data3d']
    #    self.crinfo = datap['crinfo']
    #    self.segmentation = datap['segmentation']
    #    self.slab = datap['slab']
    #    self.voxelsize_mm = datap['voxelsize_mm']
    #    self.orig_shape = datap['orig_shape']
    #    self.seeds = datap[
    #        'processing_information']['organ_segmentation']['seeds']

    def __clean_oseg_input_params(self, oseg_params):
        """
        Used for storing input params of organ segmentation. Big data are not
        stored due to big  memory usage.
        """
        oseg_params['data3d'] = None
        oseg_params['segmentation'] = None
        oseg_params.pop('self')
        return oseg_params

    def process_wvx_size_mm(self, metadata):

        # vx_size = self.working_voxelsize_mm
        vx_size = self.input_wvx_size
        if vx_size == 'orig':
            vx_size = metadata['voxelsize_mm']

        elif vx_size == 'orig*2':
            vx_size = np.array(metadata['voxelsize_mm']) * 2

        elif vx_size == 'orig*4':
            vx_size = np.array(metadata['voxelsize_mm']) * 4

        if np.isscalar(vx_size):
            vx_size = ([vx_size] * 3)

        vx_size = np.array(vx_size).astype(float)

        # if np.isscalar(vx_sizey):
        #     vx_size = (np.ones([3]) *vx_size).astype(float)

        # self.iparams['working_voxelsize_mm'] = vx_size
        self.working_voxelsize_mm = vx_size
        # return vx_size

    def __volume_blowup_criterial_funcion(self, threshold, wanted_volume,
                                          segmentation_smooth
                                          ):

        segm = (1.0 * segmentation_smooth > threshold).astype(np.int8)
        vol2 = np.sum(segm)
        criterium = (wanted_volume - vol2) ** 2
        return criterium

    def segm_smoothing(self, sigma_mm):
        """
        Shape of output segmentation is smoothed with gaussian filter.

        Sigma is computed in mm

        """
        # import scipy.ndimage
        sigma = float(sigma_mm) / np.array(self.voxelsize_mm)

        # print sigma
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        vol1 = np.sum(self.segmentation)
        wvol = vol1 * self.volume_blowup
        segsmooth = scipy.ndimage.filters.gaussian_filter(
            self.segmentation.astype(np.float32), sigma)
        # import ipdb; ipdb.set_trace()
        # import pdb; pdb.set_trace()
        # pyed = py3DSeedEditor.py3DSeedEditor(self.orig_scale_segmentation)
        # pyed.show()

        critf = lambda x: self.__volume_blowup_criterial_funcion(x, wvol, # noqa
                                                                 segsmooth)

        thr = scipy.optimize.fmin(critf, x0=0.5, disp=False)[0]

        self.segmentation = (1.0 *
                             (segsmooth > thr)  # self.volume_blowup)
                             ).astype(np.int8)
        vol2 = np.sum(self.segmentation)
        logger.debug("volume ratio " + str(vol2 / float(vol1)))
        # import ipdb; ipdb.set_trace()

    def import_dataplus(self, dataplus):
        datap = {
            'dcmfilelist': None,
        }
        datap.update(dataplus)
        # voxelsize processing
        # self.parameters = {}

        dpkeys = datap.keys()
        # self.segparams['pairwise_alpha']=25
        self.data3d = datap['data3d']

        if self.roi is not None:
            self.crop(self.roi)
            # self.data3d = qmisc.crop(self.data3d, self.roi)
            # self.crinfo = self.roi
            # self.iparams['roi'] = self.roi
            # self.iparams['manualroi'] = False

        self.voxelsize_mm = np.array(datap['voxelsize_mm'])
        self.process_wvx_size_mm(datap)
        self.autocrop_margin = self.autocrop_margin_mm / self.voxelsize_mm
        self.zoom = self.voxelsize_mm / (1.0 * self.working_voxelsize_mm)
        if 'orig_shape' in dpkeys:
            self.orig_shape = datap['orig_shape']
        else:
            self.orig_shape = self.data3d.shape

        if 'crinfo' in dpkeys:
            self.crinfo = datap['crinfo']
        if 'slab' in dpkeys:
            self.slab = datap['slab']

        if ('segmentation' in dpkeys) and datap['segmentation'] is not None:
            self.segmentation = datap['segmentation']
        else:
            self.segmentation = np.zeros(self.data3d.shape, dtype=np.int8)

        self.dcmfilelist = datap['dcmfilelist']
        # self.segparams = {'pairwiseAlpha':2, 'use_boundary_penalties':True,
        # 'boundary_penalties_sigma':50}

        self.segparams['pairwise_alpha'] = \
            self.segparams['pairwise_alpha_per_mm2'] / \
            np.mean(self.working_voxelsize_mm)

        try:
            self.seeds = datap['processing_information'][
                'organ_segmentation']['seeds']
        except:
            logger.debug('seeds not found in dataplus')
            # self.seeds = None

        # for each mm on boundary there will be sum of penalty equal 10

        if self.seeds is None:

            logger.debug("Seeds are generated")
            self.seeds = np.zeros(self.data3d.shape, dtype=np.int8)
        logger.debug("unique seeds labels " + str(np.unique(self.seeds)))
        logger.info('dir ' + str(self.datapath) + ", series_number" +
                    str(datap['series_number']) + 'voxelsize_mm' +
                    str(self.voxelsize_mm))

        # try read prev information about time processing
        try:
            time_prev = datap['processing_information']['processing_time']
            self.processing_time = time_prev
            self.time_start = time.time() - time_prev
        except:
            self.time_start = time.time()

    def crop(self, tmpcrinfo):
        """
        Function makes crop of 3d data and seeds and stores it in crinfo.

        tmpcrinfo: temporary crop information

        """
        # print ('sedds ', str(self.seeds.shape), ' se ',
        #       str(self.segmentation.shape), ' d3d ', str(self.data3d.shape))
        self.data3d = qmisc.crop(self.data3d, tmpcrinfo)
# No, size of seeds should be same as data3d
        if self.seeds is not None:
            self.seeds = qmisc.crop(self.seeds, tmpcrinfo)

        if self.segmentation is not None:
            self.segmentation = qmisc.crop(self.segmentation, tmpcrinfo)

        self.crinfo = qmisc.combinecrinfo(self.crinfo, tmpcrinfo)
        logger.debug("crinfo " + str(self.crinfo))

        # print '----sedds ', self.seeds.shape, ' se ',
# self.segmentation.shape,\
        #        ' d3d ', self.data3d.shape

    def _interactivity_begin(self):
        logger.debug('_interactivity_begin()')
        # print 'zoom ', self.zoom
        # print 'svs_mm ', self.working_voxelsize_mm
        data3d_res = scipy.ndimage.zoom(
            self.data3d,
            self.zoom,
            mode='nearest',
            order=1
        ).astype(np.int16)
        # data3d_res = data3d_res.astype(np.int16)

        logger.debug('pycut segparams ' + str(self.segparams) +
                     '\nmodelparams ' + str(self.segmodelparams)
                     )
        igc = pycut.ImageGraphCut(
            # self.data3d,
            data3d_res,
            segparams=self.segparams,
            voxelsize=self.working_voxelsize_mm,
            modelparams=self.segmodelparams,
            volume_unit='ml'
            # oxelsize=self.voxelsize_mm
        )

        igc.modelparams = self.segmodelparams
# @TODO uncomment this for kernel model
#        igc.modelparams = {
#            'type': 'kernel',
#            'params': {}
#        }
        # if self.iparams['seeds'] is not None:
        if self.seeds is not None:
            seeds_res = scipy.ndimage.zoom(
                self.seeds,
                self.zoom,
                mode='nearest',
                order=0
            )
            seeds_res = seeds_res.astype(np.int8)
            igc.set_seeds(seeds_res)

        return igc

    def _interactivity_end(self, igc):
        logger.debug('_interactivity_end()')
        # @TODO remove old code in except part
        try:
            # rint 'pred vyjimkou'
            # aise Exception ('test without skimage')
            # rint 'za vyjimkou'
            import skimage
            import skimage.transform
# Now we need reshape  seeds and segmentation to original size

            segm_orig_scale = skimage.transform.resize(
                self.segmentation, self.data3d.shape, order=0)

            seeds = skimage.transform.resize(
                igc.seeds, self.data3d.shape, order=0)

            self.segmentation = segm_orig_scale
            self.seeds = seeds
        except:

            segm_orig_scale = scipy.ndimage.zoom(
                self.segmentation,
                1.0 / self.zoom,
                mode='nearest',
                order=0
            ).astype(np.int8)
            seeds = scipy.ndimage.zoom(
                igc.seeds,
                1.0 / self.zoom,
                mode='nearest',
                order=0
            )
            self.organ_interactivity_counter = igc.interactivity_counter
            logger.debug("org inter counter " +
                         str(self.organ_interactivity_counter))

# @TODO odstranit hack pro oříznutí na stejnou velikost
# v podstatě je to vyřešeno, ale nechalo by se to dělat elegantněji v zoom
# tam je bohužel patrně bug
            # rint 'd3d ', self.data3d.shape
            # rint 's orig scale shape ', segm_orig_scale.shape
            shp = [
                np.min([segm_orig_scale.shape[0], self.data3d.shape[0]]),
                np.min([segm_orig_scale.shape[1], self.data3d.shape[1]]),
                np.min([segm_orig_scale.shape[2], self.data3d.shape[2]]),
            ]
            # elf.data3d = self.data3d[0:shp[0], 0:shp[1], 0:shp[2]]
            # mport ipdb; ipdb.set_trace() # BREAKPOINT

            self.segmentation = np.zeros(self.data3d.shape, dtype=np.int8)
            self.segmentation[
                0:shp[0],
                0:shp[1],
                0:shp[2]] = segm_orig_scale[0:shp[0], 0:shp[1], 0:shp[2]]

            del segm_orig_scale

            self.seeds[
                0:shp[0],
                0:shp[1],
                0:shp[2]] = seeds[0:shp[0], 0:shp[1], 0:shp[2]]

        if self.segmentation_smoothing:
            self.segm_smoothing(self.smoothing_mm)

        # rint 'autocrop', self.autocrop
        if self.autocrop is True:
            # rint
            # mport pdb; pdb.set_trace()

            tmpcrinfo = qmisc.crinfo_from_specific_data(
                self.segmentation,
                self.autocrop_margin)

            self.crop(tmpcrinfo)

        # seg = self
        # rint 'ms d3d ', oseg.data3d.shape
        # rint 'ms seg ', oseg.segmentation.shape
        # rint 'crinfo ', oseg.crinfo
            # elf.segmentation = qmisc.crop(self.segmentation, tmpcrinfo)
            # elf.data3d = qmisc.crop(self.data3d, tmpcrinfo)

            # elf.crinfo = qmisc.combinecrinfo(self.crinfo, tmpcrinfo)

        if self.texture_analysis not in (None, False):
            import texture_analysis
            # doplnit nějaký kód, parametry atd
            # elf.orig_scale_segmentation =
            # texture_analysis.segmentation(self.data3d,
            # self.orig_scale_segmentation, params = self.texture_analysis)
            self.segmentation = texture_analysis.segmentation(
                self.data3d,
                self.segmentation,
                self.voxelsize_mm
            )

        # set label number
# !! pomaly!!!
# @TODO make faster
        self.segmentation[self.segmentation == 1] = self.output_label
#
        self.processing_time = time.time() - self.time_start

#    def interactivity(self, min_val=800, max_val=1300):
# @TODO generovat QApplication
    def interactivity(self, min_val=None, max_val=None):
        from seed_editor_qt import QTSeedEditor
        import_gui()
        logger.debug('interactivity')
        # if self.edit_data:
        #     self.data3d = self.data_editor(self.data3d)

        igc = self._interactivity_begin()

        pyed = QTSeedEditor(igc.img,
                            seeds=igc.seeds,
                            modeFun=igc.interactivity_loop,
                            voxelSize=igc.voxelsize,
                            volume_unit='ml')

        # set window
        if min_val is None:
            min_val = np.min(self.data3d)

        if max_val is None:
            max_val = np.max(self.data3d)

        window_c = ((max_val + min_val) / 2)
        window_w = (max_val - min_val)

        pyed.changeC(window_c)
        pyed.changeW(window_w)

        pyed.exec_()

# @TODO někde v igc.interactivity() dochází k přehození nul za jedničy,
# tady se to řeší hackem
        if igc.segmentation is not None:
            self.segmentation = (igc.segmentation == 0).astype(np.int8)
        self._interactivity_end(igc)

    def ninteractivity(self):
        """Function for automatic (noninteractiv) mode."""
        # mport pdb; pdb.set_trace()
        igc = self._interactivity_begin()
        # gc.interactivity()
        igc.make_gc()
        self.segmentation = (igc.segmentation == 0).astype(np.int8)
        self._interactivity_end(igc)

    def export(self):
        slab = {}
        slab['none'] = 0
        slab['liver'] = 1
        slab['lesions'] = 6
        slab.update(self.slab)

        data = {}
        data['version'] = (1, 0, 1)
        data['data3d'] = self.data3d
        data['crinfo'] = self.crinfo
        data['segmentation'] = self.segmentation
        data['slab'] = slab
        data['voxelsize_mm'] = self.voxelsize_mm
        data['orig_shape'] = self.orig_shape
        data['vessel_tree'] = self.vessel_tree
        processing_information = {
            'organ_segmentation': {
                'processing_time': self.processing_time,
                'oseg_input_params': self.oseg_input_params,
                'organ_interactivity_counter':
                self.organ_interactivity_counter,
                'seeds': self.seeds  # qmisc.SparseMatrix(self.seeds)
            }
        }
        data['processing_information'] = processing_information
# TODO add dcmfilelist
        logger.debug("export()")
        # ogger.debug(str(data))
        logger.debug("org int ctr " + str(self.organ_interactivity_counter))
        # ata["metadata"] = self.metadata
        # mport pdb; pdb.set_trace()
        return data

    # def get_iparams(self):
    #     self.iparams['seeds'] = qmisc.SparseMatrix(self.iparams['seeds'])

    #     return self.iparams

    def add_seeds_mm(self, x_mm, y_mm, z_mm, label, radius):
        """
        Function add circle seeds to one slice with defined radius.

        It is possible set more seeds on one slice with one dimension

        x_mm, y_mm coordinates of circle in mm. It may be array.
        z_mm = slice coordinates  in mm. It may be array
        label: one number. 1 is object seed, 2 is background seed
        radius: is radius of circle in mm

        """

        x_mm = np.array(x_mm)
        y_mm = np.array(y_mm)
        z_mm = np.array(z_mm)

        for i in range(0, len(x_mm)):

            # xx and yy are 200x200 tables containing the x and y coordinates
            # values. mgrid is a mesh creation helper
            xx, yy = np.mgrid[
                :self.seeds.shape[1],
                :self.seeds.shape[2]
            ]
        # circles contains the squared distance to the (100, 100) point
        # we are just using the circle equation learnt at school
            circle = (
                (xx - x_mm[i] / self.voxelsize_mm[1]) ** 2 +
                (yy - y_mm[i] / self.voxelsize_mm[2]) ** 2
            ) ** (0.5)
        # donuts contains 1's and 0's organized in a donut shape
        # you apply 2 thresholds on circle to define the shape
            # slice jen s jednim kruhem
            slicecircle = circle < radius
            slicen = int(z_mm / self.voxelsize_mm[0])
            # slice s tim co už je v něm nastaveno
            slicetmp = self.seeds[slicen, :, :]
            # mport pdb; pdb.set_trace()

            slicetmp[slicecircle == 1] = label

            self.seeds[slicen, :, :] = slicetmp

#  QMainWindow
            # mport py3DSeedEditor
            # r=py3DSeedEditor.py3DSeedEditor(self.seeds); rr.show()

            # rom seed_editor_qt import QTSeedEditor
            # rom PyQt4.QtGui import QApplication
            # pp = QApplication(sys.argv)
            # yed = QTSeedEditor(circle)
            # yed.exec_()

            # pp.exit()
            # mpslice = # p.logical_and(
            # ircle < (6400 + 60), circle > (6400 - 60))

    def lesionsLocalization(self):
        """ Localization of lession """
        import lesions
        tumory = lesions.Lesions()
        # tumory.overlay_test()
        data = self.export()
        tumory.import_data(data)
        tumory.automatic_localization()

        self.segmentation = tumory.segmentation

    def portalVeinSegmentation(self):

        import segmentation
        outputSegmentation = segmentation.vesselSegmentation(
            self.data3d,
            self.segmentation,
            threshold=-1,
            inputSigma=0.15,
            dilationIterations=2,
            nObj=1,
            biggestObjects=False,
            useSeedsOfCompactObjects=True,
            interactivity=True,
            binaryClosingIterations=2,
            binaryOpeningIterations=0)
        slab = {'porta': 2}
        slab.update(self.slab)
        # rom PyQt4.QtCore import pyqtRemoveInputHook
        # yqtRemoveInputHook()
        # mport ipdb; ipdb.set_trace() # BREAKPOINT
        self.slab = slab
        self.segmentation[outputSegmentation == 1] = slab['porta']

        self.__vesselTree(outputSegmentation, 'porta')

    def __vesselTree(self, binaryData3d, textLabel):
        import skelet3d
        import skeleton_analyser  # histology_analyser as skan
        data3d_thr = (binaryData3d > 0).astype(np.int8)
        data3d_skel = skelet3d.skelet3d(data3d_thr)

        skan = skeleton_analyser.SkeletonAnalyser(
            data3d_skel,
            volume_data=data3d_thr,
            voxelsize_mm=self.voxelsize_mm)
        stats = skan.skeleton_analysis(guiUpdateFunction=None)

        if 'graph' not in self.vessel_tree.keys():
            self.vessel_tree['voxelsize_mm'] = self.voxelsize_mm
            self.vessel_tree['graph'] = {}

        self.vessel_tree['graph'][textLabel] = stats
        # print sa.stats
# save skeleton to special file
        misc.obj_to_file(self.vessel_tree, 'vessel_tree.yaml', filetype='yaml')

    def hepaticVeinsSegmentation(self):

        import segmentation
        outputSegmentation = segmentation.vesselSegmentation(
            self.data3d,
            self.segmentation,
            threshold=-1,
            inputSigma=0.15,
            dilationIterations=2,
            nObj=1,
            biggestObjects=False,
            useSeedsOfCompactObjects=True,
            interactivity=True,
            binaryClosingIterations=2,
            binaryOpeningIterations=0)
        slab = {'hepatic_veins': 3}
        slab.update(self.slab)
        # rom PyQt4.QtCore import pyqtRemoveInputHook
        # yqtRemoveInputHook()
        # mport ipdb; ipdb.set_trace() # BREAKPOINT
        self.slab = slab
        self.segmentation[outputSegmentation == 1] = slab['hepatic_veins']

# skeletonizace
        self.__vesselTree(outputSegmentation, 'hepatic_veins')

    def get_segmented_volume_size_mm3(self):
        """Compute segmented volume in mm3, based on subsampeled data."""

        voxelvolume_mm3 = np.prod(self.voxelsize_mm)
        volume_mm3 = np.sum(self.segmentation > 0) * voxelvolume_mm3
        return volume_mm3

    def save_outputs(self):

        # savestring_qt, ok = QInputDialog.getText(
        #     None,
        #     "Save",
        #     'Save output data? Yes/No/All with input data (y/n/a):',
        #     text="a"
        #     )

        # savestring = str(savestring_qt)

        #    if savestring in ['Y', 'y', 'a', 'A']:
        odp = self.output_datapath
        if not op.exists(odp):
            os.makedirs(odp)

        data = self.export()
        data['version'] = self.version  # qmisc.getVersionString()
        data['experiment_caption'] = self.experiment_caption
        data['lisa_operator_identifier'] = self.lisa_operator_identifier
#       data['organ_interactivity_counter'] = self.organ_interactivity_counter
        pth, filename = op.split(op.normpath(self.datapath))
        filename += "-" + self.experiment_caption
#        if savestring in ['a', 'A']:
# save renamed file too
        filepath = 'org-' + filename + '.' + self.save_filetype
        # rint filepath
        # rint 'op ', op
        filepath = op.join(odp, filepath)
        filepath = misc.suggest_filename(filepath)
        misc.obj_to_file(data, filepath, filetype=self.save_filetype)

        filepath = 'organ_last.' + self.save_filetype
        filepath = op.join(odp, filepath)
        # ilepath = misc.suggest_filename(filepath)
        misc.obj_to_file(data, filepath, filetype=self.save_filetype)
# save to mat

#        iparams = self.get_iparams()
        # filepath = 'organ_iparams.pklz'
        # filepath = op.join(odp, filepath)
        # misc.obj_to_file(iparams, filepath, filetype='pklz')

        # f savestring in ['a', 'A']:
        if False:
            # save renamed file too
            data['data3d'] = None
            filepath = 'organ_small-' + filename + '.pklz'
            filepath = op.join(odp, filepath)
            filepath = misc.suggest_filename(filepath)
            misc.obj_to_file(data, filepath, filetype='pklz')

    def save_outputs_dcm(self):
        # TODO add
        logger.debug('save dcm')
        from PyQt4.QtCore import pyqtRemoveInputHook
        pyqtRemoveInputHook()
        # mport ipdb; ipdb.set_trace() # BREAKPOINT
        odp = self.output_datapath
        pth, filename = op.split(op.normpath(self.datapath))
        filename += "-" + self.experiment_caption
        # f savestring in ['ad']:
        #       save to DICOM
        filepath = 'dicom-' + filename
        filepath = os.path.join(odp, filepath)
        filepath = misc.suggest_filename(filepath)
        output_dicom_dir = filepath
        data = self.export()
        # mport ipdb; ipdb.set_trace()  # BREAKPOINT
        overlays = {
            3:
            (data['segmentation'] == self.output_label).astype(np.int8)
        }
        if self.dcmfilelist is not None:
            datawriter.saveOverlayToDicomCopy(
                self.dcmfilelist,
                output_dicom_dir, overlays,
                data['crinfo'], data['orig_shape'])


def logger_init():
    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    fh = logging.FileHandler('lisa.log')

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.debug('logger started')


def lisa_config_init():
    """
    Generate default config from function paramteres.
    Specific config given by command line argument is implemented in
    parser_init() function.
    """
    # read confguraton from file, use default values from OrganSegmentation
    cfg = config.get_default_function_config(OrganSegmentation.__init__)

    # for parameters without support in OrganSegmentation or to overpower
    # default OrganSegmentation values use cfgplus
    cfgplus = {
        'datapath': None,
        'viewermax': 225,
        'viewermin': -125,
        'output_datapath': os.path.expanduser("~/lisa_data"),
        'input_datapath_start': os.path.expanduser("~/lisa_data")
        # config_version':[1,1]
    }

    cfg.update(cfgplus)
    # now is in cfg default values

    cfg = config.get_config("organ_segmentation.config", cfg)
    user_config_path = os.path.join(cfg['output_datapath'],
                                    "organ_segmentation.config")
    config.check_config_version_and_remove_old_records(
        user_config_path, version=config_version,
        records_to_save=['experiment_caption', 'lisa_operator_identifier'])
    # read user defined config in user data
    cfg = config.get_config(user_config_path, cfg)
    return cfg


def parser_init(cfg):

    # input parser
    conf_parser = argparse.ArgumentParser(
        # Turn off help, so we print all options in response to -h
        add_help=False
    )
    conf_parser.add_argument(
        '-cf', '--configfile', default=None,
        help="Use another config. It is loaded after default \
config and user config.")
# Read alternative config file. First is loaded default config. Then user
# config in lisa_data directory. After that is readed config defined by
# --configfile parameter
    knownargs, unknownargs = conf_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser],
        # print script description with -h/--help
        description=__doc__,
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    if knownargs.configfile is not None:
        cfg = config.get_config(knownargs.configfile, cfg)

    parser.add_argument('-dd', '--datapath',
                        default=cfg["datapath"],
                        help='path to data dir')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='run in debug mode')
    parser.add_argument(
        '-vs', '--working_voxelsize_mm',
        default=cfg["working_voxelsize_mm"],
        type=eval,  # type=str,
        help='Insert working voxelsize. It can be number or \
        array of three numbers. It is possible use original \n \
        resolution or half of original resolution. \n \
        -vs 3 \n \
        -vs [3,3,5] \n \
        -vs orig \n \
        -vs orig*2 \n \
        -vs orig*4 \n \
        '
    )
    parser.add_argument('-mroi', '--manualroi', action='store_true',
                        help='manual crop before data processing',
                        default=cfg["manualroi"])

    parser.add_argument('-op', '--output_datapath',
                        default=cfg["output_datapath"],
                        help='path for output data')

    parser.add_argument('-ol', '--output_label', default=1,
                        help='label for segmented data')
    parser.add_argument(
        '--slab',
        default=cfg["slab"],
        type=eval,
        help='labels for segmentation,\
            example -slab "{\'liver\':1, \'lesions\':6}"')
    parser.add_argument(
        '-acr', '--autocrop',
        help='automatic crop after data processing',
        default=cfg["autocrop"])
    parser.add_argument(
        '-iparams', '--iparams',
        default=None,
        help='filename of ipars file with stored interactivity')
    parser.add_argument(
        '-sp', '--segparams',
        default=cfg["segparams"],
        help='params for segmentation,\
            example -sp "{\'pairwise_alpha_per_mm2\':90}"')
    parser.add_argument(
        '-tx', '--texture_analysis', action='store_true',
        help='run with texture analysis')
    parser.add_argument('-ed', '--edit_data', action='store_true',
                        help='Run data editor')
    parser.add_argument(
        '-vmax', '--viewermax', type=eval,  # type=int,
        help='Maximum of viewer window, set None for automatic maximum.',
        default=cfg["viewermax"])
    parser.add_argument(
        '-vmin', '--viewermin', type=eval,  # type=int,
        help='Minimum of viewer window, set None for automatic minimum.',
        default=cfg["viewermin"])
    parser.add_argument(
        '--roi', type=eval,  # type=int,
        help='Minimum of viewer window, set None for automatic minimum.',
        default=cfg["roi"])
    parser.add_argument(
        '-so', '--show_output', action='store_true',
        help='Show output data in viewer')
    parser.add_argument(
        '-ni', '--no_interactivity', action='store_true',
        help='run in no interactivity mode, seeds must be defined')
    parser.add_argument('-a', '--arg', nargs='+', type=float)
    parser.add_argument(
        '-ec', '--experiment_caption', type=str,  # type=int,
        help='Short caption of experiment. No special characters.',
        default=cfg["experiment_caption"])
    parser.add_argument(
        '-ids', '--input_datapath_start', type=str,  # type=int,
        help='Start datapath for input dialog.',
        default=cfg["input_datapath_start"])
    parser.add_argument(
        '-oi', '--lisa_operator_identifier', type=str,  # type=int,
        help='Identifier of Lisa operator.',
        default=cfg["lisa_operator_identifier"])
    parser.add_argument(
        '-ss',
        '--segmentation_smoothing',
        action='store_true',
        help='Smoothing of output segmentation',
        default=cfg["segmentation_smoothing"]
    )
    parser.add_argument(
        '--save_filetype', type=str,  # type=int,
        help='File type of saving data. It can be pklz(default), pkl or mat',
        default=cfg["save_filetype"])

    args_obj = parser.parse_args()

    # next two lines brings cfg from file over input parser. This is why there
    # is no need to have cfg param in input arguments
    args = cfg
    args.update(vars(args_obj))
    return args


def main():

    #    import ipdb; ipdb.set_trace() # BREAKPOINT
    try:
        logger_init()
        cfg = lisa_config_init()
        args = parser_init(cfg)

        # rint args["arg"]
        oseg_argspec_keys = config.get_function_keys(
            OrganSegmentation.__init__)

        if args["debug"]:
            logger.setLevel(logging.DEBUG)

        if args["iparams"] is not None:
            params = misc.obj_from_file(args["iparams"], filetype='pickle')

        else:
            params = config.subdict(args, oseg_argspec_keys)

        logger.debug('params ' + str(params))
        oseg = OrganSegmentation(**params)

        if args["no_interactivity"]:
            oseg.ninteractivity()
            oseg.save_outputs()
        else:
            # mport_gui()
            from lisaWindow import OrganSegmentationWindow
            from PyQt4.QtGui import QApplication
            app = QApplication(sys.argv)
            oseg_w = OrganSegmentationWindow(oseg) # noqa
#    import pdb; pdb.set_trace()
            sys.exit(app.exec_())

    except Exception as e:
        import traceback
        # mport exceptionProcessing
        exceptionProcessing.reportException(e)
        print traceback.format_exc()
        # aise e


if __name__ == "__main__":
    main()
    print "Thank you for using Lisa"
