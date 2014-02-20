#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" LISA - organ segmentation tool. """

# from scipy.io import loadmat, savemat
import scipy
from scipy import ndimage
import numpy as np

import sys
import os
import os.path as op

from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
    QGridLayout, QLabel, QPushButton, QFrame, QFileDialog,\
    QFont, QPixmap, QComboBox
from PyQt4.Qt import QString

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))

import dcmreaddata as dcmreader
from seed_editor_qt import QTSeedEditor
import pycut
#from seg2fem import gen_mesh_from_voxels, gen_mesh_from_voxels_mc
#from viewer import QVTKViewer
import qmisc
import misc
import config
import datareader
import datawriter

from seg2mesh import gen_mesh_from_voxels, mesh2vtk, smooth_mesh

try:
    from viewer import QVTKViewer
    viewer3D_available = True

except ImportError:
    viewer3D_available = False

import time
#import audiosupport
import argparse
import logging
logger = logging.getLogger(__name__)

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
    #new versions
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
        volume_unit='ml'
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

        #print segparams
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

        if data3d is None or metadata is None:
            # if 'datapath' in self.iparams:
            #     datapath = self.iparams['datapath']

            if datapath is not None:
                reader = datareader.DataReader()
                self.data3d, self.metadata = reader.Get3DData(datapath)
                #self.iparams['series_number'] = self.metadata['series_number']
                # self.iparams['datapath'] = datapath
                self.process_dicom_data()
            else:
# data will be selected from gui
                pass
                #logger.error('No input path or 3d data')

        else:
            self.data3d = data3d
            # default values are updated in next line
            self.metadata = {'series_number': -1,
                             'voxelsize_mm': 1,
                             'datapath': None}
            self.metadata.update(metadata)
            self.process_dicom_data()

            # self.iparams['series_number'] = self.metadata['series_number']
            # self.iparams['datapath'] = self.metadata['datapath']
        #self.process_dicom_data()

    def process_wvx_size_mm(self):

        #vx_size = self.working_voxelsize_mm
        vx_size = self.input_wvx_size
        if vx_size == 'orig':
            vx_size = self.metadata['voxelsize_mm']

        elif vx_size == 'orig*2':
            vx_size = np.array(self.metadata['voxelsize_mm']) * 2

        elif vx_size == 'orig*4':
            vx_size = np.array(self.metadata['voxelsize_mm']) * 4

        if np.isscalar(vx_size):
            vx_size = ([vx_size] * 3)

        vx_size = np.array(vx_size).astype(float)

       # if np.isscalar(vx_sizey):
       #     vx_size = (np.ones([3]) *vx_size).astype(float)

        # self.iparams['working_voxelsize_mm'] = vx_size
        self.working_voxelsize_mm = vx_size
        #return vx_size

    def process_dicom_data(self):
        # voxelsize processing
        #self.parameters = {}

        #self.segparams['pairwise_alpha']=25

        if self.roi is not None:
            self.crop(self.roi)
            #self.data3d = qmisc.crop(self.data3d, self.roi)
            #self.crinfo = self.roi
            # self.iparams['roi'] = self.roi
            # self.iparams['manualroi'] = False

        self.voxelsize_mm = np.array(self.metadata['voxelsize_mm'])
        self.process_wvx_size_mm()
        self.autocrop_margin = self.autocrop_margin_mm / self.voxelsize_mm
        self.zoom = self.voxelsize_mm / (1.0 * self.working_voxelsize_mm)
        self.orig_shape = self.data3d.shape
        self.segmentation = np.zeros(self.data3d.shape, dtype=np.int8)

        #self.segparams = {'pairwiseAlpha':2, 'use_boundary_penalties':True,
        #'boundary_penalties_sigma':50}

        # for each mm on boundary there will be sum of penalty equal 10

        self.segparams['pairwise_alpha'] = \
            self.segparams['pairwise_alpha_per_mm2'] / \
            np.mean(self.working_voxelsize_mm)

        if self.seeds is None:
            self.seeds = np.zeros(self.data3d.shape, dtype=np.int8)
        logger.info('dir ' + str(self.datapath) + ", series_number" +
                    str(self.metadata['series_number']) + 'voxelsize_mm' +
                    str(self.voxelsize_mm))
        self.time_start = time.time()

    def crop(self, tmpcrinfo):
        """
        Function makes crop of 3d data and seeds and stores it in crinfo.

        tmpcrinfo: temporary crop information

        """
        #print ('sedds ', str(self.seeds.shape), ' se ',
        #       str(self.segmentation.shape), ' d3d ', str(self.data3d.shape))
        self.data3d = qmisc.crop(self.data3d, tmpcrinfo)
# No, size of seeds should be same as data3d
        if self.seeds is not None:
            self.seeds = qmisc.crop(self.seeds, tmpcrinfo)

        if self.segmentation is not None:
            self.segmentation = qmisc.crop(self.segmentation, tmpcrinfo)

        self.crinfo = qmisc.combinecrinfo(self.crinfo, tmpcrinfo)
        logger.debug("crinfo " + str(self.crinfo))

        #print '----sedds ', self.seeds.shape, ' se ',
#self.segmentation.shape,\
        #        ' d3d ', self.data3d.shape

    def _interactivity_begin(self):
        logger.debug('_interactivity_begin()')
        #print 'zoom ', self.zoom
        #print 'svs_mm ', self.working_voxelsize_mm
        data3d_res = ndimage.zoom(
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
            #self.data3d,
            data3d_res,
            segparams=self.segparams,
            voxelsize=self.working_voxelsize_mm,
            modelparams=self.segmodelparams,
            volume_unit='ml'
            #voxelsize=self.voxelsize_mm
        )

        igc.modelparams = self.segmodelparams
# @TODO uncomment this for kernel model
#        igc.modelparams = {
#            'type': 'kernel',
#            'params': {}
#        }
        # if self.iparams['seeds'] is not None:
        if self.seeds is not None:
            seeds_res = ndimage.zoom(
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

# @TODO odstranit hack pro oříznutí na stejnou velikost
# v podstatě je to vyřešeno, ale nechalo by se to dělat elegantněji v zoom
# tam je bohužel patrně bug
        #print 'd3d ', self.data3d.shape
        #print 's orig scale shape ', segm_orig_scale.shape
        shp = [
            np.min([segm_orig_scale.shape[0], self.data3d.shape[0]]),
            np.min([segm_orig_scale.shape[1], self.data3d.shape[1]]),
            np.min([segm_orig_scale.shape[2], self.data3d.shape[2]]),
        ]
        #self.data3d = self.data3d[0:shp[0], 0:shp[1], 0:shp[2]]
        #import ipdb; ipdb.set_trace() # BREAKPOINT

        self.segmentation = np.zeros(self.data3d.shape, dtype=np.int8)
        self.segmentation[
            0:shp[0],
            0:shp[1],
            0:shp[2]] = segm_orig_scale[0:shp[0], 0:shp[1], 0:shp[2]]

        del segm_orig_scale

        # self.iparams['seeds'] = np.zeros(self.data3d.shape, dtype=np.int8)
        # self.iparams['seeds'][
        #     0:shp[0],
        #     0:shp[1],
        #     0:shp[2]] = seeds[0:shp[0], 0:shp[1], 0:shp[2]]

        if self.segmentation_smoothing:
            self.segm_smoothing(self.smoothing_mm)

        #print 'autocrop', self.autocrop
        if self.autocrop is True:
            #print
            #import pdb; pdb.set_trace()

            tmpcrinfo = qmisc.crinfo_from_specific_data(
                self.segmentation,
                self.autocrop_margin)

            self.crop(tmpcrinfo)

        #oseg = self
        #print 'ms d3d ', oseg.data3d.shape
        #print 'ms seg ', oseg.segmentation.shape
        #print 'crinfo ', oseg.crinfo
            #self.segmentation = qmisc.crop(self.segmentation, tmpcrinfo)
            #self.data3d = qmisc.crop(self.data3d, tmpcrinfo)

            #self.crinfo = qmisc.combinecrinfo(self.crinfo, tmpcrinfo)

        if self.texture_analysis not in (None, False):
            import texture_analysis
            # doplnit nějaký kód, parametry atd
            #self.orig_scale_segmentation =
            # texture_analysis.segmentation(self.data3d,
            # self.orig_scale_segmentation, params = self.texture_analysis)
            self.segmentation = texture_analysis.segmentation(
                self.data3d,
                self.segmentation,
                self.voxelsize_mm
            )

        # set label number
#!!! pomaly!!!
# @TODO make faster
        self.segmentation[self.segmentation == 1] = self.output_label
#
        self.processing_time = time.time() - self.time_start

#    def interactivity(self, min_val=800, max_val=1300):
    def interactivity(self, min_val=None, max_val=None):
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
        #import pdb; pdb.set_trace()
        igc = self._interactivity_begin()
        #igc.interactivity()
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
        data['processing_time'] = self.processing_time
# TODO add dcmfilelist
        #data["metadata"] = self.metadata
        #import pdb; pdb.set_trace()
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

# xx and yy are 200x200 tables containing the x and y coordinates as values
# mgrid is a mesh creation helper
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
            #import pdb; pdb.set_trace()

            slicetmp[slicecircle == 1] = label

            self.seeds[slicen, :, :] = slicetmp

#, QMainWindow
            #import py3DSeedEditor
            #rr=py3DSeedEditor.py3DSeedEditor(self.seeds); rr.show()

            #from seed_editor_qt import QTSeedEditor
            #from PyQt4.QtGui import QApplication
            #app = QApplication(sys.argv)
            #pyed = QTSeedEditor(circle)
            #pyed.exec_()

            #app.exit()
            #tmpslice = #np.logical_and(
            #circle < (6400 + 60), circle > (6400 - 60))

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
        pth, filename = op.split(op.normpath(self.datapath))
        filename += "-" + self.experiment_caption
#        if savestring in ['a', 'A']:
# save renamed file too
        filepath = 'org-' + filename + '.pklz'
        #print filepath
        #print 'op ', op
        filepath = op.join(odp, filepath)
        filepath = misc.suggest_filename(filepath)
        misc.obj_to_file(data, filepath, filetype='pklz')

        filepath = 'organ_last.pklz'
        filepath = op.join(odp, filepath)
        #filepath = misc.suggest_filename(filepath)
        misc.obj_to_file(data, filepath, filetype='pklz')

#        iparams = self.get_iparams()
        # filepath = 'organ_iparams.pklz'
        # filepath = op.join(odp, filepath)
        # misc.obj_to_file(iparams, filepath, filetype='pklz')

        #if savestring in ['a', 'A']:
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
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        odp = self.output_datapath
        pth, filename = op.split(op.normpath(self.datapath))
        filename += "-" + self.experiment_caption
        #if savestring in ['ad']:
            # save to DICOM
        filepath = 'dicom-' + filename
        filepath = os.path.join(odp, filepath)
        filepath = misc.suggest_filename(filepath)
        output_dicom_dir = filepath
        data = self.export()
        #import ipdb; ipdb.set_trace()  # BREAKPOINT
        overlays = {
            3:
            (data['segmentation'] == self.output_label).astype(np.int8)
        }
        datawriter.saveOverlayToDicomCopy(self.metadata['dcmfilelist'],
                                          output_dicom_dir, overlays,
                                          data['crinfo'], data['orig_shape'])


# GUI
class OrganSegmentationWindow(QMainWindow):

    def __init__(self, oseg=None):

        self.oseg = oseg

        QMainWindow.__init__(self)
        self.initUI()

        if oseg is not None:
            if oseg.data3d is not None:
                self.setLabelText(self.text_dcm_dir, self.oseg.datapath)
                self.setLabelText(self.text_dcm_data, self.getDcmInfo())

        self.statusBar().showMessage('Ready')

    def initUI(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        grid = QGridLayout()
        grid.setSpacing(15)

        # status bar
        self.statusBar().showMessage('Ready')

        font_label = QFont()
        font_label.setBold(True)
        font_info = QFont()
        font_info.setItalic(True)
        font_info.setPixelSize(10)

        #############
        ### LISA logo
        # font_title = QFont()
        # font_title.setBold(True)
        # font_title.setSize(24)

        lisa_title = QLabel('LIver Surgery Analyser')
        info = QLabel('Developed by:\n' +
                      'University of West Bohemia\n' +
                      'Faculty of Applied Sciences\n' +
                      QString.fromUtf8('M. Jiřík, V. Lukeš - 2013') +
                      '\n\nVersion: ' + self.oseg.version
                      )
        info.setFont(font_info)
        lisa_title.setFont(font_label)
        lisa_logo = QLabel()
        logopath = os.path.join(path_to_script, "../applications/LISA256.png")
        logo = QPixmap(logopath)
        lisa_logo.setPixmap(logo)
        grid.addWidget(lisa_title, 0, 1)
        grid.addWidget(info, 1, 1)
        grid.addWidget(lisa_logo, 0, 2, 2, 1)
        grid.setColumnMinimumWidth(1, logo.width())

        ### dicom reader
        rstart = 2
        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        text_dcm = QLabel('DICOM reader')
        text_dcm.setFont(font_label)
        btn_dcmdir = QPushButton("Load DICOM", self)
        btn_dcmdir.clicked.connect(self.loadDcmDir)
        btn_dcmcrop = QPushButton("Crop", self)
        btn_dcmcrop.clicked.connect(self.cropDcm)

        # voxelsize gui comment
        #self.scaling_mode = 'original'
        #combo_vs = QComboBox(self)
        #combo_vs.activated[str].connect(self.changeVoxelSize)
        #keys = scaling_modes.keys()
        #keys.sort()
        #combo_vs.addItems(keys)
        #combo_vs.setCurrentIndex(keys.index(self.scaling_mode))
        #self.text_vs = QLabel('Voxel size:')
        # end-- voxelsize gui
        self.text_dcm_dir = QLabel('DICOM dir:')
        self.text_dcm_data = QLabel('DICOM data:')
        grid.addWidget(hr, rstart + 0, 0, 1, 4)
        grid.addWidget(text_dcm, rstart + 1, 1, 1, 2)
        grid.addWidget(btn_dcmdir, rstart + 2, 1)
        grid.addWidget(btn_dcmcrop, rstart + 2, 2)
        # voxelsize gui comment
        # grid.addWidget(self.text_vs, rstart + 3, 1)
        # grid.addWidget(combo_vs, rstart + 4, 1)
        grid.addWidget(self.text_dcm_dir, rstart + 5, 1, 1, 2)
        grid.addWidget(self.text_dcm_data, rstart + 6, 1, 1, 2)
        rstart += 8

        # ################ segmentation
        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        text_seg = QLabel('Segmentation')
        text_seg.setFont(font_label)
        btn_mask = QPushButton("Mask region", self)
        btn_mask.clicked.connect(self.maskRegion)
        btn_segauto = QPushButton("Automatic seg.", self)
        btn_segauto.clicked.connect(self.autoSeg)
        btn_segman = QPushButton("Manual seg.", self)
        btn_segman.clicked.connect(self.manualSeg)
        self.text_seg_data = QLabel('segmented data:')
        grid.addWidget(hr, rstart + 0, 0, 1, 4)
        grid.addWidget(text_seg, rstart + 1, 1)
        grid.addWidget(btn_mask, rstart + 2, 1)
        grid.addWidget(btn_segauto, rstart + 3, 1)
        grid.addWidget(btn_segman, rstart + 3, 2)
        grid.addWidget(self.text_seg_data, rstart + 4, 1, 1, 2)
        rstart += 5

        # ################ save/view
        # hr = QFrame()
        # hr.setFrameShape(QFrame.HLine)
        btn_segsave = QPushButton("Save", self)
        btn_segsave.clicked.connect(self.saveOut)
        btn_segsavedcm = QPushButton("Save Dicom", self)
        btn_segsavedcm.clicked.connect(self.saveOutDcm)
        btn_segview = QPushButton("View3D", self)
        if viewer3D_available:
            btn_segview.clicked.connect(self.view3D)

        else:
            btn_segview.setEnabled(False)

        grid.addWidget(btn_segsave, rstart + 0, 1)
        grid.addWidget(btn_segview, rstart + 0, 2)
        grid.addWidget(btn_segsavedcm, rstart + 1, 1)
        rstart += 2

        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        grid.addWidget(hr, rstart + 0, 0, 1, 4)

        # quit
        btn_quit = QPushButton("Quit", self)
        btn_quit.clicked.connect(self.quit)
        grid.addWidget(btn_quit, rstart + 1, 1, 1, 2)

        cw.setLayout(grid)
        self.setWindowTitle('LISA')
        self.show()

    def quit(self, event):
        self.close()

    def changeVoxelSize(self, val):
        self.scaling_mode = str(val)

    def setLabelText(self, obj, text):
        dlab = str(obj.text())
        obj.setText(dlab[:dlab.find(':')] + ': %s' % text)

    def getDcmInfo(self):
        vx_size = self.oseg.voxelsize_mm
        vsize = tuple([float(ii) for ii in vx_size])
        ret = ' %dx%dx%d,  %fx%fx%f mm' % (self.oseg.data3d.shape + vsize)

        return ret

    # def setVoxelVolume(self, vxs):
    #     self.voxel_volume = np.prod(vxs)

    def loadDcmDir(self):
        self.statusBar().showMessage('Reading DICOM directory...')
        QApplication.processEvents()

        oseg = self.oseg
        if oseg.datapath is None:
            oseg.datapath = dcmreader.get_dcmdir_qt(
                app=True,
                directory=self.oseg.input_datapath_start
            )

        if oseg.datapath is None:
            self.statusBar().showMessage('No DICOM directory specified!')
            return

        reader = datareader.DataReader()

        oseg.data3d, oseg.metadata = reader.Get3DData(oseg.datapath)
        # self.iparams['series_number'] = self.metadata['series_number']
        # self.iparams['datapath'] = self.datapath
        oseg.process_dicom_data()
        self.setLabelText(self.text_dcm_dir, oseg.datapath)
        self.setLabelText(self.text_dcm_data, self.getDcmInfo())
        self.statusBar().showMessage('Ready')

    def cropDcm(self):
        oseg = self.oseg

        if oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.statusBar().showMessage('Cropping DICOM data...')
        QApplication.processEvents()

        pyed = QTSeedEditor(oseg.data3d, mode='crop',
                            voxelSize=oseg.voxelsize_mm)
        # @TODO
        mx = self.oseg.viewermax
        mn = self.oseg.viewermin
        width = mx - mn
        #center = (float(mx)-float(mn))
        center = np.average([mx, mn])
        logger.debug("window params max %f min %f width, %f center %f" %
                     (mx, mn, width, center))
        pyed.changeC(center)
        pyed.changeW(width)
        pyed.exec_()

        crinfo = pyed.getROI()
        if crinfo is not None:
            tmpcrinfo = []
            for ii in crinfo:
                tmpcrinfo.append([ii.start, ii.stop])

            #oseg.data3d = qmisc.crop(oseg.data3d, oseg.crinfo)
            oseg.crop(tmpcrinfo)

        self.setLabelText(self.text_dcm_data, self.getDcmInfo())
        self.statusBar().showMessage('Ready')

    def maskRegion(self):
        if self.oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.statusBar().showMessage('Mask region...')
        QApplication.processEvents()

        pyed = QTSeedEditor(self.oseg.data3d, mode='mask',
                            voxelSize=self.oseg.voxelsize_mm)

        mx = self.oseg.viewermax
        mn = self.oseg.viewermin
        width = mx - mn
        #center = (float(mx)-float(mn))
        center = np.average([mx, mn])
        logger.debug("window params max %f min %f width, %f center %f" %
                     (mx, mn, width, center))
        pyed.changeC(center)
        pyed.changeW(width)
        pyed.exec_()

        self.statusBar().showMessage('Ready')

    def autoSeg(self):
        if self.oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.oseg.interactivity(
            min_val=self.oseg.viewermin,
            max_val=self.oseg.viewermax)
        self.checkSegData('auto. seg., ')

    def manualSeg(self):
        oseg = self.oseg
        #print 'ms d3d ', oseg.data3d.shape
        #print 'ms seg ', oseg.segmentation.shape
        #print 'crinfo ', oseg.crinfo
        if oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        pyed = QTSeedEditor(oseg.data3d,
                            seeds=oseg.segmentation,
                            mode='draw',
                            voxelSize=oseg.voxelsize_mm, volume_unit='ml')
        pyed.exec_()

        oseg.segmentation = pyed.getSeeds()
        self.oseg.processing_time = time.time() - self.oseg.time_start
        self.checkSegData('manual seg., ')

    def checkSegData(self, msg):
        oseg = self.oseg
        if oseg.segmentation is None:
            self.statusBar().showMessage('No segmentation!')
            return

        nzs = oseg.segmentation.nonzero()
        nn = nzs[0].shape[0]
        if nn > 0:
            voxelvolume_mm3 = np.prod(oseg.voxelsize_mm)
            tim = self.oseg.processing_time

            if self.oseg.volume_unit == 'ml':
                import datetime
                timstr = str(datetime.timedelta(seconds=round(tim)))
                logger.debug('tim = ' + str(tim))
                aux = 'volume = %.2f [ml] , time = %s' %\
                      (nn * voxelvolume_mm3 / 1000, timstr)
            else:
                aux = 'volume = %.6e mm3' % (nn * voxelvolume_mm3, )
            self.setLabelText(self.text_seg_data, msg + aux)
            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('No segmentation!')

    def saveOut(self, event=None, filename=None):
        if self.oseg.segmentation is not None:
            self.statusBar().showMessage('Saving segmentation data...')
            QApplication.processEvents()

            # if filename is None:
            #     filename = \
            #         str(QFileDialog.getSaveFileName(self,
            #                                         'Save SEG file',
            #                                         filter='Files (*.seg)'))

            # if len(filename) > 0:

            #     outdata = {'data': self.dcm_3Ddata,
            #                'segdata': self.segmentation_data,
            #                'voxelsize_mm': self.voxel_sizemm,
            #                'offset_mm': self.dcm_offsetmm}

            #     if self.segmentation_seeds is not None:
            #         outdata['segseeds'] = self.segmentation_seeds

            #     savemat(filename, outdata, appendmat=False)

            # else:
            #     self.statusBar().showMessage('No output file specified!')

            self.oseg.save_outputs()
            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('No segmentation data!')

    def saveOutDcm(self, event=None, filename=None):
        if self.oseg.segmentation is not None:
            self.statusBar().showMessage('Saving segmentation data...')
            QApplication.processEvents()

            self.oseg.save_outputs_dcm()
            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('No segmentation data!')

    def view3D(self):
        #from seg2mesh import gen_mesh_from_voxels, mesh2vtk, smooth_mesh
        #from viewer import QVTKViewer
        oseg = self.oseg
        if oseg.segmentation is not None:
            pts, els, et = gen_mesh_from_voxels(oseg.segmentation,
                                                oseg.voxelsize_mm,
                                                etype='q', mtype='s')
            pts = smooth_mesh(pts, els, et,
                              n_iter=10)
            vtkdata = mesh2vtk(pts, els, et)
            view = QVTKViewer(vtk_data=vtkdata)
            view.exec_()

        else:
            self.statusBar().showMessage('No segmentation data!')


def main():

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

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
        #'config_version':[1,1]
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

    # input parser
    parser = argparse.ArgumentParser(
        description='Segment vessels from liver \n\
                \npython organ_segmentation.py\n\
                \npython organ_segmentation.py -mroi -vs 0.6')
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
    parser.add_argument('-exd', '--exampledata', action='store_true',
                        help='run unittest')
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
    args_obj = parser.parse_args()

    # next two lines brings cfg from file over input parser. This is why there
    # is no need to have cfg param in input arguments
    args = cfg
    args.update(vars(args_obj))
    #print args["arg"]
    oseg_argspec_keys = config.get_function_keys(OrganSegmentation.__init__)

    if args["debug"]:
        logger.setLevel(logging.DEBUG)

    if args["exampledata"]:
        args["datapath"] = \
            '../sample_data/matlab/examples/sample_data/DICOM/digest_article/'

    app = QApplication(sys.argv)

    if args["iparams"] is not None:
        params = misc.obj_from_file(args["iparams"], filetype='pickle')

    else:
        params = config.subdict(args, oseg_argspec_keys)

    logger.debug('params ' + str(params))
    oseg = OrganSegmentation(**params)

    oseg_w = OrganSegmentationWindow(oseg)

    #oseg.interactivity(args["viewermin"], args["viewermax"])

    #audiosupport.beep()
    # print(
    #     "Volume " +
    #     str(oseg.get_segmented_volume_size_mm3() / 1000000.0) + ' [l]')
    # #pyed = py3DSeedEditor.py3DSeedEditor(oseg.data3d, contour =
    # # oseg.segmentation)
    # #pyed.show()
    # print("Total time: " + str(oseg.processing_time))

    # if args["show_output"]:
    #     oseg.show_output()

    #print savestring
    #  save_outputs(args, oseg, qt_app)
#    import pdb; pdb.set_trace()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    print "Thank you for using Lisa"
