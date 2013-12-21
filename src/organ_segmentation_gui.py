#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LISA - organ segmentation tool
"""

# from scipy.io import loadmat, savemat
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
from seg2mesh import gen_mesh_from_voxels, mesh2vtk, smooth_mesh
from viewer import QVTKViewer

import time
#import audiosupport
import argparse
import logging
logger = logging.getLogger(__name__)

scaling_modes = {
    'original': (None, None, None),
    'double': (None, 'x2', 'x2'),
    '3mm': (None, '3', '3'),
    }

class OrganSegmentation():
    def __init__(
        self,
        datapath=None,
        working_voxelsize_mm=3,
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
        roi=None,
        #           iparams=None,
        output_label=1,
        slab={},
        output_datapath=None,
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

        """
        self.iparams = {}
        self.datapath = datapath
        self.output_datapath = output_datapath
        self.crinfo = [[0, -1], [0, -1], [0, -1]]
        self.slab = slab
        self.output_label = output_label
        self.working_voxelsize_mm = working_voxelsize_mm
        self.segparams = segparams
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

        if data3d is None or metadata is None:
            # if 'datapath' in self.iparams:
            #     datapath = self.iparams['datapath']

            if datapath is not None:
                reader = datareader.DataReader()
                self.data3d, self.metadata = reader.Get3DData(datapath)
                # self.iparams['series_number'] = self.metadata['series_number']
                # self.iparams['datapath'] = datapath
                self.process_dicom_data()

        else:
            self.data3d = data3d
            # default values are updated in next line
            self.metadata = {'series_number': -1,
                             'voxelsize_mm': 1,
                             'datapath': None}
            self.metadata.update(metadata)

            # self.iparams['series_number'] = self.metadata['series_number']
            # self.iparams['datapath'] = self.metadata['datapath']

    def process_dicom_data(self):
        # voxelsize processing
        vx_size = self.working_voxelsize_mm
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
        #self.parameters = {}

        #self.segparams = {'pairwiseAlpha':2, 'use_boundary_penalties':True,
        #'boundary_penalties_sigma':50}

        # for each mm on boundary there will be sum of penalty equal 10
        segparams = {'pairwise_alpha_per_mm2': 10,
                     'use_boundary_penalties': False,
                     'boundary_penalties_sigma': 50}
        segparams = {'pairwise_alpha_per_mm2': 40,
                     'use_boundary_penalties': False,
                     'boundary_penalties_sigma': 50}
        #print segparams
# @TODO each axis independent alpha
        self.segparams.update(segparams)

        self.segparams['pairwise_alpha'] = \
            self.segparams['pairwise_alpha_per_mm2'] / \
            np.mean(self.working_voxelsize_mm)

        #self.segparams['pairwise_alpha']=25

        if self.roi is not None:
            self.data3d = qmisc.crop(self.data3d, self.roi)
            self.crinfo = self.roi
            # self.iparams['roi'] = self.roi
            # self.iparams['manualroi'] = False

        self.voxelsize_mm = np.array(self.metadata['voxelsize_mm'])
        self.autocrop_margin = self.autocrop_margin_mm / self.voxelsize_mm
        self.zoom = self.voxelsize_mm / (1.0 * self.working_voxelsize_mm)
        self.orig_shape = self.data3d.shape
        self.segmentation = np.zeros(self.data3d.shape, dtype=np.int8)

        # @TODO use logger
        print 'dir ', self.datapath, ", series_number",\
            self.metadata['series_number'], 'voxelsize_mm',\
            self.voxelsize_mm
        self.time_start = time.time()

    def _interactivity_begin(self):
        logger.debug('_interactivity_begin()')
        # print 'zoom ', self.zoom
        # print 'svs_mm ', self.working_voxelsize_mm
        # data3d_res = ndimage.zoom(
        #     self.data3d,
        #     self.zoom,
        #     mode='nearest',
        #     order=1
        # )
        # data3d_res = data3d_res.astype(np.int16)

        igc = pycut.ImageGraphCut(
            self.data3d,
            #           gcparams={'pairwise_alpha': 30},
            segparams=self.segparams,
            #voxelsize=self.working_voxelsize_mm
            voxelsize=self.voxelsize_mm
        )

        # version comparison
        from pkg_resources import parse_version
        import sklearn
        if parse_version(sklearn.__version__) > parse_version('0.10'):
            #new versions
            cvtype_name = 'covariance_type'
        else:
            cvtype_name = 'cvtype'

        igc.modelparams = {
            'type': 'gmmsame',
            'params': {cvtype_name: 'full', 'n_components': 3}
        }
        # if self.iparams['seeds'] is not None:
        #     seeds_res = ndimage.zoom(
        #         self.iparams['seeds'],
        #         self.zoom,
        #         mode='nearest',
        #         order=0
        #     )
        #     igc.set_seeds(seeds_res)

        return igc

    def _interactivity_end(self, igc):
        logger.debug('_interactivity_end()')
#        ndimage.zoom(
#                self.segmentation,
#                1.0 / self.zoom,
#                output=segm_orig_scale,
#                mode='nearest',
#                order=0
#                )

        self.processing_time = time.time() - self.time_start
        return
        segm_orig_scale = ndimage.zoom(
            self.segmentation,
            1.0 / self.zoom,
            mode='nearest',
            order=0
        ).astype(np.int8)
        # seeds = ndimage.zoom(
        #     igc.seeds,
        #     1.0 / self.zoom,
        #     mode='nearest',
        #     order=0
        # )

# @TODO odstranit hack pro oříznutí na stejnou velikost
# v podstatě je to vyřešeno, ale nechalo by se to dělat elegantněji v zoom
# tam je bohužel patrně bug
        shp = [
            np.min([segm_orig_scale.shape[0], self.data3d.shape[0]]),
            np.min([segm_orig_scale.shape[1], self.data3d.shape[1]]),
            np.min([segm_orig_scale.shape[2], self.data3d.shape[2]]),
        ]
        #self.data3d = self.data3d[0:shp[0], 0:shp[1], 0:shp[2]]

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

        #print 'crinfo: ', self.crinfo
        #print 'autocrop', self.autocrop
        if self.autocrop is True:
            #print
            #import pdb; pdb.set_trace()

            tmpcrinfo = self._crinfo_from_specific_data(
                self.segmentation,
                self.autocrop_margin)
            self.segmentation = self._crop(self.segmentation, tmpcrinfo)
            self.data3d = self._crop(self.data3d, tmpcrinfo)

            self.crinfo = qmisc.combinecrinfo(self.crinfo, tmpcrinfo)

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
                            voxelSize=igc.voxelsize)

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
        data['version'] = qmisc.getVersionString()
        pth, filename = op.split(op.normpath(self.datapath))
#        if savestring in ['a', 'A']:
# save renamed file too
            # filepath = 'organ_big-' + filename + '.pklz'
            # filepath = op.join(op, filename)
            # filepath = misc.suggest_filename(filepath)
            # misc.obj_to_file(data, filepath, filetype='pklz')

        filepath = 'organ.pklz'
        filepath = op.join(odp, filepath)
        #filepath = misc.suggest_filename(filepath)
        misc.obj_to_file(data, filepath, filetype='pklz')

#        iparams = self.get_iparams()
        # filepath = 'organ_iparams.pklz'
        # filepath = op.join(odp, filepath)
        # misc.obj_to_file(iparams, filepath, filetype='pklz')

        data['data3d'] = None
        filepath = 'organ_small-' + filename + '.pklz'
        filepath = op.join(odp, filepath)
        filepath = misc.suggest_filename(filepath)
        misc.obj_to_file(data, filepath, filetype='pklz')

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

        lisa_title = QLabel('LIver Surgery Analyser v0.9')
        info = QLabel('Developed by:\n' +
                      'University of West Bohemia\n' +
                      'Faculty of Applied Sciences\n' +
                      QString.fromUtf8('M. Jiřík, V. Lukeš - 2013'))
        info.setFont(font_info)
        lisa_title.setFont(font_label)
        lisa_logo = QLabel()
        logo = QPixmap("../applications/LISA256.png")
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
        self.scaling_mode = 'original'
        combo_vs = QComboBox(self)
        combo_vs.activated[str].connect(self.changeVoxelSize)
        keys = scaling_modes.keys()
        keys.sort()
        combo_vs.addItems(keys)
        combo_vs.setCurrentIndex(keys.index(self.scaling_mode))
        self.text_vs = QLabel('Voxel size:')
        self.text_dcm_dir = QLabel('DICOM dir:')
        self.text_dcm_data = QLabel('DICOM data:')
        grid.addWidget(hr, rstart + 0, 0, 1, 4)
        grid.addWidget(text_dcm, rstart + 1, 1, 1, 2)
        grid.addWidget(btn_dcmdir, rstart + 2, 1)
        grid.addWidget(btn_dcmcrop, rstart + 2, 2)
        grid.addWidget(self.text_vs, rstart + 3, 1)
        grid.addWidget(combo_vs, rstart + 4, 1)
        grid.addWidget(self.text_dcm_dir, rstart + 5, 1, 1, 2)
        grid.addWidget(self.text_dcm_data, rstart + 6, 1, 1, 2)
        rstart += 8

        # ################ segmentation
        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        text_seg = QLabel('Segmentation')
        text_seg.setFont(font_label)
        btn_segauto = QPushButton("Automatic seg.", self)
        btn_segauto.clicked.connect(self.autoSeg)
        btn_segman = QPushButton("Manual seg.", self)
        btn_segman.clicked.connect(self.manualSeg)
        self.text_seg_data = QLabel('segmented data:')
        grid.addWidget(hr, rstart + 0, 0, 1, 4)
        grid.addWidget(text_seg, rstart + 1, 1)
        grid.addWidget(btn_segauto, rstart + 2, 1)
        grid.addWidget(btn_segman, rstart + 2, 2)
        grid.addWidget(self.text_seg_data, rstart + 3, 1, 1, 2)
        rstart += 4

        # ################ save/view
        # hr = QFrame()
        # hr.setFrameShape(QFrame.HLine)
        btn_segsave = QPushButton("Save", self)
        btn_segsave.clicked.connect(self.saveOut)
        btn_segview = QPushButton("View3D", self)
        btn_segview.clicked.connect(self.view3D)
        grid.addWidget(btn_segsave, rstart + 0, 1)
        grid.addWidget(btn_segview, rstart + 0, 2)
        rstart += 1

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
           oseg.datapath = dcmreader.get_dcmdir_qt(app=True)

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
        pyed.exec_()

        crinfo = pyed.getROI()
        if crinfo is not None:
            oseg.crinfo = []
            for ii in crinfo:
                oseg.crinfo.append([ii.start, ii.stop])

            oseg.data3d = qmisc.crop(oseg.data3d, oseg.crinfo)

        self.setLabelText(self.text_dcm_data, self.getDcmInfo())
        self.statusBar().showMessage('Ready')

    def autoSeg(self):
        if self.oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.oseg.interactivity()
        self.checkSegData('auto. seg., ')

    def manualSeg(self):
        oseg = self.oseg
        if  oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        pyed = QTSeedEditor(oseg.data3d,
                            seeds=oseg.segmentation,
                            mode='draw',
                            voxelSize=oseg.voxelsize_mm)
        pyed.exec_()

        oseg.segmentation = pyed.getSeeds()
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

    def view3D(self):
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

    ## read confguraton from file, use default values from OrganSegmentation
    cfgplus = {
        'datapath': None,
        'viewermax': 300,
        'viewermin': -100,
        'output_datapath': os.path.expanduser("~/lisa_data")
    }

    cfg = config.get_default_function_config(OrganSegmentation.__init__)
    cfg.update(cfgplus)
    # now is in cfg default values

    cfg = config.get_config("organ_segmentation.config", cfg)
    # read user defined config in user data
    cfg = config.get_config(
        os.path.join(cfg['output_datapath'], "organ_segmentation.config"),
        cfg)

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
        '-so', '--show_output', action='store_true',
        help='Show output data in viewer')
    parser.add_argument('-a', '--arg', nargs='+', type=float)
    parser.add_argument(
        '-ss',
        '--segmentation_smoothing',
        action='store_true',
        help='Smoothing of output segmentation',
        default=cfg["segmentation_smoothing"]
    )
    args_obj = parser.parse_args()
    args = vars(args_obj)
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

    oseg = OrganSegmentation(**params)

    oseg_w = OrganSegmentationWindow(oseg)

    #oseg.interactivity(args["viewermin"], args["viewermax"])

    # audiosupport.beep()
    # print(
    #     "Volume " +
    #     str(oseg.get_segmented_volume_size_mm3() / 1000000.0) + ' [l]')
    # #pyed = py3DSeedEditor.py3DSeedEditor(oseg.data3d, contour =
    # # oseg.segmentation)
    # #pyed.show()
    # print("Total time: " + str(oseg.processing_time))

    # if args["show_output"]:
    #     oseg.show_output()

    # #print savestring
    # save_outputs(args, oseg, qt_app)
#    import pdb; pdb.set_trace()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    print "Thank you for using Lisa"
