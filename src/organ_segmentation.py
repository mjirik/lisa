#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/py3DSeedEditor/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)


#import apdb
#  apdb.set_trace();
#import scipy.io
import numpy as np
import scipy
#from scipy import sparse
import traceback
import time
import audiosupport

# ----------------- my scripts --------
import py3DSeedEditor
#
#import dcmreaddata as dcmr
import pycut
import argparse
#import py3DSeedEditor

import segmentation
import qmisc
import misc
import datareader
import config
import numbers

def interactive_imcrop(im):

    pass


class OrganSegmentation():
    def __init__(
            self,
            datapath=None,
            working_voxelsize_mm=1,
            series_number=None,
            autocrop=True,
            autocrop_margin_mm=[10, 10, 10],
            manualroi=None,
            texture_analysis=None,
            segmentation_smoothing=True,
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
            qt_app=None
            ):
        """
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


        self.crinfo = [[0, -1], [0, -1], [0, -1]]
        self.slab = slab
        self.output_label = output_label



        self.qt_app = qt_app
        # TODO uninteractive Serie selection
        if data3d is None or metadata is None:

            if self.iparams.has_key('datapath'):
                datapath = self.iparams['datapath']


            #self.data3d, self.metadata = dcmr.dcm_read_from_dir(datapath)
            if datapath == None:
                self.process_qt_app()
                datapath = dcmr.get_dcmdir_qt(self.qt_app)


            # @TODO dialog v qt
            #reader = dcmr.DicomReader(datapath) # , qt_app=qt_app)
            #self.data3d = reader.get_3Ddata()
            #self.metadata = reader.get_metaData()
            reader = datareader.DataReader()
            self.data3d, self.metadata = reader.Get3DData(datapath)
            self.iparams['series_number'] = self.metadata['series_number']
            self.iparams['datapath'] = datapath
        else:
            self.data3d = data3d
            # default values are updated in next line
            self.metadata = {'series_number':-1, 'voxelsize_mm':1, 'datapath':None}
            self.metadata.update(metadata)

            self.iparams['series_number'] = self.metadata['series_number']
            self.iparams['datapath'] = self.metadata['datapath']


            self.orig_shape = self.data3d.shape


        # voxelsize processing
        if working_voxelsize_mm == 'orig':
            working_voxelsize_mm = self.metadata['voxelsize_mm']
        elif working_voxelsize_mm == 'orig*2':
            working_voxelsize_mm = np.array(self.metadata['voxelsize_mm'])*2
        elif working_voxelsize_mm == 'orig*4':
            working_voxelsize_mm = np.array(self.metadata['voxelsize_mm'])*4



        if np.isscalar(working_voxelsize_mm):
            self.working_voxelsize_mm = ([working_voxelsize_mm] * 3)
        else:
            self.working_voxelsize_mm = working_voxelsize_mm


        
        self.working_voxelsize_mm = np.array(self.working_voxelsize_mm).astype(float)

       # if np.isscalar(self.working_voxelsize_mm):
       #     self.working_voxelsize_mm = (np.ones([3]) * self.working_voxelsize_mm).astype(float)

        self.iparams['working_voxelsize_mm'] = self.working_voxelsize_mm


        #self.parameters = {}


        #self.segparams = {'pairwiseAlpha':2, 'use_boundary_penalties':True,'boundary_penalties_sigma':50}

# for each mm on boundary there will be sum of penalty equal 10
        self.segparams = {'pairwise_alpha_per_mm2':10, 'use_boundary_penalties':False,'boundary_penalties_sigma':50}
        self.segparams = {'pairwise_alpha_per_mm2':40, 'use_boundary_penalties':False,'boundary_penalties_sigma':50}
        #print segparams
# @TODO each axis independent alpha
        self.segparams.update(segparams)

        self.segparams['pairwise_alpha'] = \
                self.segparams['pairwise_alpha_per_mm2'] / \
                np.mean(self.iparams['working_voxelsize_mm'])

        #self.segparams['pairwise_alpha']=25
        


        # parameters with same effect as interactivity
        #if iparams is None:
        #    self.iparams= {}
        #else:
        #    self.set_iparams(iparams)


# manualcrop
        if manualroi is not None:
# @todo opravit souřadný systém v součinnosti s autocrop
            self.process_qt_app()
            self.data3d, self.crinfo = qmisc.manualcrop(self.data3d)
            self.iparams['roi'] = self.crinfo
            self.iparams['manualroi'] = None
            #print self.crinfo
        elif roi is not None:
            self.data3d = qmisc.crop(self.data3d, roi)
            self.crinfo = roi
            self.iparams['roi'] = roi
            self.iparams['manualroi'] = None

        if seeds == None:
            self.iparams['seeds'] = np.zeros(self.data3d.shape, dtype=np.int8)
        else:

            if qmisc.isSparseMatrix(seeds):
                seeds = seeds.todense()
            self.iparams['seeds'] = seeds
        self.voxelsize_mm = np.array(self.metadata['voxelsize_mm'])
        self.autocrop = autocrop
        self.autocrop_margin_mm = np.array(autocrop_margin_mm)
        self.autocrop_margin = self.autocrop_margin_mm / self.voxelsize_mm
        self.texture_analysis = texture_analysis
        self.segmentation_smoothing = segmentation_smoothing
        self.smoothing_mm = smoothing_mm
        self.edit_data = edit_data



        self.zoom = self.voxelsize_mm /(1.0 * self.working_voxelsize_mm)

#    def set_iparams(self, iparams):
#        """
#        Set interactivity variables. Make numpy array from scipy sparse 
#        matrix.
#        """
#
#        # seeds may be stored in sparse matrix
#        try:
#            if qmisc.SparseMatrix.issparse(iparams['seeds']):
#                iparams['seeds'] = iparams['seeds'].todense()
#            #import pdb; pdb.set_trace()
#        except:
#            # patrne neni SparseMatrix
#            pass
#
#        self.iparams = iparams
        print 'dir ', self.iparams['datapath'],", series_number", self.iparams['series_number'], 'voxelsize_mm', self.voxelsize_mm

    def process_qt_app(self):
        """
        Sometimes repeatedly called QApplication causes SIGSEGV crash.
        This is why we try call it as few as possible.
        """
        if self.qt_app == None:
            #from PyQt4.QtGui import QApplication
            #import PyQt4
            from PyQt4 import QtGui
#QApplication
            self.qt_app = QtGui.QApplication(sys.argv)


    def get_iparams(self):
        self.iparams['seeds'] = qmisc.SparseMatrix(self.iparams['seeds'])

        return self.iparams

#    def save_ipars(self, filename = 'ipars.pkl'):
#        import misc
#        misc.obj_to_file(self.get_ipars(), filename)
        



    def _interactivity_begin(self):

        print 'zoom ', self.zoom 
        print 'svs_mm ', self.working_voxelsize_mm
        data3d_res = scipy.ndimage.zoom(
                self.data3d,
                self.zoom,
                mode='nearest',
                order=1
                )
        data3d_res = data3d_res.astype(np.int16)

        #print 'data shp',  data3d_res.shape
        #import pdb; pdb.set_trace()
        igc = pycut.ImageGraphCut(
                data3d_res,
#                gcparams={'pairwise_alpha': 30},
                segparams=self.segparams,
                voxelsize=self.working_voxelsize_mm
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
        if not self.iparams['seeds'] == None:
            #igc.seeds= self.seeds
            seeds_res = scipy.ndimage.zoom(
                    self.iparams['seeds'],
                    self.zoom,
                    mode='nearest',
                    order=0
                    )
            seeds_res = seeds_res.astype(np.int8)
            igc.set_seeds(seeds_res)
            #print "nastavujeme seedy"
            #import py3DSeedEditor
            #rr=py3DSeedEditor.py3DSeedEditor(
#   data3d_res, seeds = seeds_res); rr.show()

            #import pdb; pdb.set_trace()

        #print "sh2", self.data3d.shape 
        return igc

    def _interactivity_end(self, igc):
        logger.debug('_interactivity_end')
        #print "sh3", self.data3d.shape 


#        import pdb; pdb.set_trace()
#        scipy.ndimage.zoom(
#                self.segmentation,
#                1.0 / self.zoom,
#                output=segm_orig_scale,
#                mode='nearest',
#                order=0
#                )
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




        #print  np.sum(self.segmentation)*np.prod(self.voxelsize_mm)

# @TODO odstranit hack pro oříznutí na stejnou velikost
# v podstatě je to vyřešeno, ale nechalo by se to dělat elegantněji v zoom
# tam je bohužel patrně bug
        shp = [
                np.min([segm_orig_scale.shape[0], self.data3d.shape[0]]),
                np.min([segm_orig_scale.shape[1], self.data3d.shape[1]]),
                np.min([segm_orig_scale.shape[2], self.data3d.shape[2]]),
                ]
        #self.data3d = self.data3d[0:shp[0], 0:shp[1], 0:shp[2]]

        self.segmentation = np.zeros(self.data3d.shape, dtype=np.int8)
        self.segmentation[
                0:shp[0],
                0:shp[1],
                0:shp[2]] = segm_orig_scale[0:shp[0], 0:shp[1], 0:shp[2]]

        del segm_orig_scale

        self.iparams['seeds'] = np.zeros(self.data3d.shape, dtype=np.int8)
        self.iparams['seeds'][
                0:shp[0],
                0:shp[1],
                0:shp[2]] = seeds[0:shp[0], 0:shp[1], 0:shp[2]]
#
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
        self.segmentation[self.segmentation == 1] = self.output_label
#

    def interactivity(self, min_val=800, max_val=1300):
        #import pdb; pdb.set_trace()
# Staré volání
        #igc = pycut.ImageGraphCut(self.data3d, zoom = self.zoom)
        #igc.gcparams['pairwise_alpha'] = 30
        #seeds_res = scipy.ndimage.zoom(self.seeds , self.zoom,
        # prefilter=False, mode= 'nearest', order = 1)
        #seeds = self.seeds.astype(np.int8)


        logger.debug('interactivity')
        if self.edit_data:
            self.data3d = self.data_editor(self.data3d)


        # set window
        if min_val == -1:
            min_val = np.min(self.data3d)
            #print 'new min'

        if max_val == -1:
            max_val = np.max(self.data3d)
            #print 'new max'

        igc = self._interactivity_begin()
        logger.debug('_interactivity_begin()')
        self.process_qt_app()
        #import pdb; pdb.set_trace()
        igc.interactivity(qt_app = self.qt_app, min_val=min_val, max_val=max_val)
# @TODO někde v igc.interactivity() dochází k přehození nul za jedničy,
# tady se to řeší hackem
        if type (igc.segmentation) is list:
            raise Exception("Interactive object segmentation failed.\
                    You must select seeds.")
        self.segmentation = (igc.segmentation == 0).astype(np.int8)

# correct label

        self._interactivity_end(igc)
        #igc.make_gc()
        #igc.show_segmentation()

    def ninteractivity(self):
        """
        Function for automatic (noninteractiv) mode.
        """
        #import pdb; pdb.set_trace()
# Staré volání
        #igc = pycut.ImageGraphCut(self.data3d, zoom = self.zoom)
        #igc.gcparams['pairwise_alpha'] = 30
        #seeds_res = scipy.ndimage.zoom(self.seeds , self.zoom,
        # prefilter=False, mode= 'nearest', order = 1)
        #seeds = self.seeds.astype(np.int8)
        igc = self._interactivity_begin()
        #igc.interactivity()
        igc.make_gc()
        self.segmentation = (igc.segmentation == 0).astype(np.int8)
        self._interactivity_end(igc)
        #igc.show_segmentation()

    def prepare_output(self):
        pass

    def get_segmented_volume_size_mm3(self):
        """
        Compute segmented volume in mm3, based on subsampeled data
        """
        #voxelvolume_mm3 = np.prod(self.working_voxelsize_mm)
        #volume_mm3_rough = np.sum(self.segmentation > 0) * voxelvolume_mm3

        voxelvolume_mm3 = np.prod(self.voxelsize_mm)
        volume_mm3 = np.sum(self.segmentation > 0) * voxelvolume_mm3
        #import pdb; pdb.set_trace()

        #print " fine = ", volume_mm3
        #print voxelvolume_mm3
        #print volume_mm3
        #import pdb; pdb.set_trace()
        return volume_mm3

    def make_segmentation(self):
        pass

    def set_roi_mm(self, roi_mm):
        pass

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
                    :self.iparams['seeds'].shape[1], 
                    :self.iparams['seeds'].shape[2]
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
            slicetmp = self.iparams['seeds'][slicen, :, :]
            #import pdb; pdb.set_trace()

            slicetmp[slicecircle == 1] = label

            self.iparams['seeds'][slicen, :, :] = slicetmp

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
        pass

    def _crop(self, data, crinfo):
        """
        Crop data with crinfo
        """
        data = qmisc.crop(data, crinfo)
        #data[crinfo[0][0]:crinfo[0][1],
               # crinfo[1][0]:crinfo[1][1], crinfo[2][0]:crinfo[2][1]]
        return data

    def _crinfo_from_specific_data(self, data, margin):
# hledáme automatický ořez, nonzero dá indexy
        return qmisc.crinfo_from_specific_data(data, margin)


    def im_crop(self, im,  roi_start, roi_stop):
        im_out = im[ \
                roi_start[0]:roi_stop[0],\
                roi_start[1]:roi_stop[1],\
                roi_start[2]:roi_stop[2],\
                ]
        return  im_out

    def segm_smoothing(self, sigma_mm):
        """
        shape of output segmentation is smoothed with gaussian filter. Sigma
        is computed in mm
        """
        #print "smoothing"
        sigma = float(sigma_mm) / np.array(self.voxelsize_mm)

        #print sigma
        #import pdb; pdb.set_trace()
        self.segmentation = scipy.ndimage.filters.gaussian_filter(\
                self.segmentation.astype(np.float32), sigma)
        #import pdb; pdb.set_trace()
        #pyed = py3DSeedEditor.py3DSeedEditor(self.orig_scale_segmentation)
        #pyed.show()

        self.segmentation = (1.0 * (self.segmentation > 0.5)).astype(np.int8)

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
        #import pdb; pdb.set_trace()
        return data

    def data_editor(self, im3d, cval=0):
        """
        Funkce provádí změnu vstupních dat - data3d
        cval: hodnota, na kterou se nastaví "vymazaná" data
        """

        from seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication
        import numpy as np
#, QMainWindow
        print ("Select voxels for deletion")
        app = QApplication(sys.argv)
        pyed = QTSeedEditor(im3d, mode='draw')
        pyed.exec_()

        deletemask = pyed.getSeeds()
        #import pdb; pdb.set_trace()

        #pyed = QTSeedEditor(deletemask, mode='draw')
        #pyed.exec_()

        app.exit()
        #pyed.exit()
        del app
        del pyed

        im3d[deletemask != 0] = cval
        #print ("Check output")
        # rewrite input data
        #pyed = QTSeedEditor(im3d)
        #pyed.exec_()

        #el pyed
        #import pdb; pdb.set_trace()

        return im3d

    def show_output(self):
        """
        Run viewer with output data3d and segmentation
        """

        from seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication
        import numpy as np
#, QMainWindow
        print ("Select voxels for deletion")
        app = QApplication(sys.argv)
        pyed = QTSeedEditor(self.data3d, contours=self.segmentation)
        pyed.exec_()

        #import pdb; pdb.set_trace()

        #pyed = QTSeedEditor(deletemask, mode='draw')
        #pyed.exec_()

        app.exit()


def readData3d(dcmdir):
    qt_app = None
    if dcmdir == None:
        from PyQt4 import QtGui
#QApplication
        qt_app = QtGui.QApplication(sys.argv)
        dcmdir = datareader.get_dcmdir_qt(qt_app)
    reader = datareader.DataReader()
    data3d, metadata = reader.Get3DData(dcmdir)

    return data3d, metadata, qt_app

def save_config(cfg, filename):
    cfg
    misc.obj_to_file(cfg, filename, filetype="yaml")


def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

## read confguraton from file, use default values from OrganSegmentation
    cfgplus = {
            'datapath':None,
            'viewermax':300,
            'viewermin':-100
            }

    cfg = config.get_default_function_config(OrganSegmentation.__init__)
    cfg.update(cfgplus)
    # now is in cfg default values

    cfg = config.get_config("organ_segmentation.config", cfg)


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
    parser.add_argument('-vs', '--working_voxelsize_mm', 
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

    parser.add_argument('-ol', '--output_label', default=1,
            help='label for segmented data')
    parser.add_argument('--slab', 
            default=cfg["slab"], 
            type=eval,
            help='labels for segmentation,\
                    example -slab "{\'liver\':1, \'lesions\':6}"')
    parser.add_argument('-acr', '--autocrop', 
            help='automatic crop after data processing',
            default=cfg["autocrop"])
    parser.add_argument('-iparams', '--iparams', 
            default=None, 
            help='filename of ipars file with stored interactivity')
    parser.add_argument('-sp', '--segparams', 
            default=cfg["segparams"], 
            help='params for segmentation,\
            example -sp "{\'pairwise_alpha_per_mm2\':90}"')
    parser.add_argument('-tx', '--texture_analysis', action='store_true',
            help='run with texture analysis')
    parser.add_argument('-exd', '--exampledata', action='store_true',
            help='run unittest')
    parser.add_argument('-ed', '--edit_data', action='store_true',
            help='Run data editor')
    parser.add_argument('-vmax', '--viewermax', type=eval, #type=int,
            help='Maximum of viewer window, set None for automatic maximum.',
            default=cfg["viewermax"])
            #default='1300')
    parser.add_argument('-vmin', '--viewermin',type=eval, #type=int,
            help='Minimum of viewer window, set None for automatic minimum.',
            #default='800')
            default=cfg["viewermin"])
    parser.add_argument('-so', '--show_output', action='store_true',
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




    # voxelsize_mm can be number or array
    #if args.voxelsize_mm != 'orig':
#TODO easy way to set half of resolution
#    if isinstance(args["working_voxelsize_mm"], numbers.Number)
#        if args["working_voxelsize_mm"] == -1:
#            args["working_voxelsize_mm"] = 'orig'
## todo and so on
#    if not args["working_voxelsize_mm"].startswith('orig'):
#        #import pdb; pdb.set_trace()
#        args["working_voxelsize_mm"] = eval(args["working_voxelsize_mm"])

    #  
    #args["segparams"] = eval(args["segparams"])
    #args["slab"] = eval(args["slab"])


    #args["viewermin"] = eval(args["viewermin"])
    #args["viewermax"] = eval(args["viewermax"])

    if args["debug"]:
        logger.setLevel(logging.DEBUG)


    if args["exampledata"]:

        args["datapath"] = '../sample_data/\
                matlab/examples/sample_data/DICOM/digest_article/'

    if args["iparams"] is not None:
        iparams = misc.obj_from_file(args["iparams"], filetype='pickle')
        t0 = time.time()
        oseg = OrganSegmentation(**iparams)
        
    else:
    #else:
    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
        #data3d, metadata = dcmreaddata.dcm_read_from_dir()
        data3d, metadata, qt_app = readData3d(args["datapath"])

        t0 = time.time()

        oseg_params = config.subdict(args, oseg_argspec_keys)
	oseg_params["data3d"] = data3d
	oseg_params["metadata"] = metadata
        oseg = OrganSegmentation(**oseg_params)
        



    oseg.interactivity(args["viewermin"], args["viewermax"])
    

    #igc = pycut.ImageGraphCut(data3d, zoom = 0.5)
    #igc.interactivity()

    #igc.make_gc()
    #igc.show_segmentation()

    # volume
    #volume_mm3 = np.sum(oseg.segmentation > 0) * np.prod(oseg.voxelsize_mm)

    t1 = time.time()
    audiosupport.beep()
    print (
            "Volume " +
            str(oseg.get_segmented_volume_size_mm3() / 1000000.0) + ' [l]')
    #pyed = py3DSeedEditor.py3DSeedEditor(oseg.data3d, contour =
    # oseg.segmentation)
    #pyed.show()
    print ("Total time: " + str (t1 - t0))

    if args["show_output"]:
        oseg.show_output()

    savestring = raw_input('Save output data? Yes/No/All with input data (y/n/a): ')
    #sn = int(snstring)
    if savestring in ['Y', 'y','a','A']:
# rename

        data = oseg.export()
        data['version'] = qmisc.getVersionString()
        iparams = oseg.get_iparams()
        #import pdb; pdb.set_trace()
        pth, filename = os.path.split(iparams['datapath'])
        if savestring in ['a','A']:
# save renamed file too
            misc.obj_to_file(data, 'organ_big-'+ filename + '.pklz', filetype='pklz')
        misc.obj_to_file(data, "organ.pklz", filetype='pklz')
        #misc.obj_to_file(data, "organ-"+ filename + ".pkl", filetype='pickle')
        misc.obj_to_file(iparams, 'organ_iparams.pkl', filetype='pickle')
        #misc.obj_to_file(iparams, 'iparams-'+ filename + '.pkl', filetype='pickle')
        data ['data3d'] = None
        misc.obj_to_file(data, 'organ_small-'+ filename + '.pklz', filetype='pklz')
    #output = segmentation.vesselSegmentation(oseg.data3d,
    # oseg.orig_segmentation)

if __name__ == "__main__":
    main()
