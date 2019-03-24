# /usr/bin/env python
# -*- coding: utf-8 -*-
"""

LISA - organ segmentation tool.

Liver Surgery Analyser

python organ_segmentation.py

python organ_segmentation.py -mroi -vs 0.6

Author: Miroslav Jirik
Email: miroslav.jirik@gmail.com


"""

import logging
logger = logging.getLogger(__name__)
import logging.handlers

import sys
import os
import os.path as op
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(path_to_script, "../../imcut/"))
# from collections import namedtuple

# from scipy.io import loadmat, savemat
import scipy
import scipy.ndimage
import numpy as np
import scipy.sparse
import datetime
import argparse
import copy
import json
from . import json_decoder as jd

from . import exceptionProcessing
from . import config_default

# tady uz je logger
# import dcmreaddata as dcmreader
# from imcut import pycut
# try:
#     import imcut  # noqa
#     from imcut import pycut
# except:
#     path_to_script = os.path.dirname(os.path.abspath(__file__))
#     sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
#     logger.warning("Deprecated of pyseg_base as submodule")
#     import traceback
#     traceback.print_exc()
#     import pycut


# from seg2fem import gen_mesh_from_voxels, gen_mesh_from_voxels_mc
# from viewer import QVTKViewer
from io3d import datareader
from io3d import datawriter
from io3d import misc
import io3d.cachefile as cachef
import io3d.misc
from . import data_plus
from . import support_structure_segmentation as sss
from . import config_default
from . import organ_seeds
from . import lisa_data
from . import data_manipulation
from . import qmisc
from . import config
from . import volumetry_evaluation
from . import segmentation_general
# import imtools.image_manipulation
import imma.image_manipulation as ima
import imma.labeled
import imma.segmentation_labels as imsl
from . import virtual_resection

# import audiosupport
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

config_version = [1, 0, 0]


def import_gui():
    # from lisaWindow import OrganSegmentationWindow
    # from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
    #     QGridLayout, QLabel, QPushButton, QFrame, \
    #     QFont, QPixmap
    # from PyQt4.Qt import QString

    pass


def printTotals(transferred, toBeTransferred):
    print("Transferred: {0}\tOut of: {1}".format(transferred, toBeTransferred))


class OrganSegmentation():
    """
    Main object of Lisa user interface.
    """

    def set_params(self, *args, **kwargs):
        """
        Function set parameters in same way as constructor does

        :param args:
        :param kwargs:
        :return:
        """
        self.__init__(*args, **kwargs)

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
            debug_mode=False,
            seg_postproc_pars={},
            cache_filename='cache.yml',
            seg_preproc_pars={},
            after_load_processing={},
            segmentation_alternative_params=None,
            sftp_username='lisa_default',
            sftp_password='',
            input_annotation_file=None,
            output_annotation_file=None,
            # run=False,
            run_organ_segmentation=False,
            run_vessel_segmentation=False,
            run_vessel_segmentation_params={},
            run_list = None,
            get_series_number_callback=None

            #           iparams=None,
    ):
        """ Segmentation of objects from CT data.

        :param datapath: path to directory with dicom files
        :param manualroi: manual set of ROI before data processing, there is a
             problem with correct coordinates
        :param data3d, metadata: it can be used for data loading not from
        directory. If both are setted, datapath is ignored
        :param output_label: label for output segmented volume
        :param slab: aditional label system for description segmented data
        {'none':0, 'liver':1, 'lesions':6}
        :param roi: region of interest.
        [[startx, stopx], [sty, spy], [stz, spz]]
        :param seeds: ndimage array with size same as data3d
        :param experiment_caption = this caption is used for naming of outputs
        :param lisa_operator_identifier: used for logging
        :param input_datapath_start: Path where user directory selection dialog
            starts.
        :param volume_blowup: Blow up volume is computed in smoothing so it is
        working only if smoothing is turned on.
        :param seg_postproc_pars: Can be used for setting postprocessing
        parameters. For example
        :param segmentation_alternative_params: dict of alternative params f,e.
        {'vs5: {'voxelsize_mm':[5,5,5]}, 'vs3: {'voxelsize_mm':[3,3,3]}}
        :param input_annotation_file: annotation input based on dwv json export (https://github.com/ivmartel/dwv)
        :param run_list: List of functions which should be run on run() function. Default list
        with segmentataion is used if is set None.
        """

        from imcut import pycut
        default_segparams = {
            'method': pycut.methods[0],
            'pairwise_alpha_per_mm2': 40,
            'use_boundary_penalties': False,
            'boundary_penalties_sigma': 50}

        self.iparams = {}
        self.datapath = datapath
        self.set_output_datapath(output_datapath)
        self.sftp_username = sftp_username
        self.sftp_password = sftp_password
        self.input_datapath_start = input_datapath_start
        self.crinfo = [[0, None], [0, None], [0, None]]
        self.slab = data_plus.default_slab()
        self.slab.update(slab)
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
        # self.version = qmisc.getVersionString()
        # if self.version is None:
        self.version = "1.18.1"
        self.viewermax = viewermax
        self.viewermin = viewermin
        self.volume_unit = volume_unit
        self.organ_interactivity_counter = 0
        self.dcmfilelist = None
        self.save_filetype = save_filetype
        self.vessel_tree = {}
        self.debug_mode = debug_mode
        self.gui_update = None
        self.segmentation_alternative_params = config_default.default_segmentation_alternative_params
        if segmentation_alternative_params is not None:
            self.segmentation_alternative_params.update(segmentation_alternative_params)

        self.saved_seeds = {}
        # self._json_description
        # SegPostprocPars = namedtuple(
        #     'SegPostprocPars', [
        #         'smoothing_mm',
        #         'segmentation_smoothing',
        #         'volume_blowup',
        #         'snakes',
        #         'snakes_method',
        #         'snakes_params']
        # )
        self.cache = cachef.CacheFile(cache_filename)

        self.seg_postproc_pars = {
            'smoothing_mm': smoothing_mm,
            'segmentation_smoothing': segmentation_smoothing,
            'volume_blowup': volume_blowup,
            'snakes': False,
            'snakes_method': 'ACWE',
            'snakes_params': {'smoothing': 1, 'lambda1': 100, 'lambda2': 1},
            'snakes_niter': 20,
            # 'postproc_working_voxelsize': [1.0, 1.0, 1.0],
            'postproc_working_voxelsize': 'orig',
        }
        self.seg_postproc_pars.update(seg_postproc_pars)
        self.seg_preproc_pars = {
            'use_automatic_segmentation': True,
        }
        self.seg_preproc_pars.update(seg_preproc_pars)
        self.after_load_processing = {
            'run_automatic_liver_seeds': False,
        }
        self.after_load_processing.update(after_load_processing)
        self.apriori = None
        # seg_postproc_pars.update(seg_postproc_pars)
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        # self.seg_postproc_pars = SegPostprocPars(**seg_postproc_pars_default)
        # self.run = run
        self.run_organ_segmentation = run_organ_segmentation
        self.run_vessel_segmentation = run_vessel_segmentation
        self.run_vessel_segmentation_params = run_vessel_segmentation_params
        #
        oseg_input_params = locals()
        oseg_input_params = self.__clean_oseg_input_params(oseg_input_params)

        logger.debug("oseg_input_params")
        logger.debug(str(oseg_input_params))
        self.oseg_input_params = oseg_input_params

        self.input_annotaion_file = input_annotation_file
        self.output_annotaion_file = output_annotation_file

        from . import runner
        self.runner = runner.Runner(self)
        self.init_run_list(run_list)
        self.get_series_number_callback = get_series_number_callback


        if data3d is None or metadata is None:
            # if 'datapath' in self.iparams:
            #     datapath = self.iparams['datapath']

            if datapath is not None:
                reader = datareader.DataReader()
                datap = reader.Get3DData(
                    datapath, dataplus_format=True,
                    get_series_number_callback=get_series_number_callback)
                # self.iparams['series_number'] = metadata['series_number']
                # self.iparams['datapath'] = datapath
                self.import_dataplus(datap)

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

    def set_output_datapath(self, output_datapath):
        if output_datapath is None:
            output_datapath = '~/lisa_data'
        self.output_datapath = os.path.expanduser(output_datapath)

    def update(self):
        from . import update_stable
        update_stable.make_update()
        # import subprocess
        # print subprocess.call(['conda', 'update', '-y', '-c', 'mjirik', '-c', 'SimpleITK', 'lisa']) #, shell=True)

    def add_to_segmentation(self, source_segmentation, target_labels, source_labels=None):
        """
        Stores requested label from temp segmentation into slab segmentation.
        Zero label is ignored.

        :param source_segmentation: ndimage
        :param target_labels: list of (string or numeric) labels for output segmentation. Labels are paired
        with source_labels if possible.
        :param source_labels: list of numeric labels for source segmentation
        :return:
        """
        if source_labels is None:
            source_labels = list(np.unique(source_segmentation))

        # kick zero
        if 0 in source_labels:
            source_labels.pop(source_labels.index(0))

        for labels in zip(source_labels, target_labels):
            src, dst = labels
            self.segmentation[source_segmentation==src] = self.nlabels(dst)

    def update_parameters_based_on_label(self, label):
        self.update_parameters(self.segmentation_alternative_params[label])

    def update_parameters(self, params):
        """

        :param params:
        :return:
        """
        if 'segparams' in params.keys():
            self.segparams = params['segparams']
            logger.debug('segparams updated')
        if 'segmodelparams' in params.keys():
            self.segmodelparams = params['segmodelparams']
            logger.debug('segmodelparams updated')
        if 'output_label' in params.keys():
            # logger.debug("output label " + str(params["output_label"]))
            # if type(params['output_label']) is str:
            #     key = params["output_label"]
            #     params["output_label"] = self.slab[key]
            self.output_label = self.nlabels(params['output_label'])
            logger.debug("'output_label' updated to " + str(self.nlabels(self.output_label, return_mode="str")))
        if 'working_voxelsize_mm' in params.keys():
            self.input_wvx_size = copy.copy(params['working_voxelsize_mm'])
            self.working_voxelsize_mm = params['working_voxelsize_mm']
            vx_size = self.working_voxelsize_mm

            if np.isscalar(vx_size):
                vx_size = ([vx_size] * 3)

            vx_size = np.array(vx_size).astype(float)

            self.working_voxelsize_mm = vx_size
            logger.debug('working_voxelsize_mm updated')
        if 'smoothing_mm' in params.keys():
            self.smoothing_mm = params['smoothing_mm']
            logger.debug('smoothing_mm updated')
        if 'seg_postproc_pars' in params.keys():
            self.seg_postproc_pars = params['seg_postproc_pars']
            logger.debug('seg_postproc_pars updated')
        if 'clean_seeds_after_update_parameters' in params.keys():

            if self.seeds is not None:
                self.seeds[...] = 0
            logger.debug('clean_seeds_after_update_parameters')

    def run_sss(self):
        sseg = sss.SupportStructureSegmentation(
            data3d=self.data3d,
            voxelsize_mm=self.voxelsize_mm,
        )
        sseg.run()

        # sseg.bone_segmentation()
        # sseg.lungs_segmentation()
        # sseg.heart_segmentation()

        # TODO remove hack - force remove number 1 from segmentation
        # this sould be fixed in sss
        sseg.segmentation[sseg.segmentation == 1] = 0
        self.segmentation = sseg.segmentation
        self.slab = sseg.slab

    def __clean_oseg_input_params(self, oseg_params):
        """
        Used for storing input params of organ segmentation. Big data are not
        stored due to big  memory usage.
        """
        oseg_params['data3d'] = None
        oseg_params['segmentation'] = None
        oseg_params.pop('self')
        oseg_params.pop('pycut')
        return oseg_params

    def process_wvx_size_mm(self, metadata):
        """This function does something.

            Args:
                name (str):  The name to use.

            Kwargs:
                state (bool): Current state to be in.

        """

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

    def load_data(self, datapath):
        self.datapath = datapath

        reader = datareader.DataReader()

        # seg.data3d, metadata =
        datap = reader.Get3DData(self.datapath, dataplus_format=True)
        # rint datap.keys()
        # self.iparams['series_number'] = self.metadata['series_number']
        # self.iparams['datapath'] = self.datapath
        self.import_dataplus(datap)

    def get_slab_value(self, label, value=None):
        value = data_plus.get_slab_value(self.slab, label, value)
        if self.gui_update is not None:
            self.gui_update()

        return value

    def sliver_compare_with_other_volume_from_file(self, filepath):
        reader = datareader.DataReader()
        segmentation_datap = reader.Get3DData(filepath, dataplus_format=True)
        evaluation = self.sliver_compare_with_other_volume(segmentation_datap)
        return evaluation

    def sliver_compare_with_other_volume(self, segmentation_datap):
        """
        Compares actual Lisa data with other which are given by
        segmentation_datap. That means
        segmentation_datap = {
            'segmentation': 3d np.array,
            'crinfo': information about crop (optional)
            }

        """
        # if there is no segmentation, data can be stored in data3d. It is the
        # way how are data stored in sliver.
        if 'segmentation' in segmentation_datap.keys():
            segm_key = 'segmentation'
        else:
            segm_key = 'data3d'
        if 'crinfo' in segmentation_datap.keys():
            data3d_segmentation = qmisc.uncrop(
                segmentation_datap[segm_key],
                segmentation_datap['crinfo'],
                self.orig_shape)
        else:
            data3d_segmentation = segmentation_datap[segm_key]
        pass

        # now we can uncrop actual Lisa data
        data3d_segmentation_actual = qmisc.uncrop(
            self.segmentation,
            self.crinfo,
            self.orig_shape)

        label1 = 1
        label2 = 1
        # TODO make GUI in Qt
        from PyQt4.QtCore import pyqtRemoveInputHook
        pyqtRemoveInputHook()
        print('unique data1 ', np.unique(data3d_segmentation_actual))
        print('unique data2 ', np.unique(data3d_segmentation))
        print("set label1 and label2")
        print("then press 'c' and 'Enter'")
        import ipdb;
        ipdb.set_trace()  # noqa BREAKPOINT

        evaluation = volumetry_evaluation.compare_volumes_sliver(
            data3d_segmentation_actual == label1,
            data3d_segmentation == label2,
            self.voxelsize_mm
        )
        # score = volumetry_evaluation.sliver_score_one_couple(evaluation)
        segdiff = qmisc.crop(
            ((data3d_segmentation) - data3d_segmentation_actual),
            self.crinfo)
        return evaluation, segdiff

    def segm_smoothing(self, sigma_mm, labels="liver", background_label="none"):
        """
        Shape of output segmentation is smoothed with gaussian filter.

        Sigma is computed in mm

        """
        segmentation_general.segmentation_smoothing(
            self.segmentation,
            sigma_mm,
            labels=labels,
            voxelsize_mm=self.voxelsize_mm,
            slab=self.slab,
            background_label=background_label,
            volume_blowup=self.volume_blowup,
        )
        # import scipy.ndimage

    def minimize_slab(self):
        imsl.minimize_slab(self.slab, self.segmentation)

    def select_label(self, labels):
        """

        :param labels:
        :return:
        """
        selected_segmentation = ima.select_labels(self.segmentation, labels=labels, slab=self.slab)
        return selected_segmentation

    def import_segmentation_from_file(self, filepath):
        """
        Loads data from file. Expected are uncropped data.
        """
        # logger.debug("import segmentation from file")
        # logger.debug(str(self.crinfo))
        reader = datareader.DataReader()
        datap = reader.Get3DData(filepath, dataplus_format=True)
        segmentation = datap['data3d']
        segmentation = qmisc.crop(segmentation, self.crinfo)
        logger.debug(str(segmentation.shape))
        self.segmentation = segmentation

    def import_dataplus(self, dataplus):
        datap = {
            'dcmfilelist': None,
        }
        datap.update(dataplus)

        dpkeys = datap.keys()
        self.data3d = datap['data3d']

        if self.roi is not None:
            self.crop(self.roi)

        self.voxelsize_mm = np.array(datap['voxelsize_mm'])
        self.process_wvx_size_mm(datap)
        self.autocrop_margin = self.autocrop_margin_mm / self.voxelsize_mm
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
        if 'vessel_tree' in dpkeys:
            self.vessel_tree = datap['vessel_tree']

        if ('apriori' in dpkeys) and datap['apriori'] is not None:
            self.apriori = datap['apriori']
        else:
            self.apriori = None

        if 'saved_seeds' in dpkeys:
            self.saved_seeds = datap['saved_seeds']
        else:
            self.saved_seeds = {}

        self.dcmfilelist = datap['dcmfilelist']

        self.segparams['pairwise_alpha'] = \
            self.segparams['pairwise_alpha_per_mm2'] / \
            np.mean(self.working_voxelsize_mm)

        self.__import_dataplus_seeds(datap)

        # chci, abych nepřepisoval uložené seedy
        if self.after_load_processing['run_automatic_liver_seeds']:
            if self.seeds is None or (self.seeds == 0).all():
                self.automatic_liver_seeds()

        # try read prev information about time processing
        try:
            time_prev = datap['processing_information']['processing_time']
            self.processing_time = time_prev
            self.time_start = datetime.datetime.now() - time_prev
        except:
            self.time_start = datetime.datetime.now()

    def __import_dataplus_seeds(self, datap):
        """

        :type self: seeds are changed
        """
        try:
            self.seeds = datap['processing_information'][
                'organ_segmentation']['seeds']
        except:
            logger.info('seeds not found in dataplus')
            # if dicomdir is readed after something with seeds, seeds needs to be reseted
            # self.seeds = None

        # for each mm on boundary there will be sum of penalty equal 10

        if self.seeds is None:
            logger.debug("Seeds are generated")
            self.seeds = np.zeros(self.data3d.shape, dtype=np.int8)
        logger.debug("unique seeds labels " + str(np.unique(self.seeds)))
        info_text = 'dir ' + str(self.datapath)
        if "series_number" in datap.keys():
            info_text += ", series_number " + str(datap['series_number'])
        info_text += 'voxelsize_mm ' + str(self.voxelsize_mm)
        logger.info(info_text)

    def crop(self, tmpcrinfo):
        """
        Function makes crop of 3d data and seeds and stores it in crinfo.

        tmpcrinfo: temporary crop information

        """
        # print('sedds ', str(self.seeds.shape), ' se ',
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

    def json_annotation_import(self, json_annotation_file=None):
        """

        :param json_annotation_file: json file from dwm (https://github.com/ivmartel/dwv)
        :return:
        """
        # TODO implementovat Jiří Vyskočil
        # načtení vstupní anotace
        # zápis do self.seeds
        # lisu pak lze volat:
        # python -m lisa -iaf dwv_export.json -dd input_data.pklz -o output_data.pklz -ni
        #
        # -ni dělá automatické spuštění segmentace
        # po načtení je spuštěn graph cut a výstup je uložen do output_data.pklz

        # takhle lze volat tuhle funkci s argumentem i bez něj
        if json_annotation_file is None:
            json_annotation_file = self.input_annotaion_file

        datap = {}
        datap['data3d'] = self.data3d
        datap['segmentation'] = self.segmentation
        datap['slab'] = self.slab
        datap['voxelsize_mm'] = self.voxelsize_mm
        jsonfile = json.load(open(json_annotation_file))

        jd.get_segdata(jsonfile, datap)
        if "porta" in jd.description.keys():
            th = jd.description["porta"]["threshold"]
            self.run_vessel_segmentation = True

            self.run_vessel_segmentation_params = dict(
                threshold=th,
                inner_vessel_label="porta",
                organ_label="liver",
                seeds=jd.get_vesselpoint_in_seeds(jsonfile, "porta", self.data3d.shape),
                interactivity=False)
        else:
            self.run_vessel_segmentation = False

        self.seeds = jd.get_seeds(datap, "liver")
        self.run_organ_segmentation = True

    def _interactivity_begin(self):
        from imcut import pycut
        logger.debug('_interactivity_begin()')
        # TODO make copy and work with it
        # TODO really make the copy and work with it
        if self.segmentation is None:
            self.segmentation = np.zeros_like(self.data3d, dtype=np.int8)

        data3d_tmp = self.data3d
        if self.seg_preproc_pars['use_automatic_segmentation']:
            data3d_tmp = self.data3d.copy()
            data3d_tmp[(self.segmentation > 0) & (self.segmentation != self.output_label)] = -1000

        # print 'zoom ', self.zoom
        # print 'svs_mm ', self.working_voxelsize_mm
        self.zoom = self.voxelsize_mm / (1.0 * self.working_voxelsize_mm)
        import warnings
        warnings.filterwarnings('ignore', '.*output shape of zoom.*')
        data3d_res = scipy.ndimage.zoom(
            self.data3d,
            self.zoom,
            mode='nearest',
            order=1
        ).astype(np.int16)

        logger.debug('pycut segparams ' + str(self.segparams) +
                     '\nmodelparams ' + str(self.segmodelparams)
                     )
        # insert feature function instead of string description
        from . import organ_model
        self.segmodelparams = organ_model.add_fv_extern_into_modelparams(self.segmodelparams)
        self.segparams['pairwise_alpha'] = \
            self.segparams['pairwise_alpha_per_mm2'] / \
            np.mean(self.working_voxelsize_mm)

        if 'method' not in self.segparams.keys() or \
                self.segparams['method'] in pycut.accepted_methods:
            from .audiosupport import beep
            igc = pycut.ImageGraphCut(
                # self.data3d,
                data3d_res,
                segparams=self.segparams,
                voxelsize=self.working_voxelsize_mm,
                modelparams=self.segmodelparams,
                volume_unit='ml',
                interactivity_loop_finish_fcn=beep,
                debug_images=False
            )
        # elif self.segparams['method'] == '':
        else:
            import liver_segmentation
            igc = liver_segmentation.LiverSegmentation(
                data3d_res,
                segparams=self.segparams,
                voxelsize=self.working_voxelsize_mm,
            )
        if self.apriori is not None:
            apriori_res = misc.resize_to_shape(
                # seeds_res = scipy.ndimage.zoom(
                self.apriori,
                data3d_res.shape,
            )
            igc.apriori = apriori_res

        # igc.modelparams = self.segmodelparams
        # @TODO uncomment this for kernel model
        #        igc.modelparams = {
        #            'type': 'kernel',
        #            'params': {}
        #        }
        # if self.iparams['seeds'] is not None:
        if self.seeds is not None:
            seeds_res = misc.resize_to_shape(
                # seeds_res = scipy.ndimage.zoom(
                self.seeds,
                data3d_res.shape,
                mode='nearest',
                order=0
            )
            seeds_res = seeds_res.astype(np.int8)
            igc.set_seeds(seeds_res)

        # tohle je tu pro to, aby bylo možné přidávat nově objevené segmentace k těm starým
        # jinak jsou stará data přepsána
        if self.segmentation is not None:
            self.segmentation_prev = copy.copy(self.segmentation)
        else:
            self.segmentation_prev = None

        return igc

    def sync_lisa_data(self, username, password, host="147.228.47.162", callback=printTotals):
        self.sftp_username = username
        self.create_lisa_data_dir_tree()

        import sftpsync
        import paramiko

        paramiko_log = os.path.join(self.output_datapath, 'paramiko.log')
        paramiko.util.log_to_file(paramiko_log)
        sftp = sftpsync.Sftp(host=host, username=username, password=password)
        localfrom = self._output_datapath_from_server.replace(os.sep, '/')
        localto = self._output_datapath_to_server.replace(os.sep, '/')
        # this makes sure that all paths ends with slash
        if not localfrom.endswith('/'):
            localfrom += '/'
        if not localto.endswith('/'):
            localto += '/'
        remotefrom = "from_server/"
        remoteto = "to_server/"

        exclude = []

        logger.info("Download started\nremote from {}\nlocal  from {}".format(remotefrom, localfrom))
        logger.info("from")
        sftp.sync(remotefrom, localfrom, download=True, exclude=exclude, delete=False, callback=callback)
        logger.info("Download finished")
        logger.info("Upload started\nremote to {}\nlocal  to {}".format(remoteto, localto))
        sftp.sync(localto, remoteto, download=False, exclude=exclude, delete=False, callback=callback)
        logger.info("Upload finished")

    def __resize_to_orig(self, igc_seeds):
        # @TODO remove old code in except part
        self.segmentation = misc.resize_to_shape(
            self.segmentation,
            self.data3d.shape,
            self.zoom
        )
        self.seeds = misc.resize_to_shape(
            igc_seeds,
            self.data3d.shape,
            self.zoom
        ).astype(np.uint8)

    #         try:
    #             # rint 'pred vyjimkou'
    #             # aise Exception ('test without skimage')
    #             # rint 'za vyjimkou'
    #             import skimage
    #             import skimage.transform
    # # Now we need reshape  seeds and segmentation to original size
    #
    #             segm_orig_scale = skimage.transform.resize(
    #                 self.segmentation, self.data3d.shape, order=0,
    #                 preserve_range=True
    #             )
    #
    #             seeds = skimage.transform.resize(
    #                 igc_seeds, self.data3d.shape, order=0,
    #                 preserve_range=True
    #             )
    #
    #             # self.segmentation = segm_orig_scale
    #             self.seeds = seeds
    #             logger.debug('resize to orig with skimage')
    #         except:
    #
    #             segm_orig_scale = scipy.ndimage.zoom(
    #                 self.segmentation,
    #                 1.0 / self.zoom,
    #                 mode='nearest',
    #                 order=0
    #             ).astype(np.int8)
    #             seeds = scipy.ndimage.zoom(
    #                 igc_seeds,
    #                 1.0 / self.zoom,
    #                 mode='nearest',
    #                 order=0
    #             )
    #             logger.debug('resize to orig with scipy.ndimage')
    #
    # # @TODO odstranit hack pro oříznutí na stejnou velikost
    # # v podstatě je to vyřešeno, ale nechalo by se to dělat elegantněji v zoom
    # # tam je bohužel patrně bug
    #             # rint 'd3d ', self.data3d.shape
    #             # rint 's orig scale shape ', segm_orig_scale.shape
    #             shp = [
    #                 np.min([segm_orig_scale.shape[0], self.data3d.shape[0]]),
    #                 np.min([segm_orig_scale.shape[1], self.data3d.shape[1]]),
    #                 np.min([segm_orig_scale.shape[2], self.data3d.shape[2]]),
    #             ]
    #             # elf.data3d = self.data3d[0:shp[0], 0:shp[1], 0:shp[2]]
    #             # mport ipdb; ipdb.set_trace() # BREAKPOINT
    #
    #             self.segmentation = np.zeros(self.data3d.shape, dtype=np.int8)
    #             self.segmentation[
    #                 0:shp[0],
    #                 0:shp[1],
    #                 0:shp[2]] = segm_orig_scale[0:shp[0], 0:shp[1], 0:shp[2]]
    #
    #             del segm_orig_scale
    #
    #             self.seeds[
    #                 0:shp[0],
    #                 0:shp[1],
    #                 0:shp[2]] = seeds[0:shp[0], 0:shp[1], 0:shp[2]]

    def _interactivity_end(self, igc):
        """
        This is called after processing step. All data are rescaled to original
        resolution.
        """
        logger.debug('_interactivity_end()')

        self.__resize_to_orig(igc.seeds)
        self.organ_interactivity_counter = igc.interactivity_counter
        logger.debug("org inter counter " +
                     str(self.organ_interactivity_counter))
        logger.debug('nonzero segm ' + str(np.nonzero(self.segmentation)))
        # if False:
        if False:
            # TODO dodělat postprocessing PV
            import segmentation
            outputSegmentation = segmentation.vesselSegmentation(  # noqa
                self.data3d,
                self.segmentation,
                threshold=-1,
                inputSigma=0.15,
                dilationIterations=10,
                nObj=1,
                biggestObjects=False,
                seeds=(self.segmentation > 0).astype(np.int8),
                useSeedsOfCompactObjects=True,
                interactivity=True,
                binaryClosingIterations=2,
                binaryOpeningIterations=0)

        self._segmentation_postprocessing()
        # @TODO make faster
        # spojení staré a nové segmentace
        # from PyQt4.QtCore import pyqtRemoveInputHook; pyqtRemoveInputHook()
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        if self.segmentation_prev is None:
            # pokud neznáme žádnou předchozí segmentaci, tak se chováme jako dříve
            self.segmentation[self.segmentation == 1] = self.nlabels(self.output_label)
        else:
            # remove old pixels for this label
            self.segmentation_replacement(
                segmentation_new=self.segmentation,
                segmentation=self.segmentation_prev,
                label=self.output_label,
                label_new=1,
            )
            # self.segmentation_prev[self.segmentation_prev == self.output_label] = 0
            # set new labels
            # self.segmentation_prev[np.where(self.segmentation == 1)] = self.output_label

            # clean up

            self.segmentation = self.segmentation_prev
            self.segmentation_prev = None

        # rint 'autocrop', self.autocrop
        if self.autocrop is True:
            # rint
            # mport pdb; pdb.set_trace()

            tmpcrinfo = qmisc.crinfo_from_specific_data(
                self.segmentation,
                self.autocrop_margin)

            self.crop(tmpcrinfo)

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
        #
        logger.debug('self.slab')
        logger.debug(str(self.slab))
        self.processing_time = (
                datetime.datetime.now() - self.time_start).total_seconds()

        logger.debug('processing_time = ' + str(self.processing_time))

    def segmentation_replacement(
            self,
            segmentation_new,
            label,
            label_new=1,
            segmentation=None,
            **kwargs
    ):
        if segmentation is None:
            segmentation = self.segmentation

        segmentation_general.segmentation_replacement(
            segmentation,
            segmentation_new,
            label_new=label_new,
            label=label,
            slab=self.slab,
            **kwargs
        )

    def _segmentation_postprocessing(self):
        """
        :segmentation_smoothing:
        """
        logger.debug(str(self.seg_postproc_pars))
        if self.seg_postproc_pars['segmentation_smoothing']:
            # if self.segmentation_smoothing:
            self.segm_smoothing(self.seg_postproc_pars['smoothing_mm'])

        if self.seg_postproc_pars['snakes']:
            import morphsnakes as ms
            logger.debug('Making snakes')
            if self.seg_postproc_pars['snakes_method'] is 'ACWE':
                method = ms.MorphACWE
            elif self.seg_postproc_pars['snakes_method'] is 'GAC':
                method = ms.MorphGAC
            else:
                logger.error('Unknown snake method')
                return

            sp = self.seg_postproc_pars['snakes_params']
            if 'seeds' in sp.keys() and sp['seeds'] is True:
                sp['seeds'] = self.seeds

            logger.debug('snakes')
            d3d = io3d.misc.resize_to_mm(
                self.data3d,
                self.voxelsize_mm,
                self.seg_postproc_pars['postproc_working_voxelsize'])
            segw = io3d.misc.resize_to_mm(
                self.segmentation,
                self.voxelsize_mm,
                self.seg_postproc_pars['postproc_working_voxelsize'])
            macwe = method(
                d3d,
                # self.data3d,
                **self.seg_postproc_pars['snakes_params']
            )
            macwe.levelset = (
                # self.segmentation == self.slab['liver']
                    segw == self.slab['liver']
            ).astype(np.uint8)
            macwe.run(self.seg_postproc_pars['snakes_niter'])
            seg = io3d.misc.resize_to_shape(macwe.levelset, self.data3d.shape)
            # for debug visualization preprocessing use fallowing line
            # self.segmentation[seg == 1] += 1
            self.segmentation[seg == 1] = self.slab['liver']
            logger.debug('postprocessing with snakes finished')

    #    def interactivity(self, min_val=800, max_val=1300):
    # @TODO generovat QApplication
    def interactivity(self, min_val=None, max_val=None, layout=None):
        from imcut.seed_editor_qt import QTSeedEditor
        import_gui()
        logger.debug('interactivity')
        # if self.edit_data:
        #     self.data3d = self.data_editor(self.data3d)

        igc = self._interactivity_begin()
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        if layout is None:
            pyed = QTSeedEditor(igc.img,
                                seeds=igc.seeds,
                                modeFun=igc.interactivity_loop,
                                voxelSize=igc.voxelsize,
                                volume_unit='ml')
        else:
            from imcut import QTSeedEditorWidget
            pyed = QTSeedEditorWidget(igc.img,
                                      seeds=igc.seeds,
                                      modeFun=igc.interactivity_loop,
                                      voxelSize=igc.voxelsize,
                                      volume_unit='ml')
            layout.addWidget(pyed)

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
        from PyQt4 import QtCore
        QtCore.pyqtRemoveInputHook()
        # import ipdb; ipdb.set_trace()
        # @TODO někde v igc.interactivity() dochází k přehození nul za jedničy,
        # tady se to řeší hackem
        if igc.segmentation is not None:
            self.segmentation = (igc.segmentation == 0).astype(np.int8)
        self._interactivity_end(igc)

    def ninteractivity(self):
        from imcut import pycut
        """Function for automatic (noninteractiv) mode."""
        # mport pdb; pdb.set_trace()
        igc = self._interactivity_begin()
        # gc.interactivity()
        # igc.make_gc()
        igc.run()
        if ('method' not in self.segparams.keys()) or (self.segparams['method'] in pycut.methods):
            logger.debug('ninteractivity seg method GC')
            self.segmentation = (igc.segmentation == 0).astype(np.int8)
        else:
            logger.debug('ninteractivity seg method other')
            self.segmentation = np.asarray(igc.segmentation, dtype=np.int8)
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
        data['apriori'] = self.apriori
        data['slab'] = slab
        data['voxelsize_mm'] = self.voxelsize_mm
        data['orig_shape'] = self.orig_shape
        data['vessel_tree'] = self.vessel_tree
        data["saved_seeds"] = self.saved_seeds
        processing_information = {
            'organ_segmentation': {
                'processing_time': self.processing_time,
                'time_start': str(self.time_start),
                'oseg_input_params': self.oseg_input_params,
                'organ_interactivity_counter':
                    self.organ_interactivity_counter,
                'seeds': self.seeds  # qmisc.SparseMatrix(self.seeds)
            }
        }
        data['processing_information'] = processing_information
        # from PyQt4 import QtCore
        # QtCore.pyqtRemoveInputHook()
        # import ipdb; ipdb.set_trace()
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

    def automatic_liver_seeds(self):
        seeds, likdif = organ_seeds.automatic_liver_seeds(self.data3d, self.seeds, self.voxelsize_mm)
        # přenastavíme na čísla mezi nulou a jedničkou, druhá konstanta je nastavena empiricky
        self.apriori = boltzman(likdif, 0, 200).astype(np.float16)

    def add_seeds_mm(self, z_mm, x_mm, y_mm, label, radius, width=1):
        """
        Function add circle seeds to one slice with defined radius.

        It is possible set more seeds on one slice with one dimension

        x_mm, y_mm coordinates of circle in mm. It may be array.
        z_mm = slice coordinates  in mm. It may be array
        :param label: one number. 1 is object seed, 2 is background seed
        :param radius: is radius of circle in mm
        :param width: makes circle with defined width (repeat circle every milimeter)

        """

        data_manipulation.add_seeds_mm(
            self.seeds, self.voxelsize_mm,
            z_mm, x_mm, y_mm,
            label,
            radius, width
        )

    def lesionsLocalization(self):
        """ Localization of lession """
        from . import lesions
        tumory = lesions.Lesions()
        # tumory.overlay_test()
        data = self.export()
        tumory.import_data(data)
        tumory.run_gui()
        # tumory.automatic_localization()

        self.segmentation = tumory.segmentation

    def nlabels(self, label, label_meta=None, return_mode="num"):

        """
        Add one or more labels if it is necessery and return its numeric values.

        If "new" keyword is used and no other information is provided, the max + 1 label is created.
        If "new" keyword is used and additional numeric info is provided, the number is used also as a key.
        :param label: string, number or "new"
        :param label_meta: string, number or "new
        :param return_mode: "num" or "str" or "both".
        :return:
        """

        return ima.get_nlabels(self.slab, label, label_meta, return_mode=return_mode)

    def add_missing_labels(self):
        ima.add_missing_labels(self.segmentation, self.slab)

    def segmentation_relabel(self, from_label, to_label):
        """
        Relabel segmentation
        :param from_label: int or string
        :param to_label: int or `astring
        :return:
        """
        from_label = self.nlabels(from_label)
        to_label = self.nlabels(to_label)
        select = self.select_label(from_label)
        self.segmentation[select] = to_label

    def portalVeinSegmentation(self, inner_vessel_label="porta", organ_label="liver", outer_vessel_label=None,
                               forbidden_label=None, threshold=None, interactivity=True, seeds=None, **inparams):
        """
        Segmentation of vein in specified volume. It is given by label "liver".
        Usualy it is number 1. If there is no specified volume all image is
        used.

        If this function is used repeatedly (if there is some segmentation in
        this image) all segmentation labeled as 'porta' is removed and setted
        to 'liver' before processing.

        You can use additional parameters from vesselSegmentation()
        For example interactivity=False, biggestObjects=True, ...
        :param forbidden_label: int or list of ints. Labels not included into segmentable area.
        """

        from imtools import segmentation as imsegmentation
        logger.info('segmentation max label ' + str(np.max(self.segmentation)))

        if outer_vessel_label is None:
            outer_vessel_label = inner_vessel_label
        # if there is no organ segmentation, use all image
        # self.add_slab_label_carefully(numeric_label=numeric_label, string_label=string_label)

        # if there is no liver segmentation, use whole image
        # if np.max(self.segmentation) == 0:
        #     self.segmentation = self.segmentation + 1

        # remove prev segmentation
        # TODO rozdělit na vnitřní a vnější část portální žíly

        params = {
            'threshold': threshold,
            'inputSigma': 0.15,
            'aoi_dilation_iterations': 10,
            'nObj': 1,
            'biggestObjects': False,
            'useSeedsOfCompactObjects': True,
            'interactivity': interactivity,
            'binaryClosingIterations': 2,
            'binaryOpeningIterations': 0,
            'seeds': seeds,
        }
        params.update(inparams)
        # logger.debug("ogran_label ", organ_label)
        # target_segmentation = (self.segmentation == self.nlabels(organ_label)).astype(np.int8)
        target_segmentation = ima.select_labels(
            self.segmentation, organ_label, self.slab
        )
        outputSegmentation = imsegmentation.vesselSegmentation(
            self.data3d,
            voxelsize_mm=self.voxelsize_mm,
            # target_segmentation,
            segmentation=self.segmentation,
            # organ_label=organ_label,
            aoi_label=organ_label,
            forbidden_label=forbidden_label,
            slab=self.slab,
            debug=self.debug_mode,
            **params
        )

        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import ipdb; ipdb.set_trace()
        self.segmentation[(outputSegmentation == 1) & (target_segmentation == 1)] = self.nlabels(inner_vessel_label)
        self.segmentation[(outputSegmentation == 1) & (target_segmentation == 0)] = self.nlabels(outer_vessel_label)

        # self.__vesselTree(outputSegmentation, 'porta')

    def saveVesselTree(self, textLabel, fn_yaml=None, fn_vtk=None):
        """
        textLabel: 'porta' or 'hepatic_veins'
        """
        self.__vesselTree(
            self.segmentation == self.slab[textLabel],
            textLabel,
            fn_yaml=fn_yaml,
            fn_vtk=fn_vtk,
        )

    def export_seeds_to_files(self, fn_seeds):
        """
        Export actual seeds and saved seeds into file based on given file name. Data are stored as image data (data3d).
        :param fn_seeds:
        :return:
        """
        datap = self.export()
        if "saved_seeds" in datap:
            saved_seeds = datap.pop("saved_seeds")
            for key in saved_seeds:
                datap = self.export()
                if "saved_seeds" in datap:
                    datap.pop("saved_seeds")
                if "seeds" in datap:
                    datap.pop("seeds")
                if "segmentation" in datap:
                    datap.pop("segmentation")
                if "processing_information" in datap:
                    datap.pop('processing_information')
                seeds = saved_seeds[key]
                datap["data3d"] = seeds
                basefn, ext = op.splitext(fn_seeds)
                fn_seeds_key = basefn + "_" + key + ext
                io3d.write(datap, fn_seeds_key)
        if "seeds" in datap:
            seeds = datap.pop("seeds")
            if "segmentation" in datap:
                datap.pop("segmentation")
            if "processing_information" in datap:
                datap.pop('processing_information')
            datap["data3d"] = seeds
            io3d.write(datap, fn_seeds)

    def import_seeds_from_file(self, fn_seeds):
        datap = io3d.read(fn_seeds, dataplus_format=True)
        if "seeds" in datap and datap["seeds"] is not None:
            self.seeds = datap["seeds"]
        else:
            self.seeds = datap["data3d"]

    def __vesselTree(self, binaryData3d, textLabel, fn_yaml=None, fn_vtk=None):
        import skelet3d
        from skelet3d import skeleton_analyser  # histology_analyser as skan
        data3d_thr = (binaryData3d > 0).astype(np.int8)
        data3d_skel = skelet3d.skelet3d(data3d_thr)

        skan = skeleton_analyser.SkeletonAnalyser(
            data3d_skel,
            volume_data=data3d_thr,
            voxelsize_mm=self.voxelsize_mm)
        stats = skan.skeleton_analysis(guiUpdateFunction=None)

        if 'Graph' not in self.vessel_tree.keys():
            self.vessel_tree['voxelsize_mm'] = self.voxelsize_mm
            self.vessel_tree['Graph'] = {}

        self.vessel_tree['Graph'][textLabel] = stats
        # print sa.stats
        logger.debug('save vessel tree to file')
        if fn_yaml is None:
            fn_yaml = self.get_standard_ouptut_filename(filetype='yaml', suffix='-vt-' + textLabel)
        # save all skeletons to one special file
        misc.obj_to_file(self.vessel_tree, fn_yaml, filetype='yaml')
        logger.debug('save vessel tree to file - finished')
        # generate vtk file
        logger.debug('start to generate vtk file from vessel_tree')
        import fibrous.tb_vtk
        # import imtools.gen_vtk_tree
        if fn_vtk is None:
            fn_vtk = self.get_standard_ouptut_filename(filetype='vtk', suffix='-vt-' + textLabel)
        # imtools.gen_vtk_tree.vt2vtk_file(self.vessel_tree, fn_vtk, text_label=textLabel)
        fibrous.tb_vtk.vt2vtk_file(self.vessel_tree, fn_vtk, text_label=textLabel)
        logger.debug('generating vtk file from vessel_tree finished')

    def hepaticVeinsSegmentation(self):

        from imtools import segmentation
        outputSegmentation = segmentation.vesselSegmentation(
            self.data3d,
            self.segmentation,
            threshold=None,
            inputSigma=0.15,
            dilationIterations=10,
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
    #         self.__vesselTree(outputSegmentation, 'hepatic_veins')

    def get_segmented_volume_size_mm3(self, labels="liver"):
        """Compute segmented volume in mm3, based on subsampeled data."""

        voxelvolume_mm3 = np.prod(self.voxelsize_mm)
        volume_mm3 = np.sum(ima.select_labels(self.segmentation, labels, self.slab)) * voxelvolume_mm3
        return volume_mm3

    def get_standard_ouptut_filename(self, filetype=None, suffix=''):
        """
        It can be settet filename, or filename end with suffix.
        """
        if filetype is None:
            filetype = self.save_filetype

        output_dir = self.output_datapath

        if self.datapath is not None:
            pth, filename = op.split(op.normpath(self.datapath))
            filename, ext = os.path.splitext(filename)
        else:
            filename = ''
        if len(filename) > 0 and len(self.experiment_caption) > 0:
            filename += "-"
        filename += self.experiment_caption
        #        if savestring in ['a', 'A']:
        # save renamed file too
        filename = '' + filename + suffix + '.' + filetype
        filepath = op.join(output_dir, filename)
        filepath = misc.suggest_filename(filepath)

        return filepath

    def save_outputs(self, filepath=None):
        """ Save input data, segmentation and all other metadata to file.

        :param filepath:
        :return:
        """

        data = self.export()
        data['version'] = self.version  # qmisc.getVersionString()
        data['experiment_caption'] = self.experiment_caption
        data['lisa_operator_identifier'] = self.lisa_operator_identifier
        self.create_lisa_data_dir_tree()

        if filepath is None:
            filepath = self.get_standard_ouptut_filename()
        # import ipdb; ipdb.set_trace()
        import io3d
        logger.debug("save outputs to file %s" % (filepath))
        io3d.write(data, filepath)

        if self.output_annotaion_file is not None:
            self.json_annotation_export()

    def json_annotation_export(self):
        """

        :return:
        """
        # TODO Jiri Vyskocil
        output_file = self.output_annotaion_file
        # self.segmentation
        data = {}
        data['segmentation'] = self.segmentation
        data['slab'] = self.slab

        jd.write_to_json(data, output_name=output_file)

    def create_lisa_data_dir_tree(self):
        lisa_data.create_lisa_data_dir_tree(self)

    def save_seeds(self, name):
        """
        Load stored seeds
        :param name:
        :return:
        """
        seeds = copy.copy(self.seeds)
        # self.saved_seeds[name] = scipy.sparse.csr_matrix(seeds)
        self.saved_seeds[name] = seeds

    def load_seeds(self, name):
        """
        Store seeds for later use.
        :param name:
        :return:
        """
        seeds = self.saved_seeds[name]
        # if scipy.sparse.issparse(seeds):
        #     seeds = seeds.todense()
        self.seeds = seeds

    def get_list_of_saved_seeds(self):
        return list(self.saved_seeds.keys())

    def split_tissue_recusively_with_labeled_volumetric_vessel_tree(
            self, organ_label, seeds,
            organ_split_label_format_pattern="{label}{i}"
    ):
        """

        :param organ_split_label_format_pattern: Specify the patter for naming the split of tissue
        :param organ_label: label of organ to split
        :param seeds: ndarray, 1 is trunk, 2 is first level branches, 3 is second level branches ...
        :return:
        """
        un_labels_dict = imma.labeled.unique_labels_by_seeds(self.segmentation, seeds)
        # ještě mi chybí vědět, kdo je potomkem koho
        # (level, tissue_to_split, trunk, branches
        split_parameters = {1: []}
        to_process = [(1, organ_label, un_labels_dict[1][0], un_labels_dict[2])]
        while len(to_process) > 0:
            actual = to_process.pop(0)
            actual_level = actual[0]
            actual_organ_label = actual[1]
            actual_trunk_label = actual[2]
            actual_branch_labels = actual[3]
            split_labels_ij, connected_ij = self.split_tissue_with_labeled_volumetric_vessel_tree(
                organ_label=actual_organ_label,
                trunk_label=actual_trunk_label,
                branch_labels=actual_branch_labels,
                organ_split_label_format_pattern=organ_split_label_format_pattern
            )

            # prepare next branche
            # level of next trunk
            next_level = actual_level + 1
            next_level_of_branches = next_level + 1
            if next_level_of_branches <= len(un_labels_dict):
                for i in range(len(split_labels_ij)):
                    import imma.dili as imdl
                    next_trunk = actual_branch_labels[i]
                    ind = imdl.find_in_list_of_lists(connected_ij, next_trunk)
                    if ind is None:
                        logger.error("There is strange error. This should be impossible.")
                    next_organ_label = split_labels_ij[ind]
                    next_branches = list(set(connected_ij[ind]).intersection(set(un_labels_dict[next_level_of_branches])))
                    if len(next_branches) > 1:
                        next = (next_level, next_organ_label, next_trunk, next_branches)
                        to_process.append(next)

        return None, None


    def split_tissue_with_labeled_volumetric_vessel_tree(
            self, organ_label, trunk_label, branch_labels, split_labels=None,
            organ_split_label_format_pattern="{label}{i}", on_missed_branch="split"):
        """

        :param organ_label:
        :param trunk_label:
        :param branch_label1:
        :param branch_label2:
        :param seeds:
        :param split_label1:
        :param split_label2:
        :return:
        """
        # try:
        #     if trunk_label is None:
        #         trunk_label = self.segmentation[seeds == 1][0]
        #     if branch_labels is None:
        #         branch
        #         branch_label1 = self.segmentation[seeds == 2][0]
        #     if branch_label2 is None:
        #         branch_label2 = self.segmentation[seeds == 3][0]
        # except IndexError:
        #     ValueError("Trunk and branches labels should be defined or seeds with values 1,2,3 are expected.")


        trunk_label = self.nlabels(trunk_label)
        branch_labels = self.nlabels(branch_labels)
        split, connected = virtual_resection.split_tissue_on_labeled_tree(
            self.segmentation,
            trunk_label,
            branch_labels=branch_labels,
            tissue_segmentation=self.select_label(organ_label),
            ignore_labels=[self.nlabels(organ_label)],
            on_missed_branch=on_missed_branch
        )

        if split_labels is None:
            split_labels = [None] * len(branch_labels)
            for i in range(len(branch_labels)):
                split_labels[i] = organ_split_label_format_pattern.format(
                    label=self.nlabels(organ_label, return_mode="str"),
                    i=(i + 1)
                )



        # if split_label1 is None:
        #     split_label1 = self.nlabels(organ_label, return_mode="str") + "1"
        #     # split_label1 = self.nlabels(split_label1)
        # if split_label2 is None:
        #     split_label2 = self.nlabels(organ_label, return_mode="str") + "2"
        #     # split_label2 = self.nlabels(split_label2)
        for i in range(len(split_labels)):
            self.segmentation[split == (i + 1)] = self.nlabels(split_labels[i])
        # self.segmentation[split == 1] = self.nlabels(split_label1)
        # self.segmentation[split == 2] = self.nlabels(split_label2)

        return split_labels, connected

    # old version
    # def rotate(self, angle, axes):
    #     self.data3d = scipy.ndimage.interpolation.rotate(self.data3d, angle, axes)
    #     self.segmentation = scipy.ndimage.interpolation.rotate(self.segmentation, angle, axes)
    #     self.seeds = scipy.ndimage.interpolation.rotate(self.seeds, angle, axes)
    def resize_to_mm(self, voxelsize_mm):
        """
        Resize voxelsize to defined milimeters.

        :param voxelsize_mm:
        :return:
        """
        if np.isscalar(voxelsize_mm):
            voxelsize_mm = self.data3d.ndim * [voxelsize_mm]

        orig_voxelsize_mm = self.voxelsize_mm
        orig_shape = self.data3d.shape

        self.data3d = io3d.misc.resize_to_mm(self.data3d, voxelsize_mm=orig_voxelsize_mm, new_voxelsize_mm=voxelsize_mm)
        if self.segmentation is not None:
            dtype = self.segmentation.dtype
            self.segmentation = io3d.misc.resize_to_mm(
                self.segmentation, voxelsize_mm=orig_voxelsize_mm, new_voxelsize_mm=voxelsize_mm).astype(dtype)

        if not hasattr(self, "orig_voxelsize_mm"):
            # It this is first resize
            self.orig_voxelsize_mm = orig_voxelsize_mm
            self.orig_shape = orig_shape

        self.voxelsize_mm = voxelsize_mm

    def rotate(self, phi_deg, theta_deg=None, phi_axes=(1, 2), theta_axes=(0, 1), **kwargs):
        self.data3d = ima.rotate(self.data3d, phi_deg, theta_deg)
        self.segmentation = ima.rotate(self.segmentation, phi_deg, theta_deg)
        self.seeds = ima.rotate(self.seeds, phi_deg, theta_deg)

    def random_rotate(self):
        """
        Rotate data3d, segmentation and seeds with random rotation
        :return:
        """
        # TODO independent on voxlelsize (2016-techtest-rotate3d.ipynb)
        phi_deg, theta_deg = ima.random_rotate_paramteres()
        self.rotate(phi_deg, theta_deg)
        # old version
        # xi1 = np.random.rand()
        # xi2 = np.random.rand()
        #
        # # theta = np.arccos(np.sqrt(1.0-xi1))
        # theta = np.arccos(1.0 - (xi1 * 1))
        # phi = xi2 * 2 * np.pi
        #
        # # xs = np.sin(theta) * np.cos(phi)
        # # ys = np.sin(theta) * np.sin(phi)
        # # zs = np.cos(theta)
        #
        # phi_deg = np.degrees(phi)
        # self.rotate(phi_deg, (1, 2))
        # theta_deg = np.degrees(theta)
        # self.rotate(theta_deg, (0, 1))

    def mirror_z_axis(self):
        """
        mirror data3d, segmentation and seeds Z-zaxis
        :return:
        """
        self.data3d = self.data3d[-1::-1]
        if self.segmentation is not None:
            self.segmentation = self.segmentation[-1::-1]
        if self.seeds is not None:
            self.seeds = self.seeds[-1::-1]

    def save_input_dcm(self, filename):
        # TODO add
        logger.debug('save dcm')
        dw = datawriter.DataWriter()
        dw.Write3DData(self.data3d, filename, filetype='dcm',
                       metadata={'voxelsize_mm': self.voxelsize_mm})

    def save_outputs_dcm(self, filename):
        # TODO add
        logger.debug('save dcm')
        dw = datawriter.DataWriter()
        dw.Write3DData(self.segmentation.astype(np.int16), filename,
                       filetype='dcm',
                       metadata={'voxelsize_mm': self.voxelsize_mm})

    def save_outputs_dcm_overlay(self):
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
        return output_dicom_dir

    def load_segmentation_from_dicom_overlay(self, dirpath=None):
        """
        Get overlay from dicom file stack
        :param dirpath:
        :return:
        """
        if dirpath is None:
            dirpath = self.datapath
        reader = datareader.DataReader()
        data3d, metadata = reader.Get3DData(dirpath, qt_app=None, dataplus_format=False)
        overlays = reader.get_overlay()
        overlay = np.zeros(data3d.shape, dtype=np.int8)
        # print("overlays ", list(overlays.keys()))
        for key in overlays:
            overlay += overlays[key]
        if not np.allclose(self.data3d.shape, overlay.shape):
            logger.warning("Overlay shape does not fit the data shape")
        self.segmentation = overlay
        return dirpath

    def fill_holes_in_segmentation(self, label=None, background_label=0):
        """
        Fill holes in segmentation.

        Label could be set interactivelly.

        :param label: if none, the self.output_label is used
        :return:
        """
        if label is None:
            label = self.output_label
        segm_to_fill = self.segmentation == self.nlabels(label)
        # self.segmentation[segm_to_fill] = background_label
        segm_to_fill = scipy.ndimage.morphology.binary_fill_holes(segm_to_fill)
        self.segmentation[segm_to_fill] = self.nlabels(label)

        # segm = imtools.show_segmentation.select_labels(segmentation=self.segmentation, labels=labels)
        # self.

    def get_body_navigation_structures(self):
        import bodynavigation
        self.bodynavigation = bodynavigation.BodyNavigation(self.data3d, self.voxelsize_mm)

        bn = self.bodynavigation
        bn.use_new_get_lungs_setup = True
        self.segmentation[bn.get_lungs() > 0] = self.nlabels("lungs")
        self.segmentation[bn.get_spine() > 0] = self.nlabels("spine")
        # self.segmentation[bn.get_chest() > 0] = self.nlabels("chest")

    def get_body_navigation_structures_precise(self):
        import bodynavigation.organ_detection
        self.bodynavigation = bodynavigation.organ_detection.OrganDetection(self.data3d, self.voxelsize_mm)
        bn = self.bodynavigation
        self.segmentation[bn.getLungs() > 0] = self.nlabels("lungs")
        self.segmentation[bn.getBones() > 0] = self.nlabels("bones")

    def init_run_list(self, run_list):
        if run_list is not None:
            self.runner.extend(run_list)
        else:
            # default run
            if self.input_annotaion_file is not None:
                self.runner.append(self.json_annotation_import)
            if self.run_organ_segmentation:
                self.runner.append(self.ninteractivity)
            if self.run_vessel_segmentation:
                self.runner.append(self.portalVeinSegmentation, **self.run_vessel_segmentation_params)
            self.runner.append(self.save_outputs)
        pass

    def make_run(self):
        """ Non-interactive mode
        :return:
        """
        import time
        t0 = time.time()
        t1 = time.time()
        self.runner.run()
        # if self.input_annotaion_file is not None:
        #     self.json_annotation_import()
        #     tt = time.time()
        #     logger.debug("make run input af {}, {}".format(tt - t0 , tt - t1))
        #     t1 = tt
        # if self.run_organ_segmentation:
        #     self.ninteractivity()
        #     self.slab["liver"] = 7
        #     self.segmentation = (self.segmentation == 1).astype('int8') * self.slab["liver"]
        #     tt = time.time()
        #     logger.debug("makerun organ seg {}, {}".format(tt - t0, tt - t1))
        #     t1 = tt
        # self.slab["porta"] = 1
        # if self.run_vessel_segmentation:
        #     data = {}
        #     data['segmentation'] = self.segmentation
        #     data['slab'] = self.slab
        #     self.portalVeinSegmentation(**self.run_vessel_segmentation_params)
        #     tt = time.time()
        #     logger.debug("makerun pv seg{}, {}".format(tt - t0, tt - t1))
        #     t1 = tt
        #
        # self.save_outputs()
        tt = time.time()
        logger.debug("make run end time {}, {}".format(tt - t0, tt - t1))

    def split_vessel(self, input_label=None, output_label1=1, output_label2=2, **kwargs):
        """
        Split vessel based on user interactivity.

        More documentation in virtual_resection.split_vessel()


        :param input_label:
        :param output_label1:
        :param output_label2:
        :param kwargs: read function virtual_resection.split_vessel() for more information.
        :return:
        """
        if input_label is None:
            from PyQt4.QtCore import pyqtRemoveInputHook
            pyqtRemoveInputHook()
            # mport ipdb; ipdb.set_trace() # BREAKPOINT
            print("label of vessel to split")
            print("--------------------")
            print("for example >> input_label = 2 ")
            input_label = "porta"
            import ipdb
            ipdb.set_trace()

        from . import virtual_resection
        datap = self.export()
        seeds = virtual_resection.cut_editor_old(datap, label=input_label)
        lab, cut_by_user = virtual_resection.split_vessel(datap=datap, seeds=seeds, input_label=input_label, **kwargs)
        self.segmentation[lab == 1] = output_label1
        self.segmentation[lab == 2] = output_label2

    def new_label_from_compact_segmentation(self, seeds):
        """

        :param seeds:
        :return:
        """

    def split_organ_by_two_vessels(
            self, output_label1=1, output_label2=5,
            organ_label=1,
            seed_label1=1,
            seed_label2=2,
            **kwargs):
        """

        :param output_label1:
        :param output_label2:
        :param organ_label:
        :param seed_label1:
        :param seed_label2:
        :param kwargs:
        :return:


        :py:segmentation:
        """
        from . import virtual_resection
        datap = self.export()
        segm, dist1, dist2 = virtual_resection.split_organ_by_two_vessels(
            datap, seeds=self.segmentation,
            organ_label=organ_label,
            seed_label1=seed_label1,
            seed_label2=seed_label2,
            **kwargs)
        # import sed3
        # ed = sed3.sed3(segm)
        # ed.show()
        self.segmentation[segm == 1] = self.nlabels(output_label1)
        self.segmentation[segm == 2] = self.nlabels(output_label2)

    def label_volumetric_vessel_tree(self, vessel_label=None, write_to_oseg=True, new_label_str_format="{}{:03d}"):
        """
        Select one vessel tree, label it by branches and put it in segmentation and slab.

        :param vessel_label: int or string label with vessel. Everything above zero is used if vessel_label is set None.
        :param write_to_oseg: Store output into oseg.segmentation if True. The slab is also updated.
        :param new_label_str_format: format of new slab
        :return:
        """
        from . import virtual_resection
        return virtual_resection.label_volumetric_vessel_tree(
            self,
            vessel_label=vessel_label,
            write_to_oseg=write_to_oseg,
            new_label_str_format=new_label_str_format
        )


def logger_init():  # pragma: no cover
    # import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        '%(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # fformatter = logging.Formatter(
    # '%(asctime)s - %(name)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s'
    # )
    fformatter = logging.Formatter('%(asctime)s %(levelname)-8s %(name)-18s %(lineno)-5d %(funcName)-12s %(message)s')
    logfile = "lisa.log"
    if op.exists(op.expanduser("~/lisa_data/")):
        logfile = op.expanduser("~/lisa_data/lisa.log")
    # problems on windows
    # fh = logging.handlers.RotatingFileHandler(logfile, maxBytes=100000, backupCount=9)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(fformatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.debug('logger started')

    return ch, fh


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

    # cfg = config.get_config("organ_segmentation.config", cfg)
    cfg.update(config_default.CONFIG_DEFAULT)
    user_config_path = os.path.join(cfg['output_datapath'],
                                    "organ_segmentation.config")
    config.check_config_version_and_remove_old_records(
        user_config_path, version=config_version,
        records_to_save=['experiment_caption', 'lisa_operator_identifier'])
    # read user defined config in user data
    cfg = config.get_config(user_config_path, cfg)
    return cfg


def parser_init(cfg):  # pragma: no cover

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
        '-iaf', '--input_annotation_file', type=str,  # type=int,
        help='Set input json annotation file',
        default=None)
    parser.add_argument(
        '-oaf', '--output_annotation_file', type=str,  # type=int,
        help='Set output json annotation file',
        default=None)
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
        '-icn', '--make_icon',
        action='store_true',
        help='Create desktop icon on OS X and Linux',
        default=False
    )
    parser.add_argument(
        '-gsd', '--get_sample_data',
        action='store_true',
        help='Download sample data',
        default=False
    )
    parser.add_argument(
        '--autolisa',
        action='store_true',
        help='run autolisa in dir',
        default=False
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


def boltzman(x, xmid, tau):
    """
    evaluate the boltzman function with midpoint xmid and time constant tau
    over x
    """
    return 1. / (1. + np.exp(-(x - xmid) / tau))


def main(app=None, splash=None):  # pragma: no cover

    #    import ipdb; ipdb.set_trace() # BREAKPOINT
    try:
        ch, fh = logger_init()
        cfg = lisa_config_init()
        args = parser_init(cfg)

        if cfg['make_icon'] is True:
            import lisa_data
            lisa_data.make_icon()
            return
        if cfg['get_sample_data'] is True:
            import dataset
            dataset.get_sample_data()
            return

        # rint args["arg"]
        oseg_argspec_keys = config.get_function_keys(
            OrganSegmentation.__init__)

        if args["debug"]:
            ch.setLevel(logging.DEBUG)
            args["debug_mode"] = True

        if args["iparams"] is not None:
            params = misc.obj_from_file(args["iparams"], filetype='pickle')

        else:
            params = config.subdict(args, oseg_argspec_keys)

        logger.debug('params ' + str(params))
        if args["autolisa"]:
            # if splash is not None:
            #     splash.finish()
            from . import autolisa
            al = autolisa.AutoLisa()
            al.run_in_paths(args["datapath"])
            return

        oseg = OrganSegmentation(**params)

        if args["no_interactivity"]:
            oseg.make_run()
            # oseg.ninteractivity()
            # oseg.save_outputs()
        else:
            # mport_gui()
            from .lisaWindow import OrganSegmentationWindow
            import PyQt4
            import PyQt4.QtGui
            from PyQt4.QtGui import QApplication
            if app is None:
                app = QApplication(sys.argv)
            # Create and display the splash screen
            oseg_w = OrganSegmentationWindow(oseg, qapp=app)  # noqa
            if splash is not None:
                splash.finish(oseg_w)
            #    import pdb; pdb.set_trace()
            sys.exit(app.exec_())

    except Exception as e:
        import traceback
        # mport exceptionProcessing
        exceptionProcessing.reportException(e)
        print(traceback.format_exc())
        # aise e


if __name__ == "__main__":
    main()
    print("Thank you for using Lisa")
