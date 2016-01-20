#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © %YEAR%  <>
#
# Distributed under terms of the %LICENSE% license.

"""
This module is used to train liver model with intensity.

First use organ_localizator module to train intensity independent model.
"""

import logging

logger = logging.getLogger(__name__)

import os
import sys
import os.path as op
sys.path.append(op.join(op.dirname(os.path.abspath(__file__)), "../../pysegbase/"))
import argparse
import glob
import traceback
import numpy as np

import io3d
from imtools import qmisc

def add_fv_extern_into_modelparams(modelparams):
    # import PyQt4; PyQt4.QtCore.pyqtRemoveInputHook()
    # import ipdb; ipdb.set_trace()

    if "fv_type" in modelparams.keys() and modelparams['fv_type'] == 'fv_extern':
        if type(modelparams['fv_extern']) == str:
            fv_extern_str = modelparams['fv_extern']
            if fv_extern_str == "intensity_localization_fv":
                modelparams['fv_extern'] = intensity_localization_fv
    return modelparams

def intensity_localization_fv(data3dr, voxelsize_mm, seeds=None, unique_cls=None):        # scale
    """
    Use organ_localizator features plus intensity features

    :param data3dr:
    :param voxelsize_mm:
    :param seeds:
    :param unique_cls:
    :return:
    """
    import scipy
    import numpy as np
    import os.path as op
    try:
        from lisa import organ_localizator
    except:
        import organ_localizator

#         print "po importech"
    fv = []
    f0 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=1).reshape(-1, 1)
    f1 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=3).reshape(-1, 1)
    #f2 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=5).reshape(-1, 1) - f0
    #f3 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=10).reshape(-1, 1) - f0
    #f4 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=20).reshape(-1, 1) - f0
    # position
    #ss = lisa.body_navigation.BodyNavigation(data3dr, voxelsize_mm)
    #ss.feature_function(data3d, voxelsize_mm)
    #fd1 = ss.dist_to_lungs().reshape(-1, 1)
    #fd2 = ss.dist_to_spine().reshape(-1, 1)
    #fd3 = ss.dist_sagittal().reshape(-1, 1)
    #fd4 = ss.dist_coronal().reshape(-1, 1)
    #fd5 = ss.dist_to_surface().reshape(-1, 1)
    #fd6 = ss.dist_diaphragm().reshape(-1, 1)

#         print "pred f6"
    f6 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=[20, 1, 1]).reshape(-1, 1) - f1
    f7 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=[1, 20, 1]).reshape(-1, 1) - f1
    f8 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=[1, 1, 20]).reshape(-1, 1) - f1

#         print "pred organ_localizator"
    ol = organ_localizator.OrganLocalizator()
    ol.load(op.expanduser("~/lisa_data/liver.ol.p"))

    fdall = ol.feature_function(data3dr, voxelsize_mm)


    middle_liver = ol.predict_w(data3dr, voxelsize_mm, 0.85)
    mn = np.median(data3dr[middle_liver==1])
    fdn = np.ones(f0.shape) * mn


    # print "fv shapes ", f0.shape, fd2.shape, fd3.shape
    fv = np.concatenate([
            f0,
            f1,
#                 f2, f3, f4,
#                 fd1, fd2, fd3, fd4, fd5, fd6,
            fdall,
            f6, f7, f8,
            fdn,

        ], 1)
    if seeds is not None:
#             logger.debug("seeds " + str(seeds))
#             print "seeds ", seeds
        sd = seeds.reshape(-1,1)
        selection = np.in1d(sd, unique_cls)
        fv = fv[selection]
        sd = sd[selection]
        # sd = sd[]
        return fv, sd

    return fv

class ModelTrainer():
    def __init__(self, feature_function=None, modelparams={}):
        from pysegbase import pycut
        self.working_voxelsize_mm = [1.5, 1.5, 1.5]
        self.data=None
        self.target=None
#         self.cl = sklearn.naive_bayes.GaussianNB()
#         self.cl = sklearn.mixture.GMM()
        #self.cl = sklearn.tree.DecisionTreeClassifier()
        if feature_function is None:
            feature_function = intensity_localization_fv
#         self.feature_function = feature_function

        modelparams_working = {

            'fv_type': "fv_extern",
            'fv_extern': feature_function,
            'type': 'gmmsame',
            'params': {'cvtype': 'full', 'n_components': 15},
            'adaptation': 'original_data',


        }
        modelparams_working.update(modelparams)
        self.cl = pycut.Model(modelparams=modelparams_working)


    def _fv(self, data3dr, voxelsize_mm):
        fev = self.cl.features_from_image(data3dr, voxelsize_mm)
        # print fev
        return fev

    def _add_to_training_data(self, data3dr, voxelsize_mm, segmentationr):
#         print "funkce _add_to_training_data ()    "
#         print data3dr.shape
#         print segmentationr.shape
        fv = self._fv(data3dr, voxelsize_mm)
        data = fv[::50]
        target = np.reshape(segmentationr, [-1, 1])[::50]
        #         print "shape ", data.shape, "  ", target.shape

        if self.data is None:
            self.data = data
            self.target = target
        else:
            self.data = np.concatenate([self.data, data], 0)
            self.target = np.concatenate([self.target, target], 0)
        # self.cl.fit(data, target)

        #f1[segmentationr == 0]
    def fit(self):
        #         print "sf fit data shape ", self.data.shape
        self.cl.fit(self.data, self.target)

    def predict(self, data3d, voxelsize_mm):
        data3dr = qmisc.resize_to_mm(data3d, voxelsize_mm, self.working_voxelsize_mm)
        fv = self._fv(data3dr)
#         print "shape predict ", fv.shape,
        pred = self.cl.predict(fv)
#         print "predict ", pred.shape,
        return qmisc.resize_to_shape(pred.reshape(data3dr.shape), data3d.shape)

    def scores(self, data3d, voxelsize_mm):
        data3dr = qmisc.resize_to_mm(data3d, voxelsize_mm, self.working_voxelsize_mm)
        fv = self._fv(data3dr)
#         print "shape predict ", fv.shape,
        scoreslin = self.cl.scores(fv)
        scores = {}
        for key in scoreslin:
            scores[key] = qmisc.resize_to_shape(scoreslin[key].reshape(data3dr.shape), data3d.shape)

        return scores


    def __preprocessing(data3d):
        pass

    def add_train_data(self, data3d, segmentation, voxelsize_mm):
        data3dr = qmisc.resize_to_mm(data3d, voxelsize_mm, self.working_voxelsize_mm)
        segmentationr = qmisc.resize_to_shape(segmentation, data3dr.shape)

        logger.debug(str(np.unique(segmentationr)))
        logger.debug(str(data3dr.shape) + str(segmentationr.shape))
        self._add_to_training_data(data3dr, self.working_voxelsize_mm, segmentationr)
        #f1 scipy.ndimage.filters.gaussian_filter(data3dr, sigma=5)

    def train_liver_model_from_sliver_data(
            self,
            output_file="~/lisa_data/liver_intensity.Model.p",
            sliver_reference_dir='~/data/medical/orig/sliver07/training/',
            orig_pattern="*orig*[1-9].mhd",
            ref_pattern="*seg*[1-9].mhd",
        ):

        sliver_reference_dir = op.expanduser(sliver_reference_dir)

        orig_fnames = glob.glob(sliver_reference_dir + orig_pattern)

        ref_fnames = glob.glob(sliver_reference_dir + ref_pattern)

        orig_fnames.sort()
        ref_fnames.sort()

        for oname, rname in zip(orig_fnames, ref_fnames):
            logger.debug(oname)
            data3d_orig, metadata = io3d.datareader.read(oname)
            vs_mm1 = metadata['voxelsize_mm']
            data3d_seg, metadata = io3d.datareader.read(rname)
            vs_mm = metadata['voxelsize_mm']

            # liver have label 1, background have label 2
            data3d_seg = 2 - data3d_seg

            #     sf.add_train_data(data3d_orig, data3d_seg, voxelsize_mm=vs_mm)
            try:
                self.add_train_data(data3d_orig, data3d_seg, voxelsize_mm=vs_mm)
            except:
                traceback.print_exc()
                print "problem - liver model"
                pass
                # fvhn = copy.deepcopy(fvh)
                #fhs_list.append(fvh)


        self.fit()

        output_file = op.expanduser(output_file)
        self.cl.save(output_file)

def train_liver_model_from_sliver_data(
        output_file="~/lisa_data/liver_intensity.Model.p",
        sliver_reference_dir='~/data/medical/orig/sliver07/training/',
        orig_pattern="*orig*[1-9].mhd",
        ref_pattern="*seg*[1-9].mhd",
        modelparams={}
):
    force = True
    force = False
    fname_vs = 'vs_stats.csv'
    fname_fhs = 'fhs_stats.pklz'

    sliver_reference_dir = op.expanduser(sliver_reference_dir)

    orig_fnames = glob.glob(sliver_reference_dir + orig_pattern)

    ref_fnames = glob.glob(sliver_reference_dir + ref_pattern)

    orig_fnames.sort()
    ref_fnames.sort()


    #if op.exists(fname_vs) and op.exists(fname_fhs) and (force is False):
    #    dfvs = pd.read_csv(fname_vs, index_col=0)
    #    fhs_list = io3d.misc.obj_from_file(fname_fhs, 'auto')

    #else:

    sf = ModelTrainer(modelparams=modelparams)
    sf.train_liver_model_from_sliver_data(
        output_file=output_file,
        sliver_reference_dir=sliver_reference_dir,
        orig_pattern=orig_pattern,
        ref_pattern=ref_pattern
    )
    return sf.data, sf.target

    # for oname, rname in zip(orig_fnames, ref_fnames):
    #     print oname
    #     data3d_orig, metadata = io3d.datareader.read(oname)
    #     vs_mm1 = metadata['voxelsize_mm']
    #     data3d_seg, metadata = io3d.datareader.read(rname)
    #     vs_mm = metadata['voxelsize_mm']
    #
    #     # liver have label 1, background have label 2
    #     data3d_seg = 2 - data3d_seg
    #
    #     #     sf.add_train_data(data3d_orig, data3d_seg, voxelsize_mm=vs_mm)
    #     try:
    #         sf.add_train_data(data3d_orig, data3d_seg, voxelsize_mm=vs_mm)
    #     except:
    #         traceback.print_exc()
    #         print "problem - liver model"
    #         pass
    #         # fvhn = copy.deepcopy(fvh)
    #         #fhs_list.append(fvh)
    #
    #
    # sf.fit()
    #
    # output_file = op.expanduser(output_file)
    # sf.cl.save(output_file)

def main():
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    # fh = logging.FileHandler('log.txt')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.debug('start')

    # input parser
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    # parser.add_argument(
    #     '-i', '--inputfile',
    #     default=None,
    #     required=True,
    #     help='input file'
    # )
    parser.add_argument(
        '-o', '--outputfile',
        default="~/lisa_data/liver_intensity.Model.p",
        help='output file'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)
    train_liver_model_from_sliver_data(args.outputfile)


if __name__ == "__main__":
    main()
