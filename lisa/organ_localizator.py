#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © %YEAR%  <>
#
# Distributed under terms of the %LICENSE% license.

"""
Module is used to train liver localizator based on body_navigation module.
Output file is placed into ~/lisa_data/liver.ol.p
"""

import logging

logger = logging.getLogger(__name__)
import numpy as np
import argparse
import glob
import os.path as op
import traceback

import imtools
import imtools.ml
import io3d

import sys
import os
import os.path as op
sys.path.append(op.join(op.dirname(os.path.abspath(__file__)), "../../bodynavigation/"))

def externfv(data3d, voxelsize_mm):        # scale
        fv = []
        # f0 = scipy.ndimage.filters.gaussian_filter(data3d, sigma=3).reshape(-1, 1)
        #f1 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=1).reshape(-1, 1) - f0
        #f2 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=5).reshape(-1, 1) - f0
        #f3 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=10).reshape(-1, 1) - f0
        #f4 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=20).reshape(-1, 1) - f0
        # position asdfas
        import bodynavigation
        ss = bodynavigation.BodyNavigation(data3d, voxelsize_mm)
        fd1 = ss.dist_to_lungs().reshape(-1, 1)
        fd2 = ss.dist_to_spine().reshape(-1, 1)
        fd3 = ss.dist_sagittal().reshape(-1, 1)
        fd4 = ss.dist_coronal().reshape(-1, 1)
        fd5 = ss.dist_axial().reshape(-1, 1)
        fd6 = ss.dist_to_surface().reshape(-1, 1)
        fd7 = ss.dist_diaphragm().reshape(-1, 1)

        # f6 = scipy.ndimage.filters.gaussian_filter(data3d, sigma=[20, 1, 1]).reshape(-1, 1) - f0
        # f7 = scipy.ndimage.filters.gaussian_filter(data3d, sigma=[1, 20, 1]).reshape(-1, 1) - f0
        # f8 = scipy.ndimage.filters.gaussian_filter(data3d, sigma=[1, 1, 20]).reshape(-1, 1) - f0


        # print "fv shapes ", f0.shape, fd2.shape, fd3.shape
        fv = np.concatenate([
                # f0,
#                 f1, f2, f3, f4,
                fd1, fd2, fd3, fd4, fd5, fd6, fd7,
                #f6, f7, f8
            ], 1)


        return fv


class OrganLocalizator():
    def __init__(self):
        feature_function = None
        self.working_voxelsize_mm = [1.5, 1.5, 1.5]
        self.data=None
        self.target=None
#         self.cl = sklearn.naive_bayes.GaussianNB()
#         self.cl = sklearn.mixture.GMM()
        #self.cl = sklearn.tree.DecisionTreeClassifier()
        if feature_function is None:
            feature_function = externfv
        self.feature_function = feature_function
        self.cl = imtools.ml.gmmcl.GMMCl(n_components=6)

    def save(self, filename='saved.ol.p'):
        """
        Save model to pickle file
        """
        import dill as pickle
        sv = {
            # 'feature_function': self.feature_function,
            'cl': self.cl

        }
        pickle.dump(sv, open(filename, "wb"))

    def load(self, mdl_file='saved.ol.p'):
        import dill as pickle
        sv = pickle.load(open(mdl_file, "rb"))
        self.cl= sv['cl']
        # self.feature_function = sv['feature_function']


    def _fv(self, data3dr):
        return self.feature_function(data3dr, self.working_voxelsize_mm)


    def _add_to_training_data(self, data3dr, segmentationr):
        fv = self._fv(data3dr)
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
        data3dr = imtools.qmisc.resize_to_mm(data3d, voxelsize_mm, self.working_voxelsize_mm)
        fv = self._fv(data3dr)
        pred = self.cl.predict(fv)
        return imtools.qmisc.resize_to_shape(pred.reshape(data3dr.shape), data3d.shape)

    def predict_w(self, data3d, voxelsize_mm, weight, label0=0, label1=1):
        """
        segmentation with weight factor
        :param data3d:
        :param voxelsize_mm:
        :param weight:
        :return:
        """
        scores = self.scores(data3d, voxelsize_mm)
        out = scores[label1] > (weight * scores[label0])

        return out

    def scores(self, data3d, voxelsize_mm):
        data3dr = imtools.qmisc.resize_to_mm(data3d, voxelsize_mm, self.working_voxelsize_mm)
        fv = self._fv(data3dr)
        scoreslin = self.cl.scores(fv)
        scores = {}
        for key in scoreslin:
            scores[key] = imtools.qmisc.resize_to_shape(scoreslin[key].reshape(data3dr.shape), data3d.shape)

        return scores


    def __preprocessing(data3d):
        pass

    def add_train_data(self, data3d, segmentation, voxelsize_mm):
        data3dr = imtools.qmisc.resize_to_mm(data3d, voxelsize_mm, self.working_voxelsize_mm)
        segmentationr = imtools.qmisc.resize_to_shape(segmentation, data3dr.shape)

        print np.unique(segmentationr), data3dr.shape, segmentationr.shape
        self._add_to_training_data(data3dr, segmentationr)
        #f1 scipy.ndimage.filters.gaussian_filter(data3dr, sigma=5)


def train_liver_localizator_from_sliver_data(
        output_file="~/lisa_data/liver.ol.p",
        sliver_reference_dir='~/data/medical/orig/sliver07/training/',
        orig_pattern="*orig*[1-9].mhd",
        ref_pattern="*seg*[1-9].mhd"):
    """
    Train liver localization based on spine and lungs from all sliver reference data except number 10 and 20
    :param output_file:
    :param sliver_reference_dir:
    :param orig_pattern:
    :param ref_pattern:
    :return:
    """

    sliver_reference_dir = op.expanduser(sliver_reference_dir)

    orig_fnames = glob.glob(sliver_reference_dir + orig_pattern)
    ref_fnames = glob.glob(sliver_reference_dir + ref_pattern)

    orig_fnames.sort()
    ref_fnames.sort()


    sf = OrganLocalizator()
    vs = []
    hist=[]
    fhs_list = []
    for oname, rname in zip(orig_fnames, ref_fnames):
        print oname
        data3d_orig, metadata = io3d.datareader.read(oname)
        vs_mm1 = metadata['voxelsize_mm']
        data3d_seg, metadata = io3d.datareader.read(rname)
        vs_mm = metadata['voxelsize_mm']


        vs.append(vs_mm)

        try:
            sf.add_train_data(data3d_orig, data3d_seg, voxelsize_mm=vs_mm)
        except:
            traceback.print_exc()
            print "problem"
            pass
        # fvhn = copy.deepcopy(fvh)
        #fhs_list.append(fvh)


    sf.fit()
    sf.save(op.expanduser(output_file))

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
        default="~/lisa_data/liver.ol.p",
        help='output file'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)
    train_liver_localizator_from_sliver_data(args.outputfile)


if __name__ == "__main__":
    main()