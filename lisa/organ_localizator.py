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

from imtools.trainer3d import Trainer3D as OrganLocalizator

def externfv(data3d, voxelsize_mm):        # scale
    return localization_fv(data3d=data3d, voxelsize_mm=voxelsize_mm)


def localization_fv(data3d, voxelsize_mm):        # scale
        fv = []
        # f0 = scipy.ndimage.filters.gaussian_filter(data3d, sigma=3).reshape(-1, 1)
        #f1 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=1).reshape(-1, 1) - f0
        #f2 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=5).reshape(-1, 1) - f0
        #f3 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=10).reshape(-1, 1) - f0
        #f4 = scipy.ndimage.filters.gaussian_filter(data3dr, sigma=20).reshape(-1, 1) - f0
        # position asdfas
        import body_navigation as bn
        ss = bn.BodyNavigation(data3d, voxelsize_mm)
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

def localization_intensity_fv(data3d, voxelsize_mm):
    """
    feature vector combine intensity and localization information
    :param data3d:
    :param voxelsize_mm:
    :return:
    """
    fvloc = localization_fv(data3d, voxelsize_mm)
    fv = np.concatenate([
        fvloc,
        data3d.reshape(-1, 1)
        # f0,
        #                 f1, f2, f3, f4,
        #f6, f7, f8
    ], 1)
    return fv




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
        print(oname)
        data3d_orig, metadata = io3d.datareader.read(oname, dataplus_format=False)
        vs_mm1 = metadata['voxelsize_mm']
        data3d_seg, metadata = io3d.datareader.read(rname, dataplus_format=False)
        vs_mm = metadata['voxelsize_mm']


        vs.append(vs_mm)

        try:
            sf.add_train_data(data3d_orig, data3d_seg, voxelsize_mm=vs_mm)
        except:
            traceback.print_exc()
            print("problem")
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