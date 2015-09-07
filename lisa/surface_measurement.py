#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.

"""
Measurement of object surface.

data3d: 3D numpy array
segmentation: 3D numpyarray

"""

import logging
logger = logging.getLogger(__name__)
import argparse
import numpy as np
import scipy


def surface_density(segmentation, voxelsize_mm, aoi=None, sond_raster_mm=None):
    """
    :segmentation: is ndarray with 0 and 1
    :voxelsize_mm: is array of three numbers specifiing size of voxel for each
        axis
    :aoi: is specify area of interest. It is ndarray with 0 and 1
    :sond_raster_mm: unimplemented. It is parametr of sonds design
    """

    axis = 0
    if sond_raster_mm is None:
        sond_raster_mm = voxelsize_mm
    if aoi is None:
        aoi = np.ones(segmentation.shape)

    im_edg = find_edge(segmentation, axis=axis)
    im_edg = im_edg * aoi
    im_sond, aoi_sond = bufford_needle_sond(
        im_edg, voxelsize_mm, sond_raster_mm, axis=axis, aoi=aoi)

# isotropic fakir  - kubinova janacek
    # est S = 2 \frac{1}{n} \sum_{i=1}^{n} \frac{v}{l_i} \cdot l_i

# celkova delka sond
    # n_needle = (im_sond.shape[1] * im_sond.shape[2])
    # one_needle_l = im_sond.shape[0] * voxelsize_mm[0]
    # length = n_needle * one_needle_l
    length = np.sum(aoi_sond > 0) * voxelsize_mm[0]

# inverse of the probe per unit volume v/l_i
    # ippuv = (
    #     (np.prod(sond_raster_mm) * im_sond.shape[axis])
    #     /
    #     (sond_raster_mm[axis] * im_sond.shape[axis])
    # )
# Pocet pruseciku
    # Ii = np.sum(np.abs(im_sond))
    Ii = np.sum(np.abs(im_sond))

    # import sed3
    # ed = sed3.sed3(im_sond)
    # ed.show()

    # Kubinova2001
    # print "Ii = ", Ii
    Sv = 2.0 * Ii / length
    # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

    return Sv


def find_edge(segmentation, axis):
    if axis == 0:
        k = np.array([[[1]], [[-1]]])
    elif axis == 1:
        k = np.array([[[1], [-1]]])
    elif axis == 2:
        k = np.array([[[1, -1]]])

    retval = scipy.ndimage.filters.convolve(segmentation, k, mode='constant')

    # retval =  scipy.ndimage.sobel(segmentation, axis=axis, mode='constant')
    return retval


def bufford_needle_sond(data3d, voxelsize_mm, raster_mm, axis, aoi):
    # TODO implement needle
    # retval = data3d[::2, ::2, ::2]
    retval = data3d

    return retval, aoi


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
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        required=True,
        help='input file'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)


if __name__ == "__main__":
    main()
