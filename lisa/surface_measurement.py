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


def surface_per_volume(segmentation, voxelsize_mm, sond_raster_mm=None):
    axis = 0
    if sond_raster_mm is None:
        sond_raster_mm = voxelsize_mm
    im_edg = find_edge(segmentation, axis=axis)
    im_sond = bufford_needle_sond(
        im_edg, voxelsize_mm, sond_raster_mm, axis=axis)

    # est S = 2 \frac{1}{n} \sum_{i=1}^{n} \frac{v}{l_i} \cdot l_i

# celkova delka sond
    n_needle = (im_sond.shape[1] * im_sond.shape[2])
    one_needle_l = im_sond.shape[0] * voxelsize_mm[0]
    length = n_needle * one_needle_l


# inverse of the probe per unit volume v/l_i
    # ippuv = (
    #     (np.prod(sond_raster_mm) * im_sond.shape[axis])
    #     /
    #     (sond_raster_mm[axis] * im_sond.shape[axis])
    # )
# Pocet pruseciku
    Ii = np.sum(np.abs(im_sond))

    # import sed3
    # ed = sed3.sed3(im_sond)
    # ed.show()

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


def bufford_needle_sond(data3d, voxelsize_mm, raster_mm, axis):
    # TODO implement needle
    # retval = data3d[::2, ::2, ::2]
    retval = data3d

    return retval


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
