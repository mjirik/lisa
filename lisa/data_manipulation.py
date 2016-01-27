#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@hp-mjirik>
#
# Distributed under terms of the MIT license.

"""
Crop, resize, reshape
"""

import logging
logger = logging.getLogger(__name__)
import argparse
import scipy
import numpy as np

import qmisc

def select_labels(segmentation, labels):
    ds = np.zeros(segmentation.shape, np.bool)
    for lab in labels:
        print "print zpracovavam ", lab
        dadd = (segmentation == lab)

        ds = ds | dadd

    return ds

def unbiased_brick_filter(binary_data, crinfo):
    """
    return only binary object which suits with unbiased brick 

    :param data: 3D ndimage data
    :param crinfo: crinfo 
    http://www.stereology.info/the-optical-disector-and-the-unbiased-brick/
    """
    binary_data = binary_data.copy()
    crinfo = qmisc.fix_crinfo(crinfo)

    imlab, num_features = scipy.ndimage.measurements.label(binary_data)

    brick_neg_mask = np.zeros(binary_data.shape, dtype=np.uint8)
    brick_neg_mask[
            crinfo[0][0]:crinfo[0][1],
            crinfo[1][0]:crinfo[1][1],
            crinfo[2][0]:crinfo[2][1]
        ] = 1

    exclude_mask = np.zeros(binary_data.shape, dtype=np.uint8)
    exclude_mask[:,:, crinfo[2][1]:] = 1
    exclude_mask[:,crinfo[1][1]:, crinfo[2][0]:] = 1
    exclude_mask[:crinfo[0][0],crinfo[1][0]:, crinfo[2][0]:] = 1

    # remove what is not in touch with brick
    imlab = keep_what_is_in_touch_with_mask(
            imlab, brick_neg_mask, max_label=num_features)
    # remove what is in touch with exclude
    imlab = remove_what_is_in_touch_with_mask(imlab, exclude_mask)




    return (imlab > 0).astype(binary_data.dtype)
    


    # return data[
    #     __int_or_none(crinfo[0][0]):__int_or_none(crinfo[0][1]),
    #     __int_or_none(crinfo[1][0]):__int_or_none(crinfo[1][1]),
    #     __int_or_none(crinfo[2][0]):__int_or_none(crinfo[2][1])
    #     ]

def keep_what_is_in_touch_with_mask(imlab, keep_mask, max_label):
    datatmp = imlab * keep_mask
    nz = np.nonzero(datatmp)
    labels_to_keep = imlab[nz[0], nz[1], nz[2]]
    labels_to_keep = np.unique(labels_to_keep)

    for lab in range(0, max_label):
        if lab in labels_to_keep:
            pass
        else:
            imlab[imlab == lab] = 0

    return imlab

def remove_what_is_in_touch_with_mask(imlab, exclude_mask):
    datatmp = imlab * exclude_mask
    nz = np.nonzero(datatmp)
    labels_to_exclude = imlab[nz[0], nz[1], nz[2]]
    labels_to_exclude = np.unique(labels_to_exclude)


    for lab in labels_to_exclude:
        imlab[imlab == lab] = 0

    return imlab



def add_seeds_mm(data_seeds, voxelsize_mm, z_mm, x_mm, y_mm, label, radius, width=1):

    """
    Function add circle seeds to one slice with defined radius.

    It is possible set more seeds on one slice with one dimension

    x_mm, y_mm coordinates of circle in mm. It may be array.
    z_mm = slice coordinates  in mm. It may be array
    :param label: one number. 1 is object seed, 2 is background seed
    :param radius: is radius of circle in mm
    :param width: makes circle with defined width (repeat circle every milimeter)

    """

    # this do not work for ndarrays
    # if type(x_mm) is not list:
    #     x_mm = [x_mm]
    # if type(y_mm) is not list:
    #     x_mm = [y_mm]
    # if type(z_mm) is not list:
    #     z_mm = [z_mm]
    z_mm = np.asarray(z_mm)
    # repeat circle every milimiter
    for i in range(0, width + 1):
        data_seeds = _add_seeds_mm_in_one_slice(data_seeds, voxelsize_mm, z_mm + i, x_mm, y_mm, label, radius)
    return data_seeds

def _add_seeds_mm_in_one_slice(data_seeds, voxelsize_mm, z_mm, x_mm, y_mm, label, radius):
    x_mm = np.asarray(x_mm)
    y_mm = np.asarray(y_mm)
    z_mm = np.asarray(z_mm)

    for i in range(0, len(x_mm)):

        # xx and yy are 200x200 tables containing the x and y coordinates
        # values. mgrid is a mesh creation helper
        xx, yy = np.mgrid[
                 :data_seeds.shape[1],
                 :data_seeds.shape[2]
                 ]
        # circles contains the squared distance to the (100, 100) point
        # we are just using the circle equation learnt at school
        circle = (
                     (xx - x_mm[i] / voxelsize_mm[1]) ** 2 +
                     (yy - y_mm[i] / voxelsize_mm[2]) ** 2
                 ) ** (0.5)
        # donuts contains 1's and 0's organized in a donut shape
        # you apply 2 thresholds on circle to define the shape
        # slice jen s jednim kruhem
        slicecircle = circle < radius
        slicen = int(z_mm / voxelsize_mm[0])
        # slice s tim co už je v něm nastaveno
        slicetmp = data_seeds[slicen, :, :]
        # mport pdb; pdb.set_trace()

        slicetmp[slicecircle == 1] = label

        data_seeds[slicen, :, :] = slicetmp
    return data_seeds


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
