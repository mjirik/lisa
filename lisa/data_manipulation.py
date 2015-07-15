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


def unbiased_brick_filter(binary_data, crinfo):
    """
    :param data: 3D ndimage data
    :param crinfo: crinfo 
    http://www.stereology.info/the-optical-disector-and-the-unbiased-brick/
    """
    crinfo = qmisc.fix_crinfo(crinfo)

    exclude_mask = np.zeros(binary_data.shape, dtype=np.uint8)
    exclude_mask[:,:, crinfo[2][1]:] = 1
    exclude_mask[:,crinfo[1][1]:, crinfo[2][0]:] = 1
    exclude_mask[:crinfo[0][0],crinfo[1][0]:, crinfo[2][0]:] = 1

    imlab, num_features = scipy.ndimage.measurements.label(binary_data)


    datatmp = imlab * exclude_mask
    nz = np.nonzero(datatmp)
    labels_to_exclude = imlab[nz[0], nz[1], nz[2]]
    labels_to_exclude = np.unique(labels_to_exclude)


    for lab in labels_to_exclude:
        imlab[imlab == lab] = 0

    return (imlab > 0).astype(binary_data.dtype)
    


    # return data[
    #     __int_or_none(crinfo[0][0]):__int_or_none(crinfo[0][1]),
    #     __int_or_none(crinfo[1][0]):__int_or_none(crinfo[1][1]),
    #     __int_or_none(crinfo[2][0]):__int_or_none(crinfo[2][1])
    #     ]


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
