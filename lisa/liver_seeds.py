#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © %YEAR%  <>
#
# Distributed under terms of the %LICENSE% license.

"""

"""

import logging

logger = logging.getLogger(__name__)
import numpy as np
import argparse

import liver_model
from pysegbase import pycut
import imtools
import body_navigation
import data_manipulation


def automatic_liver_seeds(data3d, seeds, voxelsize_mm, fn_mdl='~/lisa_data/liver_intensity.Model.p'):
    # fn_mdl = op.expanduser(fn_mdl)
    mdl = pycut.Model({'mdl_stored_file':fn_mdl, 'fv_extern': liver_model.intensity_localization_fv})
    working_voxelsize_mm = np.asarray([4, 4, 4])

    data3dr = imtools.resize_to_mm(data3d, voxelsize_mm, working_voxelsize_mm)

    lik1 = mdl.likelihood_from_image(data3dr, working_voxelsize_mm, 0)
    lik2 = mdl.likelihood_from_image(data3dr, working_voxelsize_mm, 1)

    dl = lik2 - lik1


    # Liver seed center

    # seed tam, kde je to nejpravděpodovnější - moc nefunguje
    seed1 = np.unravel_index(np.argmax(dl), dl.shape)
    seed1_mm = seed1 * working_voxelsize_mm

    # find slice with maximum  liver profile
    sos = np.sum(np.sum(dl > 0, axis=1, keepdims=True), axis=2)
    maximal_slice_index = np.argmax(sos)
    # print sos.shape
    xynz = np.nonzero((dl>0)[maximal_slice_index, :, :])

    import PyQt4; PyQt4.QtCore.pyqtRemoveInputHook()
    import ipdb; ipdb.set_trace()

    seed1 = np.array([maximal_slice_index, np.mean(xynz[0]), np.mean(xynz[1])])
    seed1_mm = seed1 * working_voxelsize_mm



    # background seed center
    bn = body_navigation.BodyNavigation(data3d, voxelsize_mm)

    bn.get_center()
    seed2_mm = bn.center_mm

    # seeds should be on same slide
    # seed2_mm[0] = seed1_mm[0]
    seeds = data_manipulation.add_seeds_mm(
        seeds, voxelsize_mm,
        [seed1_mm[0]],
        [seed1_mm[1]],
        [seed1_mm[2]],
        label=1,
        radius=20,
        width=15
    )
    seeds = data_manipulation.add_seeds_mm(
        seeds, voxelsize_mm,
        [seed2_mm[0]],
        [seed2_mm[1]],
        [seed2_mm[2]],
        label=2,
        radius=10,
        width=5
    )
    import sed3
    sed3.show_slices(data3dr, dl > 40.0, slice_step=10)
    return seeds




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