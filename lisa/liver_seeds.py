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
import imtools
import data_manipulation
import scipy


def automatic_liver_seeds(
        data3d,
        seeds,
        voxelsize_mm,
        fn_mdl='~/lisa_data/liver_intensity.Model.p',
        return_likelihood_difference=True,
        gaussian_sigma_mm=[20,20,20]):
    from pysegbase import pycut
    # fn_mdl = op.expanduser(fn_mdl)
    mdl = pycut.Model({'mdl_stored_file':fn_mdl, 'fv_extern': liver_model.intensity_localization_fv})
    working_voxelsize_mm = np.asarray([1.5, 1.5, 1.5])
    gaussian_sigma_mm = np.asarray(gaussian_sigma_mm)

    data3dr = imtools.resize_to_mm(data3d, voxelsize_mm, working_voxelsize_mm)

    lik1 = mdl.likelihood_from_image(data3dr, working_voxelsize_mm, 0)
    lik2 = mdl.likelihood_from_image(data3dr, working_voxelsize_mm, 1)

    dl = lik2 - lik1

    seeds = add_negative_notrain_seeds(seeds,lik1, lik2)


    # Liver seed center
    import scipy

    # seed tam, kde je to nejpravděpodovnější - moc nefunguje při blbém natrénování
    dst = scipy.ndimage.filters.gaussian_filter(dl, sigma=gaussian_sigma_mm/working_voxelsize_mm)
    # seed1 = np.unravel_index(np.argmax(dl), dl.shape)

    # escte jinak
    # dáme seed doprostřed oblasti
    # dst = scipy.ndimage.morphology.distance_transform_edt(dl>0)

    seed1 = np.unravel_index(np.argmax(dst), dst.shape)
    # alternativa -
    seed1_mm = seed1 * working_voxelsize_mm
    print 'seed1 ', seed1, ' shape ', dst.shape

    seed1z = seed1[0]
    seed1z_mm = seed1_mm[0]
    print seed1z_mm


    add_negative_train_seeds_blobs(
        dl < 0,
        seeds,
        working_voxelsize_mm,
        voxelsize_mm,
        seed1z_mm, n_seed_blob=3)

    seeds = data_manipulation.add_seeds_mm(
        seeds, voxelsize_mm,
        [seed1_mm[0]],
        [seed1_mm[1]],
        [seed1_mm[2]],
        label=1,
        radius=25,
        width=1
    )
    # import sed3
    # sed3.show_slices(data3dr, dl > 40.0, slice_step=10)
    if return_likelihood_difference:
        return seeds, dl
    else:
        return seeds


def add_negative_notrain_seeds(seeds,lik1, lik2, alpha=1.3):
    """

    :param seeds:
    :param lik1:
    :param lik2:
    :param alpha: 1.2 means 20% to liver likelihood
    :return:
    """
    # dl = 2*lik2 - lik1
    dl = (alpha*lik1-lik2)>0
    # for sure we take two iterations from segmentation
    # dl[0,:,:] = True
    # dl[:,0,:] = True
    # dl[:,:,0] = True
    # dl[-1,:,:] = True
    # dl[:,-1,:] = True
    # dl[:,:,-1] = True
    dl = scipy.ndimage.morphology.binary_erosion(dl, iterations=2, border_value=1)

    # and now i will take just thin surface
    dl_before=dl
    dl = scipy.ndimage.morphology.binary_erosion(dl, iterations=2, border_value=1)
    dl = imtools.qmisc.resize_to_shape((dl_before - dl) > 0, seeds.shape)
    seeds[dl>0] = 4

    return seeds

def add_negative_train_seeds_blobs(
        mask_working,
        seeds,
        mask_voxelsize_mm,
        seeds_voxelsize_mm,
        seed1z_mm, n_seed_blob=1):
    dll = mask_working

    # aby se pocitalo i od okraju obrazku
    dll[0,:,:] = 0
    dll[:,0,:] = 0
    dll[:,:,0] = 0
    dll[-1,:,:] = 0
    dll[:,-1,:] = 0
    dll[:,:,-1] = 0
    for i in range(0, n_seed_blob):

        dst = scipy.ndimage.morphology.distance_transform_edt(dll)
        # na nasem rezu
        seed1z_px_mask = int(seed1z_mm/ mask_voxelsize_mm[0])
        seed1z_px_seeds= int(seed1z_mm/ seeds_voxelsize_mm[0])

        dstslice = dst[seed1z_px_mask, :, :]
        seed2xy = np.unravel_index(np.argmax(dstslice), dstslice.shape)
        # import PyQt4; PyQt4.QtCore.pyqtRemoveInputHook()
        # import ipdb; ipdb.set_trace()
        seed2 = np.array([seed1z_px_mask, seed2xy[0], seed2xy[1]])
        seed2_mm = seed2 * mask_voxelsize_mm

        seeds = data_manipulation.add_seeds_mm(
                seeds, seeds_voxelsize_mm,
                [seed2_mm[0]],
                [seed2_mm[1]],
                [seed2_mm[2]],
                label=2,
                radius=20,
                width=1
        )
        # for next iteration add hole where this blob is
        dll[seed1z_px_mask, seed2xy[0], seed2xy[1]] = 0

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