import logging
logger = logging.getLogger(__name__)
import logging.handlers

import sys
import os
import os.path as op
# from collections import namedtuple

# from scipy.io import loadmat, savemat
import scipy
import scipy.ndimage
import numpy as np

import imma.image_manipulation as ima

def segmentation_replacement(
        segmentation,
        segmentation_new,
        label,
        background_label=0,
        slab=None,
        label_new=1
):
    """
    Remove label from segmentation and put there a new one.

    :param segmentation:
    :param segmentation_new:
    :param label:
    :param background_label:
    :param slab:
    :param label_new:
    :return:
    """
    segmentation_old = ima.select_labels(segmentation, labels=label, slab=slab)
    segmentation[segmentation_old] = ima.get_nlabels(slab, background_label)
    segmentation_new = ima.select_labels(segmentation_new, label_new)
    segmentation[segmentation_new] = ima.get_nlabels(slab, label)
    return segmentation



def segmentation_smoothing(segmentation, sigma_mm, labels=1, background_label=0,
                   voxelsize_mm=None,
                   volume_blowup=1.,
                   slab=None,
                   volume_blowup_criterial_function=None,
                   ):
    """
    Shape of output segmentation is smoothed with gaussian filter.

    Sigma is computed in mm

    """
    # import scipy.ndimage
    if voxelsize_mm is None:
        voxelsize_mm = np.asarray([1., 1., 1.])

    if volume_blowup_criterial_function is None:
        volume_blowup_criterial_function = __volume_blowup_criterial_function
    sigma = float(sigma_mm) / np.array(voxelsize_mm)

    # print sigma
    # from PyQt4.QtCore import pyqtRemoveInputHook
    # pyqtRemoveInputHook()
    segmentation_selection = ima.select_labels(segmentation, labels=labels, slab=slab)
    vol1 = np.sum(segmentation_selection)
    wvol = vol1 * volume_blowup
    logger.debug('unique segm ' + str(np.unique(segmentation)))
    segsmooth = scipy.ndimage.filters.gaussian_filter(
        segmentation_selection.astype(np.float32), sigma)
    logger.debug('unique segsmooth ' + str(np.unique(segsmooth)))
    # import ipdb; ipdb.set_trace()
    # import pdb; pdb.set_trace()
    # pyed = sed3.sed3(self.orig_scale_segmentation)
    # pyed.show()
    logger.debug('wanted volume ' + str(wvol))
    logger.debug('sigma ' + str(sigma))

    critf = lambda x: volume_blowup_criterial_function(
        x, wvol, segsmooth)

    thr = scipy.optimize.fmin(critf, x0=0.5, disp=False)[0]
    logger.debug('optimal threshold ' + str(thr))
    logger.debug('segsmooth ' + str(np.nonzero(segsmooth)))

    segmentation_selection = (1.0 *
                              (segsmooth > thr)  # self.volume_blowup)
                              ).astype(np.int8)
    vol2 = np.sum(segmentation_selection)
    with np.errstate(divide="ignore", invalid="ignore"):
        logger.debug("volume ratio " + str(vol2 / float(vol1)))
    # import ipdb; ipdb.set_trace()
    segmentation_replacement(
        segmentation,
        label=labels,
        segmentation_new=segmentation_selection,
        background_label=background_label,
        slab=slab,
    )

def __volume_blowup_criterial_function(threshold, wanted_volume,
                                       segmentation_smooth
                                       ):

    segm = (1.0 * segmentation_smooth > threshold).astype(np.int8)
    vol2 = np.sum(segm)
    criterium = (wanted_volume - vol2) ** 2
    return criterium

