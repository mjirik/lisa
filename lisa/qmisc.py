#! /usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/sed3"))

import numpy as np

import logging
logger = logging.getLogger(__name__)

import subprocess
import scipy
import scipy.ndimage

from imma.image_manipulation import crinfo_from_specific_data, crop, uncrop, manualcrop, fix_crinfo, combinecrinfo


class SparseMatrix():
    def __init__(self, ndarray):
        self.coordinates = ndarray.nonzero()
        self.shape = ndarray.shape
        self.values = ndarray[self.coordinates]
        self.dtype = ndarray.dtype
        self.sparse = True

    def todense(self):
        dense = np.zeros(self.shape, dtype=self.dtype)
        dense[self.coordinates[:]] = self.values
        return dense


def isSparseMatrix(obj):
    if obj.__class__.__name__ == 'SparseMatrix':
        return True
    else:
        return False


# import sed3

# def manualcrop(data):  # pragma: no cover
#
#     try:
#         from imcut import seed_editor_qt
#     except:
#         logger.warning("Deprecated of pyseg_base as submodule")
#         import seed_editor_qt
#
#     pyed = seed_editor_qt.QTSeedEditor(data, mode='crop')
#     pyed.exec_()
#     # pyed = sed3.sed3(data)
#     # pyed.show()
#     nzs = pyed.seeds.nonzero()
#     crinfo = [
#         [np.min(nzs[0]), np.max(nzs[0])],
#         [np.min(nzs[1]), np.max(nzs[1])],
#         [np.min(nzs[2]), np.max(nzs[2])],
#         ]
#     data = crop(data, crinfo)
#     return data, crinfo
#
#
# def crop(data, crinfo):
#     """
#     Crop the data.
#
#     crop(data, crinfo)
#
#     :param crinfo: min and max for each axis
#
#     """
#     crinfo = fix_crinfo(crinfo)
#     return data[
#         __int_or_none(crinfo[0][0]):__int_or_none(crinfo[0][1]),
#         __int_or_none(crinfo[1][0]):__int_or_none(crinfo[1][1]),
#         __int_or_none(crinfo[2][0]):__int_or_none(crinfo[2][1])
#         ]
#
#
# def __int_or_none(number):
#     if number is not None:
#         number = int(number)
#     return number


# def combinecrinfo(crinfo1, crinfo2):
#     """
#     Combine two crinfos. First used is crinfo1, second used is crinfo2.
#     """
#     crinfo1 = fix_crinfo(crinfo1)
#     crinfo2 = fix_crinfo(crinfo2)
#
#     crinfo = [
#         [crinfo1[0][0] + crinfo2[0][0], crinfo1[0][0] + crinfo2[0][1]],
#         [crinfo1[1][0] + crinfo2[1][0], crinfo1[1][0] + crinfo2[1][1]],
#         [crinfo1[2][0] + crinfo2[2][0], crinfo1[2][0] + crinfo2[2][1]]
#         ]
#
#     return crinfo
#
#
# def crinfo_from_specific_data(data, margin):
#     # hledáme automatický ořez, nonzero dá indexy
#     logger.debug('crinfo')
#     logger.debug(str(margin))
#     nzi = np.nonzero(data)
#     logger.debug(str(nzi))
#
#     x1 = np.min(nzi[0]) - margin[0]
#     x2 = np.max(nzi[0]) + margin[0] + 1
#     y1 = np.min(nzi[1]) - margin[0]
#     y2 = np.max(nzi[1]) + margin[0] + 1
#     z1 = np.min(nzi[2]) - margin[0]
#     z2 = np.max(nzi[2]) + margin[0] + 1
#
# # ošetření mezí polí
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if z1 < 0:
#         z1 = 0
#
#     if x2 > data.shape[0]:
#         x2 = data.shape[0] - 1
#     if y2 > data.shape[1]:
#         y2 = data.shape[1] - 1
#     if z2 > data.shape[2]:
#         z2 = data.shape[2] - 1
#
# # ořez
#     crinfo = [[x1, x2], [y1, y2], [z1, z2]]
#     return crinfo
#
#
# def uncrop(data, crinfo, orig_shape, resize=False):
#
#     crinfo = fix_crinfo(crinfo)
#     data_out = np.zeros(orig_shape, dtype=data.dtype)
#
#     # print 'uncrop ', crinfo
#     # print orig_shape
#     # print data.shape
#     if resize:
#         data = resize_to_shape(data, crinfo[:, 1] - crinfo[:, 0])
#
#     startx = np.round(crinfo[0][0]).astype(int)
#     starty = np.round(crinfo[1][0]).astype(int)
#     startz = np.round(crinfo[2][0]).astype(int)
#
#     data_out[
#         # np.round(crinfo[0][0]).astype(int):np.round(crinfo[0][1]).astype(int)+1,
#         # np.round(crinfo[1][0]).astype(int):np.round(crinfo[1][1]).astype(int)+1,
#         # np.round(crinfo[2][0]).astype(int):np.round(crinfo[2][1]).astype(int)+1
#         startx:startx + data.shape[0],
#         starty:starty + data.shape[1],
#         startz:startz + data.shape[2]
#         ] = data
#
#     return data_out
#
#
# def fix_crinfo(crinfo, to='axis'):
#     """
#     Function recognize order of crinfo and convert it to proper format.
#     """
#
#     crinfo = np.asarray(crinfo)
#     if crinfo.shape[0] == 2:
#         crinfo = crinfo.T
#
#     return crinfo


# def getVersionString():
#     """
#     Function return string with version information.
#     It is performed by use one of three procedures: git describe,
#     file in .git dir and file __VERSION__.
#     """
#     version_string = None
#     try:
#         version_string = subprocess.check_output(['git', 'describe'])
#         if sys.version_info.major == 3:
#             version_string = version_string.decode("utf8")
#     except:
#         logger.warning('Command "git describe" is not working')
#
#     if version_string == None:  # noqa
#         try:
#             path_to_version = os.path.join(path_to_script,
#                                            '../.git/refs/heads/master')
#             with file(path_to_version) as f:
#                 version_string = f.read()
#         except:
#             logger.warning('Problem with reading file ".git/refs/heads/master"')
#
#     if version_string == None:  # noqa
#         try:
#             path_to_version = os.path.join(path_to_script, '../__VERSION__')
#             with file(path_to_version) as f:
#                 version_string = f.read()
#             path_to_version = path_to_version + \
#                 '  version number is created manually'
#
#         except:
#             logger.warning('Problem with reading file "__VERSION__"')
#
#     # if version_string is None:
#     #     version_string = ""
#     return version_string


def get_one_biggest_object(data):
    """ Return biggest object """
    lab, num = scipy.ndimage.label(data)
    # print("bum = "+str(num))

    maxlab = max_area_index(lab, num)

    data = (lab == maxlab)
    return data


def max_area_index(labels, num):
    """
    Return index of maxmum labeled area
    """
    mx = 0
    mxi = -1
    for l in range(1, num + 1):
        mxtmp = np.sum(labels == l)
        if mxtmp > mx:
            mx = mxtmp
            mxi = l

    return mxi

from io3d.misc import resize_to_mm, resize_to_shape


# def resize_to_shape(*pars, **params):
#     import io3d.misc
#     return io3d.misc.resize_to_shape(*pars, **params)
