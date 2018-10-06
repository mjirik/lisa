#! /opt/local/bin/python
# -*- coding: utf-8 -*-
"""
Show dicom data with overlay
"""

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/sed3/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector

import logging
logger = logging.getLogger(__name__)


#import apdb
#  apdb.set_trace();
#import scipy.io
import numpy as np
#import scipy
#from scipy import sparse
#import traceback
#import time
#import audiosupport

# ----------------- my scripts --------
import argparse
import sed3

#import segmentation
import qmisc
import misc
from io3d import datareader
#import config
#import numbers


def saveOverlayToDicomCopy(input_dcmfilelist, output_dicom_dir, overlays,
                           crinfo, orig_shape):
    """ Save overlay to dicom. """
    import datawriter as dwriter

    if not os.path.exists(output_dicom_dir):
        os.mkdir(output_dicom_dir)

    # uncrop all overlays
    for key in overlays:
        overlays[key] = qmisc.uncrop(overlays[key], crinfo, orig_shape)

    dw = dwriter.DataWriter()
    dw.DataCopyWithOverlay(input_dcmfilelist, output_dicom_dir, overlays)


def readData3d(dcmdir):
    qt_app = None
    if dcmdir is None:
        from PyQt4 import QtGui
#QApplication
        qt_app = QtGui.QApplication(sys.argv)
# same as  data_reader_get_dcmdir_qt
#        from PyQt4.QtGui import QFileDialog, QApplication
#        dcmdir = QFileDialog.getExistingDirectory(
#                caption='Select DICOM Folder',
#                options=QFileDialog.ShowDirsOnly)
#        dcmdir = "%s" %(dcmdir)
#        dcmdir = dcmdir.encode("utf8")
        dcmdir = datareader.get_dcmdir_qt(qt_app)
    reader = datareader.DataReader()
    data3d, metadata = reader.Get3DData(dcmdir, qt_app=None, dataplus_format=False)

    return data3d, metadata, qt_app


def save_config(cfg, filename):
    cfg
    misc.obj_to_file(cfg, filename, filetype="yaml")


def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

## read confguraton from file, use default values from OrganSegmentation

    # input parser
    parser = argparse.ArgumentParser(
        description=' Show dicom data with overlay')
    parser.add_argument('-i', '--inputdatapath',
                        default='',
                        help='path to data dir')
    args_obj = parser.parse_args()
    args = vars(args_obj)
    #print args["arg"]

    reader = datareader.DataReader()
    data3d, metadata = reader.Get3DData(args['inputdatapath'], qt_app=None, dataplus_format=False)
    overlays = reader.get_overlay()
    overlay = np.zeros(data3d.shape, dtype=np.int8)
    print("overlays ", list(overlays.keys()))
    for key in overlays:
        overlay += overlays[key]

    #import ipdb; ipdb.set_trace() # BREAKPOINT
    pyed = sed3.sed3(data3d, contour=overlay)
    pyed.show()

    #print savestring
#    import pdb; pdb.set_trace()
    return

if __name__ == "__main__":
    main()
