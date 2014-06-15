#! /usr/bin/python
# -*- coding: utf-8 -*-
""" Module for readin 3D dicom data
"""

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/py3DSeedEditor/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector

import logging
logger = logging.getLogger(__name__)

# -------------------- my scripts ------------

import dcmreaddata as dcmr

import SimpleITK as sitk
import numpy as np

class DataReader:

    def __init__(self):

        self.overlay_fcn = None

    def Get3DData(self, datapath, qt_app=None,
                  dataplus_format=False, gui=False,
                  start=0, stop=None, step=1):
        """
        :datapath directory with input data
        :qt_app if it is set to None (as default) all dialogs for series
        selection are performed in terminal. If qt_app is set to
        QtGui.QApplication() dialogs are in Qt.

        :dataplus_format is new data format. Metadata and data are returned in
        one structure.
        """

        if qt_app is None and gui is True:
            from PyQt4.QtGui import QApplication
            qt_app = QApplication(sys.argv)

        datapath = os.path.normpath(datapath)
        if os.path.isfile(datapath):
            path, ext = os.path.splitext(datapath)
            if ext in ('.pklz', '.pkl'):
                import misc
                data = misc.obj_from_file(datapath, filetype='pkl')
                data3d = data.pop('data3d')
                #metadata must have series_number
                metadata = {
                    'series_number': 0,
                    'datadir': datapath
                }
                metadata.update(data)

            else:
# reading raw file
                image = sitk.ReadImage(datapath)
                #image = sitk.ReadImage('/home/mjirik/data/medical/data_orig/sliver07/01/liver-orig001.mhd') #noqa
                #sz = image.GetSize()

                #data3d = sitk.Image(sz[0],sz[1],sz[2], sitk.sitkInt16)

                #for i in range(0,sz[0]):
                #    print i
                #    for j in range(0,sz[1]):
                #        for k in range(0,sz[2]):
                #            data3d[i,j,k]=image[i,j,k]

                data3d = sitk.GetArrayFromImage(image)  # + 1024
                #data3d = np.transpose(data3d)
                #data3d = np.rollaxis(data3d,1)
                metadata = {}  # reader.get_metaData()
                metadata['series_number'] = 0  # reader.series_number
                metadata['datadir'] = datapath
                spacing = image.GetSpacing()
                metadata['voxelsize_mm'] = [
                    spacing[2],
                    spacing[0],
                    spacing[1],
                ]

        else:
            # checks if data is in DICOM format
            dir_type = 'images'
            if dcmr.is_dicom_dir(datapath):
                dir_type = 'dicom'


            if dir_type == 'dicom': #reading dicom
                logger.debug('Dir - DICOM')
                reader = dcmr.DicomReader(datapath, qt_app=None, gui=True)
                data3d = reader.get_3Ddata(start, stop, step)
                metadata = reader.get_metaData()
                metadata['series_number'] = reader.series_number
                metadata['datadir'] = datapath
                self.overlay_fcn = reader.get_overlay
            else: # reading image sequence
                logger.debug('Dir - Image sequence')

                logger.debug('Getting list of readable files...')
                flist = []
                for f in os.listdir(datapath):
                    try:
                        sitk.ReadImage(os.path.join(datapath,f))
                    except:
                        logger.warning("Cant load file: "+str(f))
                        continue
                    flist.append(os.path.join(datapath,f))
                flist.sort()

                logger.debug('Reading image data...')
                image = sitk.ReadImage(flist)
                logger.debug('Getting numpy array from image data...')
                data3d = sitk.GetArrayFromImage(image)

                metadata = {}  # reader.get_metaData()
                metadata['series_number'] = 0  # reader.series_number
                metadata['datadir'] = datapath
                spacing = image.GetSpacing()
                metadata['voxelsize_mm'] = [
                    spacing[2],
                    spacing[0],
                    spacing[1],
                ]

        if dataplus_format:
            logger.debug('dataplus format')
            datap = metadata
            datap['data3d'] = data3d
            logger.debug('datap keys () : ' + str(datap.keys()))
            return datap
        else:
            return data3d, metadata

    def GetOverlay(self):
        """ Generates dictionary of ovelays
        """

        if self.overlay_fcn == None:  # noqa
            return {}
        else:
            return self.overlay_fcn()


def get_datapath_qt(qt_app):
# just call function from dcmr
    return dcmr.get_datapath_qt(qt_app)
