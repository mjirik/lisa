#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Simple program for ITK image read/write in Python
#import itk

import SimpleITK as sitk

import numpy as np

import logging
logger = logging.getLogger(__name__)


import dicom
#from sys import argv


class DataWriter:
    def Write3DData(self, data3d, path, filetype='dcm', metadata=None):
        mtd = {'voxelsize_mm': [1, 1, 1]}
        if metadata != None:
            mtd.update(metadata)


        if filetype in ['dcm', 'DCM', 'dicom']:
            #pixelType = itk.UC
            #imageType = itk.Image[pixelType, 2]
            dim = sitk.GetImageFromArray(data3d)
            vsz = mtd['voxelsize_mm']
            dim.SetSpacing([vsz[0], vsz[2], vsz[1]])
            sitk.WriteImage(dim, path)

            #data = dicom.read_file(onefile)


    def write_dicom_slice_with_overlay(
        self, filename, data3d, ovelay, i_overlay):
        pass


    def add_overlay_to_slice_file(
        self,
        filename,
        overlay,
        i_overlay,
        filename_out=None
    ):
        """ Function adds overlay to existing file.
        """
        if filename_out == None:
            filename_out = filename
        data = dicom.read_file(filename)
        data = self.encode_overlay_slice(data, overlay, 0)
        data.save_as(filename_out)
        pass


    def encode_overlay_slice(self, data, overlay, i_overlay):
        """
        """
        # overlay index
        n_bits = 8


        # On (60xx,3000) are stored ovelays.
        # First is (6000,3000), second (6002,3000), third (6004,3000),
        # and so on.
        dicom_tag1 = 0x6000 + 2*i_overlay


        # On (60xx,0010) and (60xx,0011) is stored overlay size
        data[dicom_tag1,0x0010].value = overlay.shape[0]# rows = 512
        data[dicom_tag1,0x0011].value = overlay.shape[1]# cols = 512

        print overlay.shape
        overlay_linear = np.reshape(overlay,np.prod(overlay.shape))

        encoded_linear = np.zeros(np.prod(overlay.shape)/n_bits)
        encoded_linear[10:15] = 255

        overlay_raw = encoded_linear.tostring()


        # Decoding data. Each bit is stored as array element
# TODO neni tady ta jednička blbě?
#        for i in range(1,len(overlay_raw)):
#            for k in range (0,n_bits):
#                byte_as_int = ord(overlay_raw[i])
#                decoded_linear[i*n_bits + k] = (byte_as_int >> k) & 0b1
#
        #overlay = np.array(pol)

        data[dicom_tag1 ,0x3000].value = overlay_raw
        return data
#data = np.zeros([100,100,30], dtype=np.uint8)
#data[20:60,60:70, 0:5] = 100
#dw = DataWriter()
#dw.Write3DData(data, 'soubor.dcm')
