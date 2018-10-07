#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import os.path
import os

path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest

import shutil


import numpy as np
from nose.plugins.attrib import attr

# import pydicom
# pydicom.debug(False)

#
import io3d.datawriter as dwriter
import io3d.datareader as dreader
import lisa.dataset
# import sed3 as pyed


class DicomWriterTest(unittest.TestCase):
    interactivetTest = False
#    def setUp(self):
# self.dcmdir = os.path.join(path_to_script,
# '../sample_data/jatra_06mm_jenjatraplus/')
#        self.dcmdir = os.path.join(path_to_script, '../sample_data/jatra_5mm')
# self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
#        reader = dcmr.DicomReader(self.dcmdir)
#        self.data3d = reader.get_3Ddata()
#        self.metadata = reader.get_metaData()

    def test_write_and_read(self):

        filename = 'test_file.dcm'
        data = (np.random.random([30, 100, 120]) * 30).astype(np.int16)
        data[0:5, 20:60, 60:70, ] += 30
        metadata = {'voxelsize_mm': [1, 2, 3]}
        dw = dwriter.DataWriter()
        dw.Write3DData(data, filename, filetype='dcm', metadata=metadata)

        dr = dreader.DataReader()
        newdata, newmetadata = dr.Get3DData(filename, dataplus_format=False)

        # print  "meta ", metadata
        # print  "new meta ", newmetadata
        # hack with -1024, because of wrong data reading
        self. assertEqual(data[10, 10, 10], newdata[10, 10, 10])
        self. assertEqual(data[2, 10, 1], newdata[2, 10, 1])
        self. assertEqual(
            metadata['voxelsize_mm'][0], newmetadata['voxelsize_mm'][0])
# @TODO there is a bug in SimpleITK. slice voxel size must be same
        # self. assertEqual(metadata['voxelsize_mm'][1],
        # newmetadata['voxelsize_mm'][1])
        self. assertEqual(
            metadata['voxelsize_mm'][2], newmetadata['voxelsize_mm'][2])
        os.remove(filename)

    def test_add_overlay_and_read_one_file_with_overlay(self):
        filename = 'tests_outputs/test_file.dcm'
        filedir = os.path.dirname(filename)

        # number of tested overlay
        i_overlay = 6

        if not os.path.exists('tests_outputs'):
            os.mkdir('tests_outputs')

        data = (np.random.random([30, 100, 120]) * 30).astype(np.int16)
        data[0:5, 20:60, 60:70] += 30
        overlay = np.zeros([512, 512], dtype=np.uint8)
        overlay[450:500, 30:100] = 1

        # metadata = {'voxelsize_mm': [1, 2, 3]}
        dw = dwriter.DataWriter()

        dw.add_overlay_to_slice_file(
            # 'sample_data/jatra_5mm/IM-0001-0019.dcm',
            lisa.dataset.join_sdp('volumetrie/volumetry_slice.DCM'),
            overlay,
            i_overlay,
            filename
        )
        dr = dreader.DataReader()
        newdata, newmetadata = dr.Get3DData('tests_outputs', dataplus_format=False)
        newoverlay = dr.get_overlay()
        # print overlay

        # ed = pyed.sed3(newoverlay[6])
        # ed.show()
        self.assertTrue((newoverlay[i_overlay] == overlay).all())

        # os.remove(filename)
        shutil.rmtree(filedir)

    @attr("slow")
    def test_add_overlay_to_copied_dir(self):
        """
        writes 3d label to copied dicom files
        """
        filedir = 'test_outputs_dir'
        n_files = 3

        # number of tested overlay
        i_overlay = 6

        if not os.path.exists(filedir):
            os.mkdir(filedir)

# open copied data to obtain dcmfilefilelist
        dr = dreader.DataReader()
        data3d, metadata = dr.Get3DData(
            lisa.dataset.sample_data_path() + '/jatra_5mm/', dataplus_format=False
            # 'sample_data/volumetrie/'
        )
# for test we are working only with small number of files (n_files)
        metadata['dcmfilelist'] = metadata['dcmfilelist'][:n_files]

# create overlay
        overlay = np.zeros([n_files, 512, 512], dtype=np.uint8)
        overlay[:, 450:500, 30:100] = 1
# if there is more slides, try more complicated overlay
        overlay[0, 430:460, 20:110] = 1
        overlay[-1:, 470:520, 10:120] = 1

        overlays = {i_overlay: overlay}

        dw = dwriter.DataWriter()
        dw.DataCopyWithOverlay(metadata['dcmfilelist'], filedir, overlays)

# try read written data
        dr = dreader.DataReader()
        newdata, newmetadata = dr.Get3DData(filedir, dataplus_format=False)
        newoverlay = dr.get_overlay()

        self.assertTrue((newoverlay[i_overlay] == overlays[i_overlay]).all())

        # os.remove(filename)
        shutil.rmtree(filedir)
if __name__ == "__main__":
    unittest.main()
