#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
import copy

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest



import numpy as np


#
import datawriter as dwriter
import datareader as dreader

class DicomWriterTest(unittest.TestCase):
    interactivetTest = False
#    def setUp(self):
#        #self.dcmdir = os.path.join(path_to_script, '../sample_data/jatra_06mm_jenjatraplus/')
#        self.dcmdir = os.path.join(path_to_script, '../sample_data/jatra_5mm')
#        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
#        reader = dcmr.DicomReader(self.dcmdir)
#        self.data3d = reader.get_3Ddata()
#        self.metadata = reader.get_metaData()

    def test_write_and_read(self):

        filename = 'test_file.dcm'
        data = (np.random.random([30,100,120])*30).astype(np.int16)
        data[0:5,20:60,60:70,] += 30
        metadata = {'voxelsize_mm': [1,2,3]}
        dw = dwriter.DataWriter()
        dw.Write3DData(data, filename, filetype='dcm', metadata=metadata)

        dr = dreader.DataReader()
        newdata, newmetadata = dr.Get3DData(filename)

        # hack with -1024, because of wrong data reading
        self. assertEqual(data[10,10,10], newdata[10,10,10])
        self. assertEqual(data[2,10,1], newdata[2,10,1])
        self. assertEqual(metadata['voxelsize_mm'][0], newmetadata['voxelsize_mm'][0])
# @TODO there is a bug in SimpleITK. slice voxel size must be same
        #self. assertEqual(metadata['voxelsize_mm'][1], newmetadata['voxelsize_mm'][1])
        self. assertEqual(metadata['voxelsize_mm'][2], newmetadata['voxelsize_mm'][2])


if __name__ == "__main__":
    unittest.main()
