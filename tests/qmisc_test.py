#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest


import numpy as np
import os


from lisa import qmisc
from lisa import misc


#

class QmiscTest(unittest.TestCase):
    interactivetTest = False
    # interactivetTest = True

    def test_store_to_SparseMatrix_and_back(self):
        data = np.zeros([4, 4, 4])
        data = np.zeros([4, 4, 4])
        data[1, 0, 3] = 1
        data[2, 1, 2] = 1
        data[0, 1, 3] = 2
        data[1, 2, 0] = 1
        data[2, 1, 1] = 3

        dataSM = qmisc.SparseMatrix(data)

        data2 = dataSM.todense()
        self.assertTrue(np.all(data == data2))

    def test_obj_to_and_from_file_yaml(self):
        testdata = np.random.random([4, 4, 3])
        test_object = {'a': 1, 'data': testdata}

        filename = 'test_obj_to_and_from_file.yaml'
        misc.obj_to_file(test_object, filename, 'yaml')
        saved_object = misc.obj_from_file(filename, 'yaml')

        self.assertTrue(saved_object['a'] == 1)
        self.assertTrue(saved_object['data'][1, 1, 1] == testdata[1, 1, 1])

        os.remove(filename)

    def test_obj_to_and_from_file_pickle(self):
        testdata = np.random.random([4, 4, 3])
        test_object = {'a': 1, 'data': testdata}

        filename = 'test_obj_to_and_from_file.pkl'
        misc.obj_to_file(test_object, filename, 'pickle')
        saved_object = misc.obj_from_file(filename, 'pickle')

        self.assertTrue(saved_object['a'] == 1)
        self.assertTrue(saved_object['data'][1, 1, 1] == testdata[1, 1, 1])

        os.remove(filename)

    # def test_obj_to_and_from_file_exeption(self):
    #    test_object = [1]
    #    filename = 'test_obj_to_and_from_file_exeption'
    #    self.assertRaises(misc.obj_to_file(test_object, filename ,'yaml'))

    def test_obj_to_and_from_file_with_directories(self):
        import shutil
        testdata = np.random.random([4, 4, 3])
        test_object = {'a': 1, 'data': testdata}

        dirname = '__test_write_and_read'
        filename = '__test_write_and_read/test_obj_to_and_from_file.pkl'

        misc.obj_to_file(test_object, filename, 'pickle')
        saved_object = misc.obj_from_file(filename, 'pickle')

        self.assertTrue(saved_object['a'] == 1)
        self.assertTrue(saved_object['data'][1, 1, 1] == testdata[1, 1, 1])

        shutil.rmtree(dirname)

    def test_crop_and_uncrop(self):
        shape = [10, 10, 5]
        img_in = np.random.random(shape)

        crinfo = [[2, 8], [3, 9], [2, 5]]

        img_cropped = qmisc.crop(img_in, crinfo)

        img_uncropped = qmisc.uncrop(img_cropped, crinfo, shape)

        self.assertTrue(img_uncropped[4, 4, 3] == img_in[4, 4, 3])

    def test_multiple_crop_and_uncrop(self):
        """
        test combination of multiple crop
        """

        shape = [10, 10, 5]
        img_in = np.random.random(shape)

        crinfo1 = [[2, 8], [3, 9], [2, 5]]
        crinfo2 = [[2, 5], [1, 4], [1, 2]]

        img_cropped = qmisc.crop(img_in, crinfo1)
        img_cropped = qmisc.crop(img_cropped, crinfo2)

        crinfo_combined = qmisc.combinecrinfo(crinfo1, crinfo2)

        img_uncropped = qmisc.uncrop(img_cropped, crinfo_combined, shape)

        self.assertTrue(img_uncropped[4, 4, 3] == img_in[4, 4, 3])

    # @unittest.skip("waiting for implementation")
    def test_suggest_filename(self):
        """
        Testing some files. Not testing recursion in filenames. It is situation
        if there exist file0, file1, file2 and input file is file
        """
        filename = "mujsoubor"
        # import ipdb; ipdb.set_trace() # BREAKPOINT
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(new_filename == "mujsoubor2")

        filename = "mujsoubor112"
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(new_filename == "mujsoubor113")

        filename = "mujsoubor-2.txt"
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(new_filename == "mujsoubor-3.txt")

        filename = "mujsoubor-a24.txt"
        new_filename = misc.suggest_filename(filename, exists=False)
        self.assertTrue(new_filename == "mujsoubor-a24.txt")

    def test_getVersionString(self):
        """
        """
        verstr = qmisc.getVersionString()

        self.assertTrue(type(verstr) == str)

    def test_resize_to_shape(self):

        data = np.random.rand(3, 4, 5)
        new_shape = [5, 6, 6]
        data_out = qmisc.resize_to_shape(data, new_shape)
        # print data_out.shape
        # print data
        # print data_out
        self.assertItemsEqual(new_shape, data_out.shape)

    def test_fix_crinfo(self):
        crinfo = [[10, 15], [30, 40], [1, 50]]
        cri_fixed = qmisc.fix_crinfo(crinfo)

        # print crinfo
        # print cri_fixed

        self.assertTrue(cri_fixed[1, 1] == 40)
        self.assertTrue(cri_fixed[2, 1] == 50)

    def test_resize_to_mm(self):

        data = np.random.rand(3, 4, 5)
        voxelsize_mm = [2, 3, 1]
        new_voxelsize_mm = [1, 3, 2]
        expected_shape = [6, 4, 3]
        data_out = qmisc.resize_to_mm(data, voxelsize_mm, new_voxelsize_mm)
        print data_out.shape
        # print data
        # print data_out
        self.assertItemsEqual(expected_shape, data_out.shape)

if __name__ == "__main__":
    unittest.main()
