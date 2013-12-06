#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest



import numpy as np
import os


import qmisc
import misc


#

class QmiscTest(unittest.TestCase):
    interactivetTest = False
    #interactivetTest = True


    def test_store_to_SparseMatrix_and_back(self):
        data = np.zeros([4,4,4])
        data = np.zeros([4,4,4])
        data[1,0,3] = 1
        data[2,1,2] = 1
        data[0,1,3] = 2
        data[1,2,0] = 1
        data[2,1,1] = 3

        dataSM = qmisc.SparseMatrix(data)

        data2 = dataSM.todense()
        self.assertTrue(np.all(data==data2))

    def test_obj_to_and_from_file_yaml(self):
        testdata = np.random.random([4,4,3])
        test_object = {'a':1, 'data': testdata}

        filename = 'test_obj_to_and_from_file.yaml'
        misc.obj_to_file(test_object, filename ,'yaml')
        saved_object = misc.obj_from_file(filename,'yaml')

        self.assertTrue(saved_object['a'] == 1)
        self.assertTrue(saved_object['data'][1,1,1] == testdata[1,1,1])

        os.remove(filename)

    def test_obj_to_and_from_file_pickle(self):
        testdata = np.random.random([4,4,3])
        test_object = {'a':1, 'data': testdata}

        filename = 'test_obj_to_and_from_file.pkl'
        misc.obj_to_file(test_object, filename ,'pickle')
        saved_object = misc.obj_from_file(filename,'pickle')

        self.assertTrue(saved_object['a'] == 1)
        self.assertTrue(saved_object['data'][1,1,1] == testdata[1,1,1])

        os.remove(filename)

    #def test_obj_to_and_from_file_exeption(self):
    #    test_object = [1]
    #    filename = 'test_obj_to_and_from_file_exeption'
    #    self.assertRaises(misc.obj_to_file(test_object, filename ,'yaml'))


    def test_crop_and_uncrop(self):
        shape = [10,10,5]
        img_in = np.random.random(shape)

        crinfo = [[2,8],[3,9],[2,5]]


        img_cropped = qmisc.crop(img_in, crinfo)


        img_uncropped = qmisc.uncrop(img_cropped, crinfo, shape)


        self.assertTrue(img_uncropped[4,4,3] == img_in[4,4,3])



    def test_multiple_crop_and_uncrop(self):
        """
        test combination of multiple crop
        """

        shape = [10,10,5]
        img_in = np.random.random(shape)

        crinfo1 = [[2,8],[3,9],[2,5]]
        crinfo2 = [[2,5],[1,4],[1,2]]

        img_cropped = qmisc.crop(img_in, crinfo1)
        img_cropped = qmisc.crop(img_cropped, crinfo2)

        crinfo_combined = qmisc.combinecrinfo(crinfo1, crinfo2)



        img_uncropped = qmisc.uncrop(img_cropped, crinfo_combined, shape)


        self.assertTrue(img_uncropped[4,4,3] == img_in[4,4,3])

    @unittest.skip("waiting for implementation")
    def test_suggest_filename(self):
        import misc
        filename = "mujsoubor"
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(filename == new_filename)

        filename = "mujsoubor-1"
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(new_filename == "mujsoubor-2")

        filename = "mujsoubor-1"
        new_filename = misc.suggest_filename(filename, exists=True)
        self.assertTrue(new_filename == "mujsoubor-3")

    def test_getVersionString(self):
        """
        """
        verstr = qmisc.getVersionString()

        self.assertTrue(type(verstr) == str)


if __name__ == "__main__":
    unittest.main()
