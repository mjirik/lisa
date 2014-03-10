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
sys.path.append(os.path.join(path_to_script, "../experiments/"))
import unittest

import numpy as np


import dcmreaddata as dcmr
import experiments


class ExperimentsTest(unittest.TestCase):

    def test_get_subdirs(self):
        dirpath = os.path.join(path_to_script, "..")
        dirlist = experiments.get_subdirs(dirpath)
        #import pdb; pdb.set_trace()

        self.assertTrue('tests' in dirlist)
        self.assertTrue('extern' in dirlist)
        self.assertTrue('src' in dirlist)
        self.assertFalse('README.md' in dirlist)

    def test_eval_sliver_volume(self):
        import volumetry_evaluation as ve

        vol1 = np.zeros([20,21,22], dtype=np.int8)
        vol1 [10:15, 10:15, 10:15] = 1

        vol2 = np.zeros([20,21,22], dtype=np.int8)
        vol2 [10:15, 10:16, 10:15] = 1

        eval1 = ve.compare_volumes(vol1, vol2, [1, 1, 1])
        print eval1

        pass



if __name__ == "__main__":
    unittest.main()
