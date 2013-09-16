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


import dcmreaddata1 as dcmr
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





if __name__ == "__main__":
    unittest.main()
