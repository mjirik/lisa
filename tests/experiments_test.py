#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
import copy

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest

import numpy as np


import dcmreaddata1 as dcmr
import experiments


class ExperimentsTest(unittest.TestCase):

    def test_get_subdirs(self):
        dirlist = experiments.get_subdirs(os.path.join(path_to_script, ".."))



        import pdb; pdb.set_trace()
        errorrate = 0

        self.assertLess(errorrate,0.1)




if __name__ == "__main__":
    unittest.main()
