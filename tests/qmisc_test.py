#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest



import numpy as np


import qmisc


#import dcmreaddata1 as dcmr

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

if __name__ == "__main__":
    unittest.main()
