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


import dcmreaddata as dcmr

class HistologyTest(unittest.TestCase):

    # @TODO znovu zprovoznit test
    @unittest.skip("Cekame, az to Tomas opravi")

    def test_synthetic_data_vessel_tree_evaluation(self):
        """
        Generovani umeleho stromu do 3D a jeho evaluace.
        V testu dochazi ke kontrole predpokladaneho objemu a deky cev


        """
        import lesions
        expected_length = 10
        evaluated_length = 10

        data = {'data':{1:{'X', 'Y'}}}



        self.assertGreater(evaluated_length, expected_length)
        self.assertLess(evaluated_length, expected_length)




if __name__ == "__main__":
    unittest.main()
