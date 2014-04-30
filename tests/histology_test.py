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


    def test_synthetic_data_vessel_tree_evaluation(self):
        """
        Generovani umeleho stromu do 3D a jeho evaluace.
        V testu dochazi ke kontrole predpokladaneho objemu a deky cev


        """
        import gen_volume_tree
        expected_length = 10
        evaluated_length = 10

# nacteni dat

        #tvg = gen_volume_tree.TreeVolumeGenerator()

        #yaml_path = os.path.join(path_to_script, "./hist_tree_test.yaml")
        #tvg.importFromYaml(yaml_path)


# histology analyser

        #ha = HistologyAnalyser(data3d, metadata, args.threshold)


# histology report
        #hr = HistologyReport()


        self.assertGreater(evaluated_length, expected_length * 0.9  )
        self.assertLess(evaluated_length, expected_length * 1.1 )




if __name__ == "__main__":
    unittest.main()
