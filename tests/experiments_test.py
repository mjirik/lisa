#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path
#import copy

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
sys.path.append(os.path.join(path_to_script, "../experiments/"))
import unittest

import logging
logger = logging.getLogger(__name__)

import numpy as np


#import dcmreaddata as dcmr
import experiments
import volumetry_evaluation


class ExperimentsTest(unittest.TestCase):

    def test_get_subdirs(self):
        dirpath = os.path.join(path_to_script, "..")
        dirlist = experiments.get_subdirs(dirpath)
        #import pdb; pdb.set_trace()

        self.assertTrue('tests' in dirlist)
        self.assertTrue('extern' in dirlist)
        self.assertTrue('src' in dirlist)
        self.assertFalse('README.md' in dirlist)

    def atest_eval_sliver_matrics(self):
        import volumetry_evaluation as ve

        vol1 = np.zeros([20, 21, 22], dtype=np.int8)
        vol1[10:15, 10:15, 10:15] = 1

        vol2 = np.zeros([20, 21, 22], dtype=np.int8)
        vol2[10:15, 10:16, 10:15] = 1

        eval1 = ve.compare_volumes(vol1, vol2, [1, 1, 1])
        print eval1

    def test_eval_sliver_distance_for_two_pixels_bigger_volume(self):
        """
        maxd is measured in corner. It is  space diagonal of 2 pixels cube.
        """

        import volumetry_evaluation as ve

        vol1 = np.zeros([20, 21, 22], dtype=np.int8)
        vol1[10:15, 10:15, 10:15] = 1

        vol2 = np.zeros([20, 21, 22], dtype=np.int8)
        vol2[8:17, 8:17, 8:17] = 1

        eval1 = ve.distance_matrics(vol1, vol2, [1, 1, 1])

        self.assertAlmostEquals(eval1[2], 3 ** (0.5) * 2)
        #print eval1

    def test_eval_sliver_distance_two_points(self):
        """
        Two points 2 by 2 pixels diagonal.
        """
        import volumetry_evaluation as ve

        vol1 = np.zeros([20, 21, 22], dtype=np.int8)
        vol1[10, 10, 10] = 1

        vol2 = np.zeros([20, 21, 22], dtype=np.int8)
        vol2[10, 12, 12] = 1

        eval1 = ve.distance_matrics(vol1, vol2, [1, 1, 1])
        self.assertAlmostEquals(eval1[2], np.sqrt(2) * 2)
        #print eval1

    def test_eval_sliver_distance_two_points_with_half_voxelsize(self):
        """
        Two points 2 by 2 pixels diagonal and voxelsize is 0.5
        """
        import volumetry_evaluation as ve

        vol1 = np.zeros([20, 21, 22], dtype=np.int8)
        vol1[10, 10, 10] = 1

        vol2 = np.zeros([20, 21, 22], dtype=np.int8)
        vol2[10, 12, 12] = 1

        eval1 = ve.distance_matrics(vol1, vol2, [0.5, 0.5, 0.5])
        #print eval1
        self.assertAlmostEquals(eval1[2], np.sqrt(2))

    def test_volumetry_evaluation_yaml_generator(self):
        """
        Test creates two directories and some empty files. Based on this
        structure a yaml file data are constructed.
        """
        import os
        import shutil

# Creating directory structure

        sliver_dir = '__test_sliver_dir'
        pklz_dir = '__test_pklz_dir'

        #d = os.path.dirname(sliver_dir)
        d = sliver_dir
        if os.path.exists(d):
            shutil.rmtree(d)
        #if not os.path.exists(d):
        os.makedirs(d)

        #d = os.path.dirname(pklz_dir)
        d = pklz_dir
        if os.path.exists(d):
            shutil.rmtree(d)
        #if not os.path.exists(d):
        os.makedirs(d)

        filelist1 = ['liver-seg001.mhd', 'liver-seg002.mhd',
                     'liver-seg006.mhd']
        for fl in filelist1:
            open(os.path.join(sliver_dir, fl), 'a').close()

        filelist2 = ['soubor_seg001to.pklz', 'soubor_seg002.pklz',
                     'sao_seg003.pklz', 'so_seg002tre3.pklz', 'ijij.pklz']
        for fl in filelist2:
            open(os.path.join(pklz_dir, fl), 'a').close()

# construct yaml data
        yamldata = volumetry_evaluation.generate_input_yaml(
            sliver_dir, pklz_dir)

# assert

        fls0 = os.path.join(sliver_dir, filelist1[0])
        flp0 = os.path.join(pklz_dir, filelist2[0])
        # we only hope, that this will be first record
        self.assertEqual(yamldata['data'][0]['sliverseg'], fls0)
        self.assertEqual(yamldata['data'][0]['ourseg'], flp0)

# Clean
        #if os.path.exists(d):
        shutil.rmtree(sliver_dir)
        #if os.path.exists(d):
        shutil.rmtree(pklz_dir)

    def test_volumetry_evaluation_sliver_score(self):
        """
        Testing Volume Difference score. Score for negative values must be
        equal. Score for far high values must be 0.
        """
        score = volumetry_evaluation.sliverScore([1, -1, 30, 50, 100], 'vd')
        self.assertAlmostEquals(score[0], score[1])
        self.assertEqual(score[2], 0)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logger.setLevel(logging.DEBUG)
    unittest.main()
