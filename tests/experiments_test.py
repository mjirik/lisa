#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import logging
import os.path
import sys
import unittest

logger = logging.getLogger(__name__)

from nose.plugins.attrib import attr
import numpy as np
import shutil

import matplotlib.pyplot as plt

path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../"))
# sys.path.append(os.path.join(path_to_script, "../lisa/"))

import lisa.volumetry_evaluation as ve
import lisa
import lisa.experiments


# just test if everything is going ok
def speceval(vol1, vol2, vs):
    """
    special_evaluation_function
    """
    return {'one': 1}

class ExperimentsTest(unittest.TestCase):

    def test_experiment_set_(self):
        """
        pklz_dirs as prefix is tested.
        """

# if directory exists, remove it
        experiment_dir = os.path.abspath(
            path_to_script + "./../tests/tmp_exp_small2/")
        experiment_name = 'exp2'

        if os.path.exists(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.mkdir(experiment_dir)

        # experiment_dir = os.path.abspath(
        #     path_to_script + "./../sample_data/exp_small2/")

        sliver_reference_dir = os.path.abspath(
            path_to_script + "./../sample_data/exp/seg")

# this is setup for visualization
        markers = ['ks',
                   # 'r<'
                   ]
        labels = ['vs 6 mm',
                  'vs 7 mm'
                  ]
        input_data_path_pattern = os.path.abspath(
            path_to_script + "./../sample_data/exp_small/seeds/*.pklz")

        conf_default = {
            'config_version': [1, 0, 0], 'working_voxelsize_mm': 2.0,
            'segmentation_smoothing': False,
            'segparams': {'pairwise_alpha_per_mm2': 50,
                          'return_only_object_with_seeds': True}
        }
        conf_list = [
            {'working_voxelsize_mm': 6},
            {'working_voxelsize_mm': 7}
        ]

#
# # experiment_support.report(pklz_dirs, labels, markers)
        ramr = lisa.experiments.RunAndMakeReport(
            experiment_dir, labels, sliver_reference_dir,
            input_data_path_pattern,
            conf_default=conf_default,
            conf_list=conf_list,
            show=False, use_plt=False,
            experiment_name=experiment_name,
            markers=markers
        )
        ramr.config()
        dir_eval = os.listdir(experiment_dir)
        # self.assertIn('exp1_eval.pkl', dir_eval)
        # self.assertIn('exp1_eval.csv', dir_eval)
        self.assertIn('exp2-vs6mm.config', dir_eval)
        self.assertIn('exp2-vs7mm.config', dir_eval)
        # self.assertIn('exp1.yaml', dir_eval)
#         # self.assertTrue(False)
        ramr.run_experiments()
        ramr.evaluation()
        ramr.report()
        # shutil.rmtree(dire)

    @attr("slow")
    def test_experiment_set(self):
        import lisa.experiments

        os.path.join(path_to_script, "..")
        pklz_dirs = [
            os.path.abspath(path_to_script + "./../sample_data/exp/exp1"),
            os.path.abspath(path_to_script + "./../sample_data/exp/exp2"),
            # "/home/mjirik/projects/lisa/sample_data/exp1",
            # "/home/mjirik/projects/lisa/sample_data/exp2",

        ]
        sliver_reference_dir = os.path.abspath(
            path_to_script + "./../sample_data/exp/seg")
        # "/home/mjirik/data/medical/orig/sliver07/training/"

# this is setup for visualization
        markers = ['ks', 'r<']
        labels = ['3gaus', '02smoothing']
        input_data_path_pattern = os.path.abspath(
            path_to_script + "./../sample_data/exp/seeds/*.pklz")

# if directory exists, remove it
        for dire in pklz_dirs:
            shutil.rmtree(dire)

# experiment_support.report(pklz_dirs, labels, markers)
        lisa.experiments.run_and_make_report(
            pklz_dirs, labels, markers, sliver_reference_dir,
            input_data_path_pattern, show=False)
        import io3d.misc
        obj = io3d.misc.obj_from_file(pklz_dirs[0] + '.yaml', filetype='yaml')
        self.assertGreater(len(obj['data']), 0)
        # self.assertTrue(False)

    # not sure why is this test failing on travis. On local there are no
    # problems.  Maybe it is not called.
    @attr("incomplete")
    @attr("slow")
    def test_experiment_set_small(self):
        import lisa.experiments

        # os.path.join(path_to_script, "..")
        pklz_dirs = [
            os.path.abspath(
                path_to_script + "./../sample_data/exp_small/exp1"),
            # os.path.abspath(
            #     path_to_script + "./../sample_data/exp_small/exp2"),
        ]
        sliver_reference_dir = os.path.abspath(
            path_to_script + "./../sample_data/exp_small/seg")
        # "/home/mjirik/data/medical/orig/sliver07/training/"

# this is setup for visualization
        markers = ['ks',
                   # 'r<'
                   ]
        labels = ['vs6mm',
                  # '02smoothing'
                  ]
        input_data_path_pattern = os.path.abspath(
            path_to_script + "./../sample_data/exp_small/seeds/*.pklz")

        conf_default = {
            'config_version': [1, 0, 0], 'working_voxelsize_mm': 2.0,
            'segmentation_smoothing': False,
            'segparams': {'pairwise_alpha_per_mm2': 50,
                          'return_only_object_with_seeds': True}
        }
        conf_list = [
            # {'working_voxelsize_mm': 4},
            {'working_voxelsize_mm': 6}
        ]

# if directory exists, remove it
        for dire in pklz_dirs:
            if os.path.exists(dire):
                shutil.rmtree(dire)

# experiment_support.report(pklz_dirs, labels, markers)
        lisa.experiments.run_and_make_report(
            pklz_dirs, labels, sliver_reference_dir,
            input_data_path_pattern,
            conf_default=conf_default,
            conf_list=conf_list,
            show=False,
            markers=markers
        )
        import io3d.misc
        obj = io3d.misc.obj_from_file(pklz_dirs[0] + '.yaml', filetype='yaml')
        self.assertGreater(len(obj['data']), 0)
        # self.assertTrue(False)

    @attr("actual")
    # @unittest.skip("this test is little rebel under travis-ci")
    def test_experiment_set_small_per_partes(self):
        plt.ioff()
        # import lisa.experiments

        # os.path.join(path_to_script, "..")
        experiment_dir = os.path.abspath(
            path_to_script + "./../sample_data/exp22")

        sliver_reference_dir = os.path.abspath(
            path_to_script + "./../sample_data/exp_small/seg")
        # "/home/mjirik/data/medical/orig/sliver07/training/"

# this is setup for visualization
        labels = ['vs6mm',
                  # '02smoothing'
                  ]
        input_data_path_pattern = os.path.abspath(
            path_to_script + "./../sample_data/exp_small/seeds/*.pklz")

        conf_default = {
            'config_version': [1, 0, 0], 'working_voxelsize_mm': 2.0,
            'segmentation_smoothing': False,
            'segparams': {'pairwise_alpha_per_mm2': 50,
                          'return_only_object_with_seeds': True}
        }
        conf_list = [
            # {'working_voxelsize_mm': 4},
            {'working_voxelsize_mm': 6}
        ]

        if os.path.exists(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.mkdir(experiment_dir)
#
# # experiment_support.report(pklz_dirs, labels, markers)
        ramr = lisa.experiments.RunAndMakeReport(
            experiment_dir, labels, sliver_reference_dir,
            input_data_path_pattern,
            conf_default=conf_default,
            conf_list=conf_list,
            show=False, use_plt=False)

        self.assertTrue(ramr.is_evaluation_necessary())
        self.assertTrue(ramr.is_run_experiments_necessary())


        ramr.config()
        ramr.run_experiments()
        # ramr.evaluation(special_evaluation_function=speceval)
        ramr.evaluation(
            special_evaluation_function=lisa.volumetry_evaluation.compare_volumes_boundingbox)
        ramr.report()

        self.assertFalse(ramr.is_evaluation_necessary())
        self.assertFalse(ramr.is_run_experiments_necessary())

        data_path = os.path.abspath(
            path_to_script + "./../sample_data/exp22/")
        dir_eval = os.listdir(data_path)
        self.assertIn('exp22-vs6mm_eval.pkl', dir_eval)
        self.assertIn('exp22-vs6mm_eval.csv', dir_eval)
        self.assertIn('exp22-vs6mm.config', dir_eval)
        self.assertIn('exp22-vs6mm.yaml', dir_eval)
        import io3d.misc
        obj = io3d.misc.obj_from_file(
            os.path.join(experiment_dir, 'exp22-vs6mm.yaml'),
            filetype='yaml')
        self.assertGreater(len(obj['data']), 0)
#         # self.assertTrue(False)

    def prepare_data_for_fast_experiment(self):
        """
        not used
        """

        pklz_dirs = [
            os.path.abspath(path_to_script + "./../sample_data/exp_synth/exp1"),
            os.path.abspath(path_to_script + "./../sample_data/exp_synth/exp2"),

            ]

        os.mkdir(
            os.path.abspath(path_to_script + "./../sample_data/exp_synth"),
        )
        os.mkdir(pklz_dirs[0])
        os.mkdir(pklz_dirs[1])
        pass

    def test_get_subdirs(self):
        dirpath = os.path.join(path_to_script, "..")
        dirlist = lisa.experiments.get_subdirs(dirpath)
        # import pdb; pdb.set_trace()

        self.assertTrue('tests' in dirlist)
        self.assertTrue('lisa' in dirlist)
        self.assertFalse('README.md' in dirlist)

    def test_sliver_overall_score_for_one_couple(self):

        score = {
            'vd': 15.5,
            'voe': 15.6,
            'avgd': 14.4,
            'rmsd': 20,
            'maxd': 10,
        }

        overall_score = ve.sliver_overall_score_for_one_couple(score)
        self.assertAlmostEqual(15.1, overall_score)

    def test_eval_sliver_matrics(self):
        voxelsize_mm = [1, 2, 3]

        vol1 = np.zeros([20, 21, 22], dtype=np.int8)
        vol1[10:15, 10:15, 10:15] = 1

        vol2 = np.zeros([20, 21, 22], dtype=np.int8)
        vol2[10:15, 10:16, 10:15] = 1

        eval1 = ve.compare_volumes(vol1, vol2, voxelsize_mm)
        # print ve.sliver_score(eval1['vd'], 'vd')

        self.assertAlmostEqual(eval1['vd'], 20.0)

        score = ve.sliver_score_one_couple(eval1)
        # score is 21.875
        self.assertGreater(score['vd'], 20)
        self.assertLess(score['vd'], 25)

    def test_eval_sliver_distance_for_two_pixels_bigger_volume(self):
        """
        maxd is measured in corner. It is  space diagonal of 2 pixels cube.
        """

        vol1 = np.zeros([20, 21, 22], dtype=np.int8)
        vol1[10:15, 10:15, 10:15] = 1

        vol2 = np.zeros([20, 21, 22], dtype=np.int8)
        vol2[8:17, 8:17, 8:17] = 1

        eval1 = ve.distance_matrics(vol1, vol2, [1, 1, 1])

        self.assertAlmostEquals(eval1[2], 3 ** (0.5) * 2)

    def test_eval_sliver_distance_two_points(self):
        """
        Two points 2 by 2 pixels diagonal.
        """

        vol1 = np.zeros([20, 21, 22], dtype=np.int8)
        vol1[10, 10, 10] = 1

        vol2 = np.zeros([20, 21, 22], dtype=np.int8)
        vol2[10, 12, 12] = 1

        eval1 = ve.distance_matrics(vol1, vol2, [1, 1, 1])
        self.assertAlmostEquals(eval1[2], np.sqrt(2) * 2)

    def test_eval_sliver_distance_two_points_non_comutative(self):
        """
        Two volumes - obj1 1px and obj2 2px
        There is different maximal distance from first to second then
        from second to first

        """

        vol1 = np.zeros([20, 21, 22], dtype=np.int8)
        vol1[10, 10, 10] = 1

        vol2 = np.zeros([20, 21, 22], dtype=np.int8)
        vol2[10, 12, 10] = 1
        vol2[10, 14, 10] = 1

        eval1 = ve.distance_matrics(vol1, vol2, [1, 1, 1])

        self.assertLess(eval1[0], 4)
        self.assertGreater(eval1[0], 2)

    def test_eval_sliver_distance_two_points_with_half_voxelsize(self):
        """
        Two points 2 by 2 pixels diagonal and voxelsize is 0.5
        """

        vol1 = np.zeros([20, 21, 22], dtype=np.int8)
        vol1[10, 10, 10] = 1

        vol2 = np.zeros([20, 21, 22], dtype=np.int8)
        vol2[10, 12, 12] = 1

        eval1 = ve.distance_matrics(vol1, vol2, [0.5, 0.5, 0.5])
        self.assertAlmostEquals(eval1[2], np.sqrt(2))

    @attr("incomplete")
    def test_compare_eval_sliver_distance(self):
        """
        comparison of two methods for surface distance computation
        Second implementation is obsolete
        """

        vol1 = np.zeros([20, 21, 22], dtype=np.int8)
        vol1[10:15, 10:15, 10:15] = 1

        vol2 = np.zeros([20, 21, 22], dtype=np.int8)
        vol2[8:17, 8:17, 8:17] = 1

        eval1 = ve.distance_matrics(vol1, vol2, [1, 1, 1])

        self.assertAlmostEquals(eval1[2], 3 ** (0.5) * 2)
        import lisa.liver_segmentation as ls
        import nearpy
        engine = nearpy.Engine(dim=3)
        eval2 = ls.vzdalenosti(vol1, vol2, [1, 1, 1], engine)
        self.assertLess(eval1[0], 1.1 * eval2[0])
        self.assertGreater(eval1[0], 0.9 * eval2[0])

    @attr("incomplete")
    def test_compare_eval_sliver_distance_bigger(self):
        """
        comparison of two methods for surface distance computation on bigger
        object
        Second implementation is obsolete
        """

        vol1 = np.zeros([100, 210, 220], dtype=np.int8)
        vol1[10:50, 50:150, 100:150] = 1

        vol2 = np.zeros([100, 210, 220], dtype=np.int8)
        vol2[12:52, 45:160, 100:150] = 1

        eval1 = ve.distance_matrics(vol1, vol2, [1, 1, 1])

        import lisa.liver_segmentation as ls
        import nearpy
        engine = nearpy.Engine(dim=3)
        eval2 = ls.vzdalenosti(vol1, vol2, [1, 1, 1], engine)
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        self.assertLess(eval1[0], 1.1 * eval2[0])
        self.assertGreater(eval1[0], 0.9 * eval2[0])

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

        # d = os.path.dirname(sliver_dir)
        d = sliver_dir
        if os.path.exists(d):
            shutil.rmtree(d)
        # if not os.path.exists(d):
        os.makedirs(d)

        # d = os.path.dirname(pklz_dir)
        d = pklz_dir
        if os.path.exists(d):
            shutil.rmtree(d)
        # if not os.path.exists(d):
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
        yamldata = ve.generate_input_yaml(
            sliver_dir, pklz_dir)

# assert

        fls0 = os.path.join(sliver_dir, filelist1[0])
        flp0 = os.path.join(pklz_dir, filelist2[0])
        # we only hope, that this will be first record
        self.assertEqual(yamldata['data'][0]['sliverseg'], fls0)
        self.assertEqual(yamldata['data'][0]['ourseg'], flp0)

# Clean
        # if os.path.exists(d):
        shutil.rmtree(sliver_dir)
        # if os.path.exists(d):
        shutil.rmtree(pklz_dir)

    def test_volumetry_evaluation_sliver_score(self):
        """
        Testing Volume Difference score. Score for negative values must be
        equal. Score for far high values must be 0.
        """
        score = ve.sliver_score([1, -1, 30, 50, 100], 'vd')
        self.assertAlmostEquals(score[0], score[1])
        self.assertEqual(score[2], 0)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logger.setLevel(logging.DEBUG)
    unittest.main()
