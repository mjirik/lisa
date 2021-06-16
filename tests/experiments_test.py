#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
from loguru import logger
import os.path
import sys
import unittest

# logger = logging.getLogger()

import pytest
import numpy as np
import shutil

import matplotlib.pyplot as plt

path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../"))
# sys.path.append(os.path.join(path_to_script, "../lisa/"))

import lisa.volumetry_evaluation as ve
import lisa
import lisa.experiments
import lisa.dataset
from lisa.dataset import join_sdp


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
        experiment_dir = join_sdp("tmp_tests/tmp_exp_small2/")
        experiment_name = 'exp2'

        if os.path.exists(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir)

        # experiment_dir = join_sdp("exp_small2/")

        sliver_reference_dir = join_sdp("exp_small/seg")

# this is setup for visualization
        markers = ['ks',
                   # 'r<'
                   ]
        labels = ['vs 6 mm',
                  'vs 7 mm'
                  ]
        input_data_path_pattern = join_sdp("exp_small/seeds/*.pklz")

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

    @pytest.mark.slow
    def test_experiment_set(self):
        import lisa.experiments

        experiment_path = join_sdp("exp")
        # this var is no necessary
        pklz_dirs = [
            join_sdp("exp/exp1"),
            join_sdp("exp/exp2"),
            # "/home/mjirik/projects/lisa/sample_data/exp1",
            # "/home/mjirik/projects/lisa/sample_data/exp2",

        ]
        sliver_reference_dir = join_sdp("exp/seg")
        # "/home/mjirik/data/medical/orig/sliver07/training/"

# this is setup for visualization
        markers = ['ks', 'r<']
        labels = ['3gaus', '02smoothing']
        input_data_path_pattern = join_sdp("exp/seeds/*.pklz")

# if directory exists, remove it
        for dire in pklz_dirs:
            if os.path.exists(dire):
                shutil.rmtree(dire)

# experiment_support.report(pklz_dirs, labels, markers)
        lisa.experiments.run_and_make_report(
            experiment_path, labels, sliver_reference_dir,
            input_data_path_pattern, markers=markers, pklz_dirs=pklz_dirs, show=False)
        import io3d.misc
        obj = io3d.misc.obj_from_file(pklz_dirs[0] + '.yaml', filetype='yaml')
        self.assertGreater(len(obj['data']), 0)
        # self.assertTrue(False)

    # not sure why is this test failing on travis. On local there are no
    # problems.  Maybe it is not called.
    @pytest.mark.incomplete
    @pytest.mark.slow
    def test_experiment_set_small(self):
        import lisa.experiments

        # os.path.join(path_to_script, "..")
        # pklz_dirs = [
        #     join_sdp("exp_small/exp1"),
        #     # join_sdp("exp_small/exp2"),
        # ]
        experiment_dir = join_sdp("tmp_tests/exp1")
        sliver_reference_dir = join_sdp("exp_small/seg")
        # "/home/mjirik/data/medical/orig/sliver07/training/"

# this is setup for visualization
        markers = ['ks',
                   # 'r<'
                   ]
        labels = ['vs6mm',
                  # '02smoothing'
                  ]
        input_data_path_pattern = join_sdp("exp_small/seeds/*.pklz")

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
        if os.path.exists(experiment_dir):
            shutil.rmtree(experiment_dir)

# experiment_support.report(pklz_dirs, labels, markers)
        lisa.experiments.run_and_make_report(
            experiment_dir, labels, sliver_reference_dir,
            input_data_path_pattern,
            conf_default=conf_default,
            conf_list=conf_list,
            show=False,
            markers=markers
        )
        import io3d.misc
        path_to_yaml = join_sdp("tests_tmp/exp1/exp1-vs6mm.yaml")
        obj = io3d.misc.obj_from_file(path_to_yaml, filetype='yaml')
        self.assertGreater(len(obj['data']), 0)
        # self.assertTrue(False)

    @pytest.mark.actual
    # @unittest.skip("this test is little rebel under travis-ci")
    def test_experiment_set_small_per_partes(self):
        plt.ioff()
        # import lisa.experiments

        # os.path.join(path_to_script, "..")
        experiment_dir = lisa.dataset.join_sdp("exp22")

        sliver_reference_dir = join_sdp("exp_small/seg")
        # "/home/mjirik/data/medical/orig/sliver07/training/"

# this is setup for visualization
        labels = ['vs6mm',
                  # '02smoothing'
                  ]
        input_data_path_pattern = lisa.dataset.join_sdp("exp_small/seeds/*.pklz")

        conf_default = {
            'config_version': [1, 0, 0], 'working_voxelsize_mm': 2.0,
            'segmentation_smoothing': False,
            "run_organ_segmentation": True,
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

        data_path = lisa.dataset.join_sdp("exp22/")
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
            lisa.dataset.join_sdp("exp_synth/exp1"),
            lisa.dataset.join_sdp("exp_synth/exp2")
            ]

        os.mkdir(
            lisa.dataset.join_sdp("exp_synth"),
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

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logger.setLevel(logging.DEBUG)
    unittest.main()
