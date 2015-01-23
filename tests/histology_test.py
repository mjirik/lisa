#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest
from nose.plugins.attrib import attr

from lisa.gen_volume_tree import TreeGenerator
from lisa.histology_analyser import HistologyAnalyser
from lisa.histology_report import HistologyReport


class HistologyTest(unittest.TestCase):
    interactiveTests = False

    @attr("interactive")
    def test_vessel_tree_lar(self):
        import lisa.gt_lar
        tvg = TreeGenerator(gt_lar.GTLar)
        yaml_path = os.path.join(path_to_script, "./hist_stats_test.yaml")
        tvg.importFromYaml(yaml_path)
        tvg.voxelsize_mm = [1, 1, 1]
        tvg.shape = [100, 100, 100]
        output = tvg.generateTree() # noqa
        if self.interactiveTests:
            tvg.show()

    def test_synthetic_data_vessel_tree_evaluation(self):
        """
        Generovani umeleho stromu do 3D a jeho evaluace.
        V testu dochazi ke kontrole predpokladaneho objemu a deky cev


        """
        # import segmentation
        # import misc

        # generate 3d data from yaml for testing
        tvg = TreeGenerator()
        yaml_path = os.path.join(path_to_script, "./hist_stats_test.yaml")
        tvg.importFromYaml(yaml_path)
        tvg.voxelsize_mm = [1, 1, 1]
        tvg.shape = [100, 100, 100]
        data3d = tvg.generateTree()

        # init histology Analyser
        metadata = {'voxelsize_mm': tvg.voxelsize_mm}
        data3d = data3d * 10
        threshold = 2.5
        ha = HistologyAnalyser(data3d, metadata, threshold)

        # segmented data
        ha.data_to_binar()
        ha.data_to_skeleton()

        # get statistics
        ha.data_to_statistics()
        yaml_new = os.path.join(path_to_script, "hist_stats_new.yaml")
        ha.writeStatsToYAML(filename=yaml_new)

        # get histology reports
        hr = HistologyReport()
        hr.importFromYaml(yaml_path)
        hr.generateStats()
        stats_orig = hr.stats['Report']

        hr = HistologyReport()
        hr.importFromYaml(yaml_new)
        hr.generateStats()
        stats_new = hr.stats['Report']

        # compare
        self.assertGreater(stats_orig['Other']['Total length mm'],stats_new['Other']['Total length mm']*0.9)  # noqa
        self.assertLess(stats_orig['Other']['Total length mm'],stats_new['Other']['Total length mm']*1.1)  # noqa

        self.assertGreater(stats_orig['Other']['Avg length mm'],stats_new['Other']['Avg length mm']*0.9)  # noqa
        self.assertLess(stats_orig['Other']['Avg length mm'],stats_new['Other']['Avg length mm']*1.1)  # noqa

        self.assertGreater(stats_orig['Other']['Avg radius mm'],stats_new['Other']['Avg radius mm']*0.9)  # noqa
        self.assertLess(stats_orig['Other']['Avg radius mm'],stats_new['Other']['Avg radius mm']*1.1)  # noqa

    @attr("actual")
    def test_surface_measurement(self):
        tvg = TreeGenerator()
        yaml_path = os.path.join(path_to_script, "./hist_stats_test.yaml")
        tvg.importFromYaml(yaml_path)
        tvg.voxelsize_mm = [1, 1, 1]
        tvg.shape = [100, 100, 100]
        data3d = tvg.generateTree()

        # init histology Analyser
        metadata = {'voxelsize_mm': tvg.voxelsize_mm}
        data3d = data3d * 10
        threshold = 2.5


if __name__ == "__main__":
    unittest.main()
