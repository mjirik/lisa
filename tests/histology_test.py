#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest
from nose.plugins.attrib import attr
import numpy as np

from lisa.gen_volume_tree import TreeGenerator
from lisa.histology_analyser import HistologyAnalyser
from lisa.histology_report import HistologyReport


class HistologyTest(unittest.TestCase):
    interactiveTests = False

    @attr("LAR")
    def test_vessel_tree_lar(self):
        import lisa.gt_lar
        tvg = TreeGenerator(lisa.gt_lar.GTLar)
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
    def test_surface_per_volume_gensei_data(self):
        import lisa.surface_measurement as sm
        import io3d
        dr = io3d.datareader.DataReader()
        datap = dr.Get3DData('sample_data/gensei_slices/',
                             dataplus_format=True)
        # total object volume fraction:           0.081000
        # total object volume [(mm)^3]:           81.000000
        # total object surface fraction [1/(mm)]: 0.306450
        # total object surface [(mm)^2]:          306.449981
        segmentation = (datap['data3d'] > 100).astype(np.int8)
        voxelsize_mm = [0.2, 0.2, 0.2]
        volume = np.sum(segmentation) * np.prod(voxelsize_mm)

        Sv = sm.surface_per_volume(segmentation, voxelsize_mm)
        self.assertGreater(volume, 80)
        self.assertLess(volume, 85)
        self.assertGreater(Sv, 0.3)
        self.assertLess(Sv, 0.4)

    def test_surface_measurement(self):
        import lisa.surface_measurement as sm
        data1 = np.zeros([30, 30, 30])
        voxelsize_mm = [1, 1, 1]
        data1[10:20, 10:20, 10:20] = 1

        Sv1 = sm.surface_per_volume(data1, voxelsize_mm)

        data2 = np.zeros([30, 30, 30])
        voxelsize_mm = [1, 1, 1]
        data2[10:20, 10:20, 10:20] = 1
        data2[10:15, 10:15, 10:15] = 0
        Sv2 = sm.surface_per_volume(data2, voxelsize_mm)

        self.assertEqual(Sv2, Sv1)

        data3 = np.zeros([30, 30, 30])
        voxelsize_mm = [1, 1, 1]
        data3[10:20, 10:20, 10:20] = 1
        data3[13:18, 13:18, 10:15] = 0
        Sv3 = sm.surface_per_volume(data3, voxelsize_mm)
        self.assertGreater(Sv3, Sv1)
        # import sed3
        # ed = sed3.sed3(im_edg)
        # ed.show()

    def test_surface_measurement_find_edge(self):
        import lisa.surface_measurement as sm
        tvg = TreeGenerator()
        yaml_path = os.path.join(path_to_script, "./hist_stats_test.yaml")
        tvg.importFromYaml(yaml_path)
        tvg.voxelsize_mm = [1, 1, 1]
        tvg.shape = [100, 100, 100]
        data3d = tvg.generateTree()

        # init histology Analyser
        # metadata = {'voxelsize_mm': tvg.voxelsize_mm}
        # data3d = data3d * 10
        # threshold = 2.5

        im_edg = sm.find_edge(data3d, 0)
        # in this area should be positive edge
        self.assertGreater(
            np.sum(im_edg[25:30, 25:30, 30]),
            3
        )
        # self.assert(im_edg
        # import sed3
        # ed = sed3.sed3(im_edg)
        # ed.show()


if __name__ == "__main__":
    unittest.main()
