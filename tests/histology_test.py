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
        from gen_volume_tree import TreeVolumeGenerator
        from histology_analyser import HistologyAnalyser
        from histology_report import HistologyReport
        import segmentation
        import misc
        
        # generate 3d data from yaml for testing
        tvg = TreeVolumeGenerator()
        yaml_path = os.path.join(path_to_script, "./hist_stats_test.yaml")
        tvg.importFromYaml(yaml_path)
        tvg.voxelsize_mm = [1,1,1]
        tvg.shape = [100,100,100]
        tvg.generateTree()
        
        # init histology Analyser
        metadata = {'voxelsize_mm': tvg.voxelsize_mm}
        data3d = tvg.data3d*10
        threshold = 2.5
        ha = HistologyAnalyser(data3d, metadata, threshold)
        
        # modifited ha.data_to_skeleton() function
        data3d_thr = segmentation.vesselSegmentation(
            data3d,
            segmentation=np.ones(tvg.data3d.shape, dtype='int8'),
            threshold=threshold,
            inputSigma=0.15,
            dilationIterations=2,
            nObj=1,
            biggestObjects=False,
            interactivity=False,
            binaryClosingIterations=5,
            binaryOpeningIterations=1)
        data3d_skel = ha.binar_to_skeleton(data3d_thr)
        
        # get statistics
        ha.skeleton_to_statistics(data3d_thr, data3d_skel)
        yaml_new = "hist_stats_new.yaml"
        ha.writeStatsToYAML(filename=yaml_new)
        
        # get histology reports
        hr = HistologyReport()
        hr.importFromYaml(yaml_path)
        hr.generateStats()
        stats_orig = hr.stats
        
        hr = HistologyReport()
        hr.importFromYaml(os.path.join(path_to_script, yaml_new))
        hr.generateStats()
        stats_new = hr.stats
        
        # compare
        self.assertGreater(stats_orig['Total length mm'],stats_new['Total length mm']*0.9)
        self.assertLess(stats_orig['Total length mm'],stats_new['Total length mm']*1.1)
        
        self.assertGreater(stats_orig['Avg length mm'],stats_new['Avg length mm']*0.9)
        self.assertLess(stats_orig['Avg length mm'],stats_new['Avg length mm']*1.1)
        
        self.assertGreater(stats_orig['Avg radius mm'],stats_new['Avg radius mm']*0.9)
        self.assertLess(stats_orig['Avg radius mm'],stats_new['Avg radius mm']*1.1)


if __name__ == "__main__":
    unittest.main()
