# ! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import lisa.volumetry_evaluation as ve


class VolumetryEvaluationTest(unittest.TestCase):


    # @unittest.skip("Waiting for implementation")
    def test_compare_volumes(self):
        aa = np.zeros([6,6,6], dtype=np.uint8)
        bb = np.zeros([6,6,6], dtype=np.uint8)

        aa[1:5, 1:5, 1:3] = 1
        bb[1:5, 1:3, 1:5] = 1

        import lisa
        stats = lisa.volumetry_evaluation.compare_volumes(aa, bb, [1,1,1])
        # lisa.volumetry_evaluation.compare_volumes(aa.astype(np.int8), bb.astype(np.int), [1,1,1])
        self.assertEqual(stats["dice"], 0.5)
        self.assertAlmostEqual(stats["jaccard"], 1/3.)
        self.assertAlmostEqual(stats["voe"], 200/3.)  # 66.666666
        # self.assertTrue(False)

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

    @pytest.mark.incomplete
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
        # import lisa.liver_segmentation_cerveny as ls
        # import nearpy
        # engine = nearpy.Engine(dim=3)
        # eval2 = ls.vzdalenosti(vol1, vol2, [1, 1, 1], engine)
        # self.assertLess(eval1[0], 1.1 * eval2[0])
        # self.assertGreater(eval1[0], 0.9 * eval2[0])

    # @pytest.mark.incomplete
    # def test_compare_eval_sliver_distance_bigger(self):
    #     """
    #     comparison of two methods for surface distance computation on bigger
    #     object
    #     Second implementation is obsolete
    #     """
    #
    #     vol1 = np.zeros([100, 210, 220], dtype=np.int8)
    #     vol1[10:50, 50:150, 100:150] = 1
    #
    #     vol2 = np.zeros([100, 210, 220], dtype=np.int8)
    #     vol2[12:52, 45:160, 100:150] = 1
    #
    #     eval1 = ve.distance_matrics(vol1, vol2, [1, 1, 1])
    #
    #     import lisa.liver_segmentation_cerveny as ls
    #     import nearpy
    #     engine = nearpy.Engine(dim=3)
    #     eval2 = ls.vzdalenosti(vol1, vol2, [1, 1, 1], engine)
    #     # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
    #     self.assertLess(eval1[0], 1.1 * eval2[0])
    #     self.assertGreater(eval1[0], 0.9 * eval2[0])

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

