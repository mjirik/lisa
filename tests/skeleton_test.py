#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2014 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.

"""

"""
import logging
logger = logging.getLogger(__name__)
import numpy as np
import unittest
from nose.plugins.attrib import attr
import skelet3d.skeleton_analyser as sk
import copy


class TemplateTest(unittest.TestCase):

    @attr('actual')
    @attr('slow')
    def test_nodes_aggregation_big_data(self):

        data = np.zeros([1000, 1000, 100], dtype=np.int8)
        voxelsize_mm = [14, 10, 6]
        
        # snake
        # data[15:17, 13, 13] = 1
        data[18, 3:17, 12] = 1
        data[18, 17, 13:17] = 1
        data[18, 9, 4:12] = 1
        data[18, 14, 12:19] = 1
        # data[18, 18, 15:17] = 1

        # T-junction on the left
        data[18, 4:16, 3] = 1
        import sed3
        # ed = sed3.sed3(data)
        # ed.show()

        skel = data

        skan = sk.SkeletonAnalyser(
                copy.copy(skel), 
                volume_data=data,
                voxelsize_mm=voxelsize_mm, 
                cut_wrong_skeleton=False, 
                aggregate_near_nodes_distance=20)
        vessel_tree = skan.skeleton_analysis()
        # print 'skan completeskan completed'

        # ed = sed3.sed3(skan.sklabel, contour=data)
        # ed.show()
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        # number of terminals + inner nodes
        self.assertEqual(np.min(skan.sklabel), -7)
        # number of edges
        self.assertEqual(np.max(skan.sklabel), 6)

    def test_nodes_aggregation(self):

        data = np.zeros([20, 20, 20], dtype=np.int8)
        voxelsize_mm = [14, 10, 6]

        # snake
        # data[15:17, 13, 13] = 1
        data[17, 3:17, 12] = 1
        data[17, 17, 13:17] = 1
        data[17, 9, 4:12] = 1
        data[17, 14, 12:19] = 1
        # data[18, 18, 15:17] = 1

        # T-junction on the left
        data[17, 4:16, 3] = 1
        import sed3
        # ed = sed3.sed3(data)
        # ed.show()

        skel = data

        skan = sk.SkeletonAnalyser(
                copy.copy(skel), 
                volume_data=data,
                voxelsize_mm=voxelsize_mm, 
                cut_wrong_skeleton=False, 
                aggregate_near_nodes_distance=20)
        vessel_tree = skan.skeleton_analysis()

        # ed = sed3.sed3(skan.sklabel, contour=data)
        # ed.show()
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        # number of terminals + inner nodes
        self.assertEqual(np.min(skan.sklabel), -7)
        # number of edges
        self.assertEqual(np.max(skan.sklabel), 6)

    def test_generate_elipse(self):
        mask = sk.generate_binary_elipsoid([6, 4, 3])
        print(mask.shape)



        self.assertEqual(mask[0][0][0], 0)
        self.assertEqual(mask[6][4][3], 1)
# on axis border should be zero
        self.assertEqual(mask[0][4][3], 0)
        self.assertEqual(mask[6][0][3], 0)
        self.assertEqual(mask[6][4][0], 0)

# on axis border one pixel into center should be one
        self.assertEqual(mask[1][4][3], 1)
        self.assertEqual(mask[6][1][3], 1)
        self.assertEqual(mask[6][4][1], 1)

    def test_length_types(self):
        """
        Test for comparation of various length estimation methods.
        No strong assert here.
        """
        data = np.zeros([20, 20, 20], dtype=np.int8)
        # snake
        data[8:10, 14, 13] = 1
        data[10:12, 15, 13] = 1
        data[12:14, 16, 13] = 1
        # data[18, 14:17, 13] = 1
        # data[18, 17, 14:17] = 1
        # data[14:18, 17, 17] = 1

        skel = data

        skan = sk.SkeletonAnalyser(copy.copy(skel), volume_data=data,
                                   voxelsize_mm=[1, 20, 300])
        # skan.spline_smoothing = 5
        vessel_tree = skan.skeleton_analysis()
        pixel = vessel_tree[1]['lengthEstimationPixel']
        poly = vessel_tree[1]['lengthEstimationPoly']
        spline = vessel_tree[1]['lengthEstimationSpline']
        # print 'pixel', pixel
        # print poly
        # print spline

        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        # self.assertAlmostEqual
        # self.assertAlmostEqual(vessel_tree[1]['lengthEstimationPixel'], 10)
        # self.assertAlmostEqual(vessel_tree[2]['lengthEstimationPixel'], 200)
        # self.assertAlmostEqual(vessel_tree[3]['lengthEstimationPixel'], 3000)
        # self.assertAlmostEqual(vessel_tree[1]['lengthEstimationPixel'],
        #                        diag_length)

    def test_length(self):

        data = np.zeros([20, 20, 20], dtype=np.int8)
        data[2:13, 4, 4] = 1
        data[6, 2:13, 6] = 1
        data[8, 8, 2:13] = 1

        # diagonala
        data[11, 11, 11] = 1
        data[12, 12, 12] = 1
        data[13, 13, 13] = 1

        # snake
        # data[15:17, 13, 13] = 1
        data[18, 14:17, 13] = 1
        data[18, 17, 14:17] = 1
        data[14:18, 17, 17] = 1
        # data[18, 18, 15:17] = 1

        skel = data

        skan = sk.SkeletonAnalyser(copy.copy(skel), volume_data=data,
                                   voxelsize_mm=[1, 20, 300], cut_wrong_skeleton=False)
        vessel_tree = skan.skeleton_analysis()

        self.assertAlmostEqual
        self.assertAlmostEqual(vessel_tree[1]['lengthEstimationPixel'], 10)
        self.assertAlmostEqual(vessel_tree[2]['lengthEstimationPixel'], 200)
        self.assertAlmostEqual(vessel_tree[3]['lengthEstimationPixel'], 3000)
        diag_length = 2 * ((1**2 + 20**2 + 300**2)**0.5)
        self.assertAlmostEqual(vessel_tree[4]['lengthEstimationPixel'],
                               diag_length)
        # test spline
        self.assertLess(
            vessel_tree[1]['lengthEstimationPixel']
            - vessel_tree[1]['lengthEstimationSpline'],
            0.001
        )
        # test poly
        self.assertLess(
            vessel_tree[1]['lengthEstimationPixel']
            - vessel_tree[1]['lengthEstimationPoly'],
            0.001
        )

    def test_tortuosity(self):
        import skelet3d

        data = np.zeros([20, 20, 20], dtype=np.int8)
        # banana
        data[5, 3:11, 4:6] = 1
        data[5, 10:12, 5:12] = 1

        # bar
        data[5, 3:11, 15:17] = 1

        # import sed3 as ped
        # pe = ped.sed3(data)
        # pe.show()
        skel = skelet3d.skelet3d(data)

        # pe = ped.sed3(skel)
        # pe.show()

        skan = sk.SkeletonAnalyser(copy.copy(skel), volume_data=data,
                                   voxelsize_mm=[1, 1, 1])
        vessel_tree = skan.skeleton_analysis()

        # banana
        self.assertGreater(vessel_tree[1]['tortuosity'], 1.2)
        # bar
        self.assertLess(
            vessel_tree[2]['tortuosity'] - 1,
            0.00001
        )

    def test_fileter_small(self):
        import skelet3d

        data = np.zeros([20, 20, 20], dtype=np.int8)
        data[5, 3:17, 5] = 1
        # crossing
        data[5, 12, 5:13] = 1
        # vyrustek
        data[5, 8, 5:9] = 1

        data = skelet3d.skelet3d(data)
        # pe = ped.sed3(data)
        # pe.show()

        skan = sk.SkeletonAnalyser(copy.copy(data))
        output = skan.filter_small_objects(data, 3)
        # skan.skeleton_analysis()

        # pe = ped.sed3(output)
        # pe.show()

        self.assertEqual(output[5, 8, 7], 0)


if __name__ == "__main__":
    unittest.main()
