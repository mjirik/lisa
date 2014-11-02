#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2014 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy as np
import unittest
from nose.plugins.attrib import attr
import lisa.skeleton_analyser as sk
# import sed3 as ped
import copy


class TemplateTest(unittest.TestCase):

    @attr('actual')
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
        output = skan.filter_small(data, 3)
        # skan.skeleton_analysis()

        # pe = ped.sed3(output)
        # pe.show()

        self.assertEqual(output[5, 8, 7], 0)


if __name__ == "__main__":
    unittest.main()
