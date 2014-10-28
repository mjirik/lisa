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
import src.skeleton_analyser as sk
import py3DSeedEditor as ped


class TemplateTest(unittest.TestCase):

    @attr('actual')
    @attr('interactive')
    def test_skeleton(self):
        import skelet3d

        data = np.zeros([20, 20, 20], dtype=np.int8)
        data[5, 3:17, 5] = 1
        # crossing
        data[5, 12, 5:13] = 1
        # vyrustek
        data[5, 8, 5:9] = 1

        pe = ped.py3DSeedEditor(data)
        pe.show()
        data = skelet3d.skelet3d(data)

        skan = sk.SkeletonAnalyser(data)
        skan.skeleton_analysis()

        pe = ped.py3DSeedEditor(skan.sklabel)
        pe.show()


if __name__ == "__main__":
    unittest.main()
