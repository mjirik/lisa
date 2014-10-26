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
    def test_skeleton(self):
        data = np.zeros([20, 20, 20], dtype=np.int8)
        data [3:17, 5, 5] = 1
        # crossing
        data [12, 5:13, 5] = 1
        # vyrustek
        data [8, 5, 5:8] = 1

        skan = sk.SkeletonAnalyser(data)
        skan.skeleton_analysis()

        pe = ped.py3DSeedEditor(skan.sklabel)
        pe.show()

        pass

if __name__ == "__main__":
    unittest.main()
