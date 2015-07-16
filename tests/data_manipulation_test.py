#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@hp-mjirik>
#
# Distributed under terms of the MIT license.

"""

"""
import unittest
from nose.plugins.attrib import attr
import numpy as np
import scipy
import lisa.data_manipulation as dama

class DataManipulationTest(unittest.TestCase):

    @attr('actual')
    def test_unbiased_brick(self):
        """
        Test unbiased brick. There sould be correct number of object in
        sum of subbrick.
        """
# import np.random
        shp = [40, 42, 44]
        data = np.zeros(shp, dtype=np.uint8)
        data[14:18, 22, 17] = 1
        data[12:16, 15, 15] = 1
        data[13:18, 18, 13] = 1
        # data[26:29, 21, 18] = 1
        # data[21, 23:26, 25] = 1
        # data[21, 17:21, 24] = 1
        # data[16, 26:29, 21] = 1
        # data[24, 28, 13:23] = 1
        # data[19, 28, 12:16] = 1
        # data[19, 15, 22:26] = 1


        # import sed3
        # se = sed3.sed3(data)
        # se.show()

# pokud to pocita dobr
        sh = [None] * 8
        sh[0] = [[10,20],[10,20],[10,20]]
        sh[1] = [[10,20],[10,20],[20,30]]
        sh[2] = [[10,20],[20,30],[10,20]]
        sh[3] = [[10,20],[20,30],[20,30]]
        sh[4] = [[20,30],[10,20],[10,20]]
        sh[5] = [[20,30],[10,20],[20,30]]
        sh[6] = [[20,30],[20,30],[10,20]]
        sh[7] = [[20,30],[20,30],[20,30]]

        suma = 0

        for shi in sh:

            outputdata = dama.unbiased_brick_filter(data, shi)
            imlab, num_features = scipy.ndimage.measurements.label(outputdata)
            print num_features
            suma += num_features

        print "for is over"

        print suma
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT


        

if __name__ == "__main__":
    unittest.main()
