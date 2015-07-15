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
import lisa.data_manipulation as dama

class TemplateTest(unittest.TestCase):

    @attr('actual')
    def test_unbiased_brick(self):
# import np.random
        shp = [30,32,34]
        data = np.zeros(shp, dtype=np.uint8)
        data[11:17, 11:13, 14:15] = 1
        data[6:14, 14:16, 14:15] = 1
        data[14:16, 11:18, 16:18] = 1
        data[16:18, 13:16, 18:23] = 1
        data[13:15, 11:16, 7:12] = 1
        data[16:18, 7:12, 16:18] = 1
        # import sed3
        # se = sed3.sed3(data)
        # se.show()
        # data += (np.random.random(shp) * 50).astype(np.int16)
        outputdata = dama.unbiased_brick_filter(data, [[10,20],[10,20],[10,20]])
        import sed3
        se = sed3.sed3(outputdata)
        se.show()
        

if __name__ == "__main__":
    unittest.main()
