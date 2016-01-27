#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from nose.plugins.attrib import attr
import lisa.convert
import os.path as op
import os


class MyTestCase(unittest.TestCase):
    @attr('interactive')
    def test_something(self):
        seg = np.zeros([20,21,22], dtype=np.uint8)
        seg [4:11, 3:15, 11:17] = 1
        seg [4:7, 3:10, 11:14] = 0

        path_to_script = op.dirname(op.abspath(__file__))
        fn_stl = op.join(path_to_script, 'test_stl.stl')
        fn_tmp = op.join(path_to_script, 'test_stl_tmp.vtk')

        lisa.convert.seg2stl(seg, outputfile=fn_stl, tempfile=fn_tmp)
        self.assertTrue(op.exists(fn_stl))
        os.remove(fn_stl)
        os.remove(fn_tmp)






if __name__ == '__main__':
    unittest.main()
