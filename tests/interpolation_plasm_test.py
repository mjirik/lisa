#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../../lar-cc/lib/py/"))
import unittest
from nose.plugins.attrib import attr

# import numpy as np


class InterpolationPlasmTest(unittest.TestCase):
    interactiveTests = False
    # interactiveTest = True

    @attr("LAR")
    def test_store_to_SparseMatrix_and_back(self):
        """
        Test has not strong assert part.
        Its passed if any funcion does throw exception.
        """
        from lisa.extern.interpolation_pyplasm import TRIANGLE_DOMAIN
        import pyplasm

        dom2D = TRIANGLE_DOMAIN(5, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertIsInstance(dom2D, list)
        self.assertIsInstance(dom2D[0], pyplasm.xgepy.Hpc)

    # @unittest.skipIf(not interactiveTests, "test is with visualization")
    @attr("LAR")
    def test_complex_sample(self):
        from larcc import INTERVALS, BEZIER, S2, S1, MAP, STRUCT, SKELETON, VIEW
        from lisa.extern.interpolation_pyplasm import TRIANGLE_DOMAIN,\
            TRIANGULAR_COONS_PATCH
        import pyplasm

        BEZIER(S2)
        dom1D = INTERVALS(1)(32) # noqa
        dom2D = TRIANGLE_DOMAIN(32, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Cab0 = BEZIER(S1)([[10, 0, 0], [6, 0, 3], [3, 0, 3], [0, 0, 0]])
        Cbc0 = BEZIER(S1)( # noqa
            [[10, 0, 0], [10, 2, 4], [8, 8, -4], [2, 10, 4], [0, 10, 0]])
        Cbc1 = BEZIER(S2)(
            [[10, 0, 0], [10, 2, 4], [8, 8, -4], [2, 10, 4], [0, 10, 0]])
        Cca0 = BEZIER(S1)([[0, 10, 0], [0, 6, -5], [0, 3, 5], [0, 0, 0]])

        patch = TRIANGULAR_COONS_PATCH([Cab0, Cbc1, Cca0])
        map_input = MAP(patch)(STRUCT(dom2D))
        if self.interactiveTests:
            VIEW(map_input)
            VIEW(SKELETON(1)(map_input))
        self.assertIsInstance(dom2D[0], pyplasm.xgepy.Hpc)
        self.assertIsInstance(map_input, pyplasm.xgepy.Hpc)

if __name__ == "__main__":
    unittest.main()
