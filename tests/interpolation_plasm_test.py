#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/extern"))
sys.path.append(os.path.join(path_to_script, "../../lar-cc/lib/py/"))
import unittest

# import numpy as np


class InterpolationPlasmTest(unittest.TestCase):
    interactiveTests = True
    # interactivetTest = True

    def test_store_to_SparseMatrix_and_back(self):
        """
        Test has no assert part. Its passed if any funcion does throw exception.
        """
        import interpolation_pyplasm as ip

        dom2D = ip.TRIANGLE_DOMAIN(5, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    @unittest.skipIf(not interactiveTests, "test is with visualization")
    def test_complex_sample(self):
        from larcc import *
        from splines import *
        import interpolation_pyplasm as ip

        # DRAW = COMP([VIEW, STRUCT, MKPOLS])
        # print 'S1 bezier ', S1
        # S0 = SEL(0)
        BEZIER(S2)
        # dom1D = INTERVALS(1)(32)
        dom1D = INTERVALS(1)(6)
        dom2D = ip.TRIANGLE_DOMAIN(32, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Cab0 = BEZIER(S1)([[10, 0, 0], [6, 0, 3], [3, 0, 3], [0, 0, 0]])
        # VIEW(MAP(Cab0)(dom1D))
        Cbc0 = BEZIER(S1)(
            [[10, 0, 0], [10, 2, 4], [8, 8, -4], [2, 10, 4], [0, 10, 0]])
        Cbc1 = BEZIER(S2)(
            [[10, 0, 0], [10, 2, 4], [8, 8, -4], [2, 10, 4], [0, 10, 0]])
        # VIEW(MAP(Cbc0)(dom1D))
        Cca0 = BEZIER(S1)([[0, 10, 0], [0, 6, -5], [0, 3, 5], [0, 0, 0]])
        VIEW(MAP(Cca0)(dom1D))
        map_input = ip.TRIANGULAR_COONS_PATCH([Cab0, Cbc1, Cca0])
        map_function = MAP(map_input)
        print 'Cab0 ', Cab0
        print 'map_input', map_input
        print 'map_i0 ', Cca0

        # out = map_function(dom2D)
        # VIEW(out)
        # VIEW(SKELETON(1)(out))
        # self.assertTrue(np.all(data == data2))

if __name__ == "__main__":
    unittest.main()
