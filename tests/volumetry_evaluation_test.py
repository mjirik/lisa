# ! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import lisa


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