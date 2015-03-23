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
# from nose.plugins.attrib import attr
import lisa.shape_model as shm
import numpy as np


class ShapeModelTest(unittest.TestCase):

    # @attr('interactive')
    def test_shape_model(self):
        """
        Run shape model
        """
        sm = shm.ShapeModel()
        sm.model_margin = [0, 0, 0]

# train with first model
        sh0 = np.zeros([20, 21, 1])
        sh0[13:19, 7:16] = 1
        sh0[17:19, 12:16] = 0
        sh0[13:15, 13:16] = 0
        sh0_vs = [2, 1, 1]
        sm.train_one(sh0, sh0_vs)


# train with second model
        sh1 = np.zeros([40, 20, 1])
        sh1[16:27, 7:13] = 1
        sh1[23:27, 11:13] = 0
        sh1_vs = [1, 1, 1]
        sm.train_one(sh1, sh1_vs)

        sm.get_model([[15, 25], [10, 25], [0, 1]], [30, 30, 1])
        # print mdl.shape

        # import matplotlib.pyplot as plt
        # import ipdb; ipdb.set_trace()
        # plt.imshow(np.squeeze(sh1))
        # plt.imshow(np.squeeze(sm.model[:, :, 0]))
        # plt.imshow(np.squeeze(mdl))
        # plt.show()
        # import sed3
        # ed = sed3.sed3(sh1)
        # ed.show()


if __name__ == "__main__":
    unittest.main()
