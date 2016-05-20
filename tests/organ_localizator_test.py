#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from nose.plugins.attrib import attr
import numpy as np

import os.path as op
from lisa import organ_localizator
import lisa.dataset
import io3d
import io3d.datareader


class LocalizationTests(unittest.TestCase):
    @attr('slow')
    def test_training(self):
        sample_data_path = lisa.dataset.sample_data_path()
        organ_localizator.train_liver_localizator_from_sliver_data(
            "liver.ol.p",
            sliver_reference_dir=sample_data_path,
            orig_pattern="*orig*001.mhd",
            ref_pattern="*seg*001.mhd",

        )

        ol = organ_localizator.OrganLocalizator()
        ol.load("liver.ol.p")

        dr = io3d.DataReader()
        data3d, metadata = dr.Get3DData(op.join(sample_data_path , 'liver-orig001.mhd'))
        out = ol.predict(data3d, metadata['voxelsize_mm'])


        seg, metadata = dr.Get3DData(op.join(sample_data_path , 'liver-seg001.mhd'))

        err = np.sum(np.abs(seg-out))
        # import sed3
        # sed3.show_slices(out-seg, out-seg, slice_step=20)
        # self.assertEqual(True, False)
        # less then 10% error expected
        self.assertGreater(np.prod(seg.shape)*0.1, err)


if __name__ == '__main__':
    unittest.main()
