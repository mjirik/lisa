#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from nose.plugins.attrib import attr

from lisa import organ_localizator
import lisa.data

class LocalizationTests(unittest.TestCase):
    # @attr('interactive')
    def test_training(self):
        sample_data_path = lisa.data.sample_data_path()
        organ_localizator.train_liver_localizator_from_sliver_data(
            "liver.ol.p",
            sliver_reference_dir=sample_data_path,
            orig_pattern="*orig*001.mhd",
            ref_pattern="*seg*001.mhd",

        )

        ol = organ_localizator.OrganLocalizator()
        ol.load("liver.ol.p")


        # self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
