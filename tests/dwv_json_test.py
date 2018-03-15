#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)
import unittest
import os.path as op

import lisa
import io3d.datasets

path_to_script = op.dirname(op.abspath(__file__))

class DicomWebViewJsonTest(unittest.TestCase):
    def test_json_ircad_import(self):
        input_annotation_file = op.join(path_to_script, "test_dwv_3Dircadb1.12.json")
        input_data_path = io3d.datasets.join("3Dircadb1.12/PATIENT_DICOM/")

        lisa.organ_segmentation.OrganSegmentation(
            input_annotation_file=input_annotation_file,
            datapath=input_data_path
        )

        pass
