#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)
import unittest
from nose.plugins.attrib import attr
import os.path as op

import lisa
import io3d.datasets

path_to_script = op.dirname(op.abspath(__file__))

class DicomWebViewJsonTest(unittest.TestCase):
    @unittest.skip("waiting for finishing the import function")
    def test_json_ircad_import(self):
        input_annotation_file = op.join(path_to_script, "test_dwv_3Dircadb1.12.json")
        input_data_path = io3d.datasets.join_path("3Dircadb1.12/PATIENT_DICOM/")

        oseg = lisa.organ_segmentation.OrganSegmentation(
            input_annotation_file=input_annotation_file,
            datapath=input_data_path,
            output_annotation_file="output.json",
            autocrop=False,
        )
        oseg.make_run()

    @attr("slow")
    def test_json_jatra_5mm_import(self):

        input_annotation_file = op.join(path_to_script, "test_dwv_jatra_5mm.json")
        input_data_path = io3d.datasets.join_path("jatra_5mm/")
        output_file = "output.json"

        oseg = lisa.organ_segmentation.OrganSegmentation(
            input_annotation_file=input_annotation_file,
            datapath=input_data_path,
            output_annotation_file=output_file,
            autocrop=False,

        )
        oseg.make_run()
        self.assertTrue(op.exists(output_file))

    def test_json_jatra_5mm_import_fast(self):
        """
        Workaround to skkip slowest part of test which is export.
        :return:
        """

        input_annotation_file = op.join(path_to_script, "test_dwv_jatra_5mm.json")
        input_data_path = io3d.datasets.join_path("jatra_5mm/")
        output_file = "output.json"

        oseg = lisa.organ_segmentation.OrganSegmentation(
            input_annotation_file=input_annotation_file,
            datapath=input_data_path,
            output_annotation_file=None,
            autocrop=False,

        )

        oseg.make_run()
        oseg.output_annotaion_file = output_file
        # set small segmentation to make export faster
        oseg.segmentation[:] = 0
        oseg.segmentation[10:20, 10:20, 10:20] = 1
        oseg.segmentation[17:20, 17:20, 17:20] = 2
        oseg.json_annotation_export()
        self.assertTrue(op.exists(output_file))
