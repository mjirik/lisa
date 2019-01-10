# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
List of functions to run
"""
import logging
logger = logging.getLogger(__name__)

import glob
import os.path as op

from . import organ_segmentation


class AutoLisa:
    def __init__(self):
        self.config = None
        pass

    def run_one(self, input_data_path, output_datapath=None):
        import os.path as op
        import io3d.datasets

        # input_data_path = io3d.datasets.join_path("jatra_5mm/")


        oseg = organ_segmentation.OrganSegmentation(
            # input_annotation_file=input_annotation_file,
            datapath=input_data_path,
            get_series_number_callback="guess for liver",
            # output_annotation_file="output.json",
            autocrop=False,
            run_list=["get_body_navigation_structures_precise", "save_outputs"],
            output_datapath=output_datapath

        )
        oseg.make_run()
        # oseg.sliver_compare_with_other_volume_from_file("file.pklz")

    def run_in_paths(self, path):
        if type(path) in (list, tuple):
            files = path
        else:
            files = glob.glob(path)
        for pth in files:
            self.run_one(pth)






