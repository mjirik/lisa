#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import numpy as np

from lisa import organ_segmentation
import imcut.dcmreaddata as dcmr
import os.path


def main():

    dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
    #data3d, metadata = dcmr.dcm_read_from_dir(dcmdir)
    oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4, manualroi=False)

    oseg.interactivity()

main()
