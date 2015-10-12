#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Modul is used for skeleton binary 3D data analysis
"""

import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))

import logging
logger = logging.getLogger(__name__)
import argparse

import io3d

import traceback

import numpy as np
import scipy.ndimage

def vt2esofspy(vesseltree, outputfilename="tracer.txt"):

    import io3d.misc
    if os.path.isfile(vesseltree):
        vt = io3d.misc.obj_from_file(vesseltree)
    else:
        vt = vesseltree
    print vt['general']
    print vt.keys()






if __name__ == "__main__":
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(description='Experiment support')
    parser.add_argument('--get_subdirs', action='store_true',
                        default=None,
                        help='path to data dir')
    parser.add_argument('-o', '--output', default="tracer.txt",
                        help='output file name')
    parser.add_argument('-i', '--input', default=None,
                        help='input')
    args = parser.parse_args()

    if args.get_subdirs:
        if args.output is None:
            args.output = 'experiment_data.yaml'
        get_subdirs(dirpath=args.input, outputfile=args.output)
    vt2esofspy(args.input, outputfilename=args.output)
