#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Generator of histology report

"""
import logging
logger = logging.getLogger(__name__)

import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))


import argparse
import numpy as np
import datawriter

import py3DSeedEditor as se


class TreeVolumeGenerator:
    def __init__(self):
        self.data = None
        self.data3d = None
        self.voxelsize_mm = [1, 1, 1]
        self.shape = None

    def importFromYaml(self, filename):
        data = misc.obj_from_file(filename=filename, filetype='yaml')
        self.data = data

    def generateTree(self):
        """
        Funkce na vygenerování objemu stromu ze zadaných dat.


        """
        #self.data3d = něco geniálního :-)

    def saveToFile(self, outputfile):
        dw = datawriter.DataWriter()
        dw.Write3DData(self.data3d, outputfile)


if __name__ == "__main__":
    import misc
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    #ch = logging.StreamHandler()
    #logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(
        description='Histology analyser reporter'
    )
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        help='input file, yaml file'
    )
    parser.add_argument(
        '-o', '--outputfile',
        default=None,
        help='output file, .raw, .dcm, .tiff, given by extension '
    )
    parser.add_argument(
        '-vs', '--voxelsize',
        default=[1.0, 1.0, 1.0],
        help='size of voxel'
    )
    parser.add_argument(
        '-ds', '--datashape',
        default=[100, 100, 100],
        help='size of output data in pixels for each axis'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    hr = TreeVolumeGenerator()
    hr.importFromYaml(args.inputfile)
    hr.voxelsize_mm = args.voxelsize
    hr.shape = args.datashape
    hr.generateTree()
#vizualizace
    se.py3DSeedEditor(hr.data3d)
    se.show()
#ukládání do souboru
    hr.saveToFile(args.outputfile)
