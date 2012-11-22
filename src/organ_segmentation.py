#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os
sys.path.append("../extern/pycat/")
sys.path.append("../extern/pycat/extern/py3DSeedEditor/")
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)


#import pdb
#  pdb.set_trace();
#import scipy.io
import numpy as np

# ----------------- my scripts --------
import dcmreaddata
import pycat
import argparse


class organ_segmentation():
    def __init__(self, datadir, workingvoxelsizemm = 1):
        self.datadir = datadir
        self.workingvoxelsizemm = workingvoxelsizemm


    def interactivity(self):
        pass

    def noninteractivity(self, seeds):
        pass
        




class Tests(unittest.TestCase):
    def setUp(self):
        """ Nastavení společných proměnných pro testy  """
        self.assertTrue(True)
    def test_whole_organ_segmentation(self):
        """
        Function uses organ_segmentation object for segmentation
        """
        dcmdir = './../sample_data/matlab/examples/sample_data/DICOM/digest_article/'
        oseg = organ_segmentation(dcmdir, workingvoxelsizemm = 1)
        #oseg.noninteractivity()
        #oseg.set_roi()
        pass

    def test_dicomread_and_graphcut(self):
        """
        Test dicomread module and graphcut module
        """
        #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
        data3d, metadata = dcmreaddata.dcm_read_from_dir('./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')

        print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )

        igc = pycat.ImageGraphCut(data3d, zoom = 0.5)
        seeds = igc.seeds
        seeds[0,:,0] = 1
        seeds[60:66,60:66,5:6] = 2
        igc.noninteractivity(seeds)


        igc.make_gc()
        segmentation = igc.segmentation
        self.assertTrue(segmentation[14, 4, 1] == 0)
        self.assertTrue(segmentation[127, 120, 10] == 1)
        self.assertTrue(np.sum(segmentation==1) > 100)
        self.assertTrue(np.sum(segmentation==0) > 100)
        #igc.show_segmentation()


if __name__ == "__main__":

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(description='Segment vessels from liver')
    parser.add_argument('-d', '--debug', action='store_true',
            help='run in debug mode')
    parser.add_argument('-t', '--tests', action='store_true', 
            help='run unittest')
    args = parser.parse_args()


    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.tests:
        # hack for use argparse and unittest in one module
        sys.argv[1:]=[]
        unittest.main()
        sys.exit() 
    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
    data3d, metadata = dcmreaddata.dcm_read_from_dir()

    print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )

    igc = pycat.ImageGraphCut(data3d, zoom = 0.25)
    igc.interactivity()


    igc.make_gc()
    igc.show_segmentation()
