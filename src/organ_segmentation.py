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


#import apdb
#  apdb.set_trace();
#import scipy.io
import numpy as np

# ----------------- my scripts --------
import dcmreaddata
import pycat
import argparse
import py3DSeedEditor


def interactive_imcrop(im):

    pass




class organ_segmentation():
    def __init__(self, datadir, working_voxelsize_mm = 0.25, SeriesNumber = None):
        
        self.datadir = datadir
        self.working_voxelsize_mm = working_voxelsize_mm

        # TODO uninteractive Serie selection
        self.data3d, self.metadata = dcmreaddata.dcm_read_from_dir(datadir)
        voxelsize_mm = self.metadata['voxelsizemm']
        
        if np.isscalar(working_voxelsize_mm):
            working_voxelsize_mm = np.ones([3]) * working_voxelsize_mm


        self.zoom = voxelsize_mm/working_voxelsize_mm

        #import pdb; pdb.set_trace()


    def interactivity(self):
        igc = pycat.ImageGraphCut(self.data3d, zoom = self.zoom)
        igc.interactivity()
        igc.make_gc()
        igc.show_segmentation()
        pass



    def make_segmentation(self):
        pass


    def ni_set_roi(self, roi_mm):
        pass


    def ni_set_seeds(self, coordinates_mm, label, radius):
        pass

    def im_crop(self, im,  roi_start, roi_stop):
        im_out = im[ \
                roi_start[0]:roi_stop[0],\
                roi_start[1]:roi_stop[1],\
                roi_start[2]:roi_stop[2],\
                ]
        return  im_out


        




class Tests(unittest.TestCase):
    def setUp(self):
        """ Nastavení společných proměnných pro testy  """
        self.assertTrue(True)
    def test_whole_organ_segmentation(self):
        """
        Function uses organ_segmentation object for segmentation
        """
        dcmdir = './../sample_data/matlab/examples/sample_data/DICOM/digest_article/'
        oseg = organ_segmentation(dcmdir, working_voxelsize_mm = 1)

        oseg.interactivity()

        roi_mm = [[3,3,3],[150,150,50]]
        oseg.ni_set_roi()
        coordinates_mm = [[110,50,30], [10,10,10]]
        label = [1,2]
        radius = [5,5]
        oseg.ni_set_seeds(coordinates_mm, label, radius)

        oseg.make_segmentation()

        #oseg.noninteractivity()
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
    parser.add_argument('-ed', '--exampledata', action='store_true', 
            help='run unittest')
    args = parser.parse_args()


    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.tests:
        # hack for use argparse and unittest in one module
        sys.argv[1:]=[]
        unittest.main()
        sys.exit() 

    if args.exampledata:

        data3d, metadata = dcmreaddata.dcm_read_from_dir('../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
    else:
    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
        data3d, metadata = dcmreaddata.dcm_read_from_dir()


    print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )

    igc = pycat.ImageGraphCut(data3d, zoom = 0.5)
    igc.interactivity()


    igc.make_gc()
    igc.show_segmentation()
