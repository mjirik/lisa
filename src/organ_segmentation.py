#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/"))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/extern/py3DSeedEditor/"))
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

import segmentation
import qmisc


def interactive_imcrop(im):

    pass




class OrganSegmentation():
    def __init__(self, datadir, working_voxelsize_mm = 0.25, SeriesNumber = None, autocrop = True, autocrop_margin = [0,0,0], manualroi = False, texture_analysis=None):
        """
        manualroi: manual set of ROI before data processing, there is a 
             problem with correct coordinates
        """
        
        self.datadir = datadir
        if np.isscalar(working_voxelsize_mm):
            self.working_voxelsize_mm = [working_voxelsize_mm] * 3
        else:
            self.working_voxelsize_mm = working_voxelsize_mm


        # TODO uninteractive Serie selection
        self.data3d, self.metadata = dcmreaddata.dcm_read_from_dir(datadir)
        self.voxelsize_mm = self.metadata['voxelsizemm']
        self.autocrop = autocrop
        self.autocrop_margin = autocrop_margin
        self.crinfo = [[0,-1],[0,-1],[0,-1]]
        self.texture_analysis = texture_analysis

# manualcrop
        if manualroi:
# @todo opravit souřadný systém v součinnosti s autocrop
            self.data3d, self.crinfo = qmisc.manualcrop(self.data3d)

        
        if np.isscalar(working_voxelsize_mm):
            working_voxelsize_mm = np.ones([3]) * working_voxelsize_mm


        self.zoom = self.voxelsize_mm/working_voxelsize_mm

        #import pdb; pdb.set_trace()


    def interactivity(self):
        
        #import pdb; pdb.set_trace()
        igc = pycat.ImageGraphCut(self.data3d, zoom = self.zoom)
        igc.gcparams['pairwiseAlpha'] = 30
# version comparison
        from pkg_resources import parse_version
        import sklearn
        if parse_version(sklearn.__version__) > parse_version('0.10'):
            #new versions
            cvtype_name=  'covariance_type'
        else:
            cvtype_name=  'cvtype'

        igc.modelparams = {'type':'gmmsame','params':{cvtype_name:'full', 'n_components':3}}
        igc.interactivity()
        #igc.make_gc()
        #igc.show_segmentation()
        self.segmentation = igc.segmentation
        if self.autocrop == None:
            self.orig_scale_segmentation = igc.get_orig_shape_segmentation()
        else:
            self.orig_scale_segmentation, self.crinfo = igc.get_orig_shape_cropped_segmentation(self.autocrop_margin)

        if not self.texture_analysis == None:
            import texture_analysis
            # doplnit nějaký kód, parametry atd
            self.orig_scale_segmentation = texture_analysis.segmentation(self.data3d, self.orig_scale_segmentation, params = self.texture_analysis)
#
            pass
        #self.prepare_output()
        #self.orig_segmentation = igc.get_orig_shape_segmentation()

    def prepare_output(self):
        pass
        

    def get_segmented_volume_size_mm3(self):
        """
        Compute segmented volume in mm3, based on subsampeled data 
        """
        # neumim napsat typ lip
#        if nptype(self.working_voxelsize_mm) == type(3):
#        #np.isscalar()
#            voxelvolume_mm3 = np.power(self.working_voxelsize_mm,3)
#            #print 'jedna D'
#            
#        elif np.prod(self.working_voxelsize_mm.shape) == 3:
#            voxelvolume_mm3 = np.prod(self.working_voxelsize_mm)
        voxelvolume_mm3 = np.prod(self.working_voxelsize_mm)
        volume_mm3 = np.sum(self.segmentation > 0) * voxelvolume_mm3
        #print voxelvolume_mm3 
        #print volume_mm3
        #import pdb; pdb.set_trace()
        return volume_mm3

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
        oseg = OrganSegmentation(dcmdir, working_voxelsize_mm = 4)

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
    parser = argparse.ArgumentParser(description=
            'Segment vessels from liver \n \npython organ_segmentation.py\n \n python organ_segmentation.py -mroi -vs 0.6')
    parser.add_argument('-dd','--dcmdir',
            default=None,
            help='path to data dir')
    parser.add_argument('-d', '--debug', action='store_true',
            help='run in debug mode')
    parser.add_argument('-vs', '--voxelsizemm',default = 5, type = float,
            help='Insert working voxelsize ')
    parser.add_argument('-mroi', '--manualroi', action='store_true',
            help='manual crop before data processing')
    parser.add_argument('-t', '--tests', action='store_true', 
            help='run unittest')
    parser.add_argument('-tx', '--textureanalysis', action='store_true', 
            help='run with texture analysis')
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

        args.dcmdir = '../sample_data/matlab/examples/sample_data/DICOM/digest_article/'
        
    #else:
    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
        #data3d, metadata = dcmreaddata.dcm_read_from_dir()


    oseg = OrganSegmentation(args.dcmdir, working_voxelsize_mm = args.voxelsizemm, manualroi = args.manualroi, texture_analysis = args.textureanalysis)

    oseg.interactivity()


    #print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )

    #igc = pycat.ImageGraphCut(data3d, zoom = 0.5)
    #igc.interactivity()


    #igc.make_gc()
    #igc.show_segmentation()

    # volume 
    #volume_mm3 = np.sum(oseg.segmentation > 0) * np.prod(oseg.voxelsize_mm)

    print ( "Volume " + str(oseg.get_segmented_volume_size_mm3()/1000000.0) + ' [l]' )

    #output = segmentation.vesselSegmentation(oseg.data3d, oseg.orig_segmentation)
    
