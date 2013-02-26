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
#import numpy as np

# ----------------- my scripts --------
#import dcmreaddata
#import pycat
import argparse
#import py3DSeedEditor

import segmentation
import inspector
import organ_segmentation
import misc




if __name__ == "__main__":

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(description='Segment vessels from liver')
    parser.add_argument('-dd','--dcmdir',
            default=None,
            help='path to data dir')
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

        args.dcmdir = '../sample_data/matlab/examples/sample_data/DICOM/digest_article/'

    #else:
    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
        #data3d, metadata = dcmreaddata.dcm_read_from_dir()


    oseg = organ_segmentation.OrganSegmentation(args.dcmdir, working_voxelsize_mm = 6, autocrop = True, autocrop_margin = [5,5,5])

    oseg.interactivity()

    #print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )

    #igc = pycat.ImageGraphCut(data3d, zoom = 0.5)
    #igc.interactivity()


    #igc.make_gc()
    #igc.show_segmentation()

    # volume
    #volume_mm3 = np.sum(oseg.segmentation > 0) * np.prod(oseg.voxelsize_mm)

    print ( "Volume " + str(oseg.get_segmented_volume_size_mm3()/1000000.0) + ' [l]' )
    import py3DSeedEditor
    # pyed = py3DSeedEditor.py3DSeedEditor(oseg.orig_scale_segmentation)
    #pyed.show()
# information about crop
    cri = oseg.crinfo
    oseg.data3d = oseg.data3d[cri[0][0]:cri[0][1],cri[1][0]:cri[1][1],cri[2][0]:cri[2][1]]
    pyed = py3DSeedEditor.py3DSeedEditor(oseg.data3d, contour = oseg.orig_scale_segmentation)
    pyed.show()
    # oseg.orig_scale_segmentation

    outputTmp = segmentation.vesselSegmentation(
        oseg.data3d,
        segmentation = oseg.orig_scale_segmentation,
        threshold = -1,
        inputSigma = 0.15,
        dilationIterations = 2,
        nObj = 1,
        dataFiltering = True,
        interactivity = False,
        binaryClosingIterations = 1,
        binaryOpeningIterations = 1)

    inspect = inspector.inspector(outputTmp)
    output = inspect.run()
    del(inspect)

# segmentation labeling
    slab={}
    slab['none'] = 0
    slab['liver'] = 1
    slab['porta'] = 2

    data = {}
    data['data3d'] = oseg.data3d
    data['crinfo'] = oseg.crinfo
    data['segmentation'] = oseg.orig_scale_segmentation
    data['segmentation'][output==1] = slab['porta']
    data['slab'] = slab

#    import pdb; pdb.set_trace()
    pyed = py3DSeedEditor.py3DSeedEditor(data['data3d'],  contour=data['segmentation']==slab['porta'])
    pyed.show()

    savestring = raw_input ('Save output data? (y/n): ')
    #sn = int(snstring)
    if savestring in ['Y','y']:

        misc.obj_to_file(data, "out", filetype = 'pickle')
