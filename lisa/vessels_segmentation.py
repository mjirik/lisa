#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)

# import apdb
# apdb.set_trace();
# import scipy.io
import numpy as np

# ----------------- my scripts --------
import argparse

import segmentation
import misc

# Import garbage collector
import gc as garbage

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
    parser.add_argument('-bg', '--biggest', action='store_true',
            help='find biggest object')
    parser.add_argument('-ed', '--exampledata', action='store_true',
            help='run unittest')
    parser.add_argument('-i', '--inputfile',  default=None,
            help='input file from organ_segmentation')
    parser.add_argument('-ii', '--defaultinputfile',  action='store_true',
            help='"organ.pkl" as input file from organ_segmentation')
    parser.add_argument('-o', '--outputfile',  default=None,
            help='output file')
    parser.add_argument('-oo', '--defaultoutputfile',  action='store_true',
            help='"vessels.pickle" as output file')
    args = parser.parse_args()


    if args.debug:
        logger.setLevel(logging.DEBUG)

    defaultoutputfile =  "vessels.pkl"
    if args.defaultoutputfile:
        args.outputfile = defaultoutputfile

    if args.exampledata:

        args.dcmdir = '../sample_data/matlab/examples/sample_data/DICOM/digest_article/'

    #else:
    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
        #data3d, metadata = dcmreaddata.dcm_read_from_dir()
    if args.defaultinputfile:
        args.inputfile = "organ.pkl"

    if args.inputfile == None:
        oseg = organ_segmentation.OrganSegmentation(args.dcmdir, working_voxelsize_mm = 6, autocrop = True, autocrop_margin_mm = [10,10,10])
        oseg.interactivity()
        # Uvolneni pameti
        garbage.collect()
        print ( "Volume " + str(oseg.get_segmented_volume_size_mm3()/1000000.0) + ' [l]' )
# get data in list
        data = oseg.export()
    else:
        data = misc.obj_from_file(args.inputfile, filetype = 'pickle')

    import sed3
    # pyed = sed3.sed3(oseg.orig_scale_segmentation)
    #pyed.show()
# information about crop
    #cri = oseg.crinfo
    #oseg.data3d = oseg.data3d[cri[0][0]:cri[0][1],cri[1][0]:cri[1][1],cri[2][0]:cri[2][1]]
    #pyed = sed3.sed3(oseg.data3d, contour = oseg.orig_scale_segmentation)

    print 'slab', data['slab']
    #import ipdb; ipdb.set_trace()  # BREAKPOINT
    #pyed = sed3.sed3(data['data3d'], contour = data['segmentation'])
    #pyed.show()
    #import pdb; pdb.set_trace()

    outputTmp = segmentation.vesselSegmentation(
        data['data3d'],
        segmentation = data['segmentation'],
        #segmentation = oseg.orig_scale_segmentation,
        threshold = -1,
        inputSigma = 0.15,
        dilationIterations = 2,
        nObj = 1,
        biggestObjects = args.biggest,
#        dataFiltering = True,
        interactivity = True,
        binaryClosingIterations = 2,
        binaryOpeningIterations = 0)

    # Uvolneni pameti
    garbage.collect()

    import inspector
    inspect = inspector.inspector(outputTmp)
    output = inspect.run()

    # Uvolneni pameti
    del(inspect)
    garbage.collect()

    #pyed = sed3.sed3(outputTmp)
    #pyed.show()
# segmentation labeling
    #slab={}
    data['slab']['none'] = 0
    data['slab']['liver'] = 1
    data['slab']['porta'] = 2


    #print np.max(output)
    #import pdb; pdb.set_trace()
    #data = {}
    #data['data3d'] = oseg.data3d
    #data['crinfo'] = oseg.crinfo
    #data['segmentation'] = oseg.segmentation
    data['segmentation'][output] = data['slab']['porta']
    #data['slab'] = slab


    pyed = sed3.sed3(data['data3d'],  contour=data['segmentation']==data['slab']['porta'])
    pyed.show()

    #pyed = sed3.sed3(data['segmentation'])
    #pyed.show()
    # Uvolneni pameti
    garbage.collect()


    if args.outputfile == None:

        savestring = raw_input ('Save output data? (y/n): ')
        #sn = int(snstring)
        if savestring in ['Y','y']:
            pth, filename = os.path.split(os.path.normpath(args.inputfile))

            misc.obj_to_file(data,
                             defaultoutputfile + '-' + filename,
                             filetype = 'pickle')
    else:
        misc.obj_to_file(data, args.outputfile, filetype = 'pickle')

