#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
import io3d
path_to_script = os.path.dirname(os.path.abspath(__file__))
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
from . import segmentation_general

class VesselSegmentation:
    def __init__(self):
        pass

    def set_params(self):
        pass

    def run(self):
        pass

    def get_output(self):
        pass



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
    parser.add_argument('-i', '--inputfile',  default=None,
            help='input file or directory with data')
    args = parser.parse_args()


    if args.debug:
        logger.setLevel(logging.DEBUG)

    defaultoutputfile =  "vessels.pkl"
    if args.defaultoutputfile:
        args.outputfile = defaultoutputfile

    #else:
    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
        #data3d, metadata = dcmreaddata.dcm_read_from_dir()
    datap = io3d.read(args.inputfile)

    import sed3
    # pyed = sed3.sed3(oseg.orig_scale_segmentation)
    #pyed.show()
# information about crop
    #cri = oseg.crinfo
    #oseg.data3d = oseg.data3d[cri[0][0]:cri[0][1],cri[1][0]:cri[1][1],cri[2][0]:cri[2][1]]
    #pyed = sed3.sed3(oseg.data3d, contour = oseg.orig_scale_segmentation)

    print('slab', datap['slab'])
    #import ipdb; ipdb.set_trace()  # BREAKPOINT
    #pyed = sed3.sed3(data['data3d'], contour = data['segmentation'])
    #pyed.show()
    #import pdb; pdb.set_trace()

    outputTmp = segmentation_general.vesselSegmentation(
        datap['data3d'],
        segmentation = datap['segmentation'],
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


    datap['slab']['none'] = 0
    datap['slab']['liver'] = 1
    datap['slab']['porta'] = 2


    #print np.max(output)
    #import pdb; pdb.set_trace()
    #data = {}
    #data['data3d'] = oseg.data3d
    #data['crinfo'] = oseg.crinfo
    #data['segmentation'] = oseg.segmentation
    datap['segmentation'][output] = datap['slab']['porta']
    #data['slab'] = slab


    pyed = sed3.sed3(datap['data3d'], contour=datap['segmentation'] == datap['slab']['porta'])
    pyed.show()

    #pyed = sed3.sed3(data['segmentation'])
    #pyed.show()
    # Uvolneni pameti


    if args.outputfile == None:
        io3d.write(datap, args.outpufile)


