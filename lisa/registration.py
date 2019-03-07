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

class Registration:
    """
    Register all data to first given.
    """
    def __init__(self):
        pass

    def set_params(self):
        pass

    def add_data(self, datap):
        """
        Add
        :param datap: dict with key ["data3d", "voxelsize_mm"].
        datap = {
            "data3d" : np.zeros(10, 15, 15),
            "voxelsize_mm" : [1.1, 0.5, 0.5]
        }
        :return:
        """
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
    parser = argparse.ArgumentParser(description='Registration data from different files')
    parser.add_argument('-d', '--debug', action='store_true',
            help='run in debug mode')
    parser.add_argument('-i', '--inputfile',  default=None, nargs="N",
            help='input file or directory with data')
    parser.add_argument('-o', '--outpufile',  default=None,
                        help='output file or directory with data')
    args = parser.parse_args()


    if args.debug:
        logger.setLevel(logging.DEBUG)



    for fn in args.inputfile:
        datap = io3d.read(fn)

    # pyed = sed3.sed3(datap['data3d'], contour=datap['segmentation'] == datap['slab']['porta'])
    # pyed.show()

    #pyed = sed3.sed3(data['segmentation'])
    #pyed.show()
    # Uvolneni pameti


    if args.outputfile == None:
        io3d.write(datap_out, args.outpufile)


