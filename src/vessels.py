#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./src/")
import pdb
#  pdb.set_trace();

import scipy.io

import logging
logger = logging.getLogger(__name__)

#import unittest

import argparse

# coputer dependent constant
defaultdatabasedir = '/home/mjirik/data/'
defaultdatatraindir = 'medical/data_orig/jatra-kma/jatra_5mm'
defaultdatatraindir = 'medical/data_orig/51314913'



# TODO vyrobit nevim co
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    ch = logging.StreamHandler()
#   output configureation
    #logging.basicConfig(format='%(asctime)s %(message)s')
    logging.basicConfig(format='%(message)s')

    formatter = logging.Formatter("%(levelname)-5s [%(module)s:%(funcName)s:%(lineno)d] %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)

    logger.addHandler(ch)


    # input parser
    parser = argparse.ArgumentParser(description='Segment vessels from liver')
    parser.add_argument('filename', type=str,
            help='Base dir with data')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--outputfile', type=str,
        default='output.mat',help='output file name')
    args = parser.parse_args()


    if args.debug:
        logger.setLevel(logging.DEBUG)
    logger.debug('input params')
    print args


    #pdb.set_trace();
    #mat = scipy.io.loadmat(args.filename)
    mat = scipy.io.loadmat(args.filename, variable_names=['threshold'])

    print mat['threshold'][0][0]


    # zastavení chodu programu pro potřeby debugu
    pdb.set_trace();
