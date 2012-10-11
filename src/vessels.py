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


def vesselSegmentation(data, segmentation, threshold=1185, dataFiltering=False, nObj=1):
    """ Volumetric vessel segmentation from liver.
    data: CT (or MRI) 3D data
    segmentation: labeled image with same size as data where label: 
    1 mean liver pixels,
    -1 interesting tissuse (bones)
    0 othrewise
    """
#   Funkce pracuje z počátku na principu jednoduchého prahování. Nalezne se 
#   největší souvislý objekt nad stanoveným prahem, Průběžně bude segmentace 
#   zpřesňována. Bude nutné hledat cévy, které se spojují mimo játra, ale 
#   ignorovat žebra. 
#   Proměnné threshold, dataFiltering a nObj se postupně pokusíme eliminovat a 
#   navrhnout je automaticky. 
#   threshold: ručně určený práh
#   dataFiltering: označuje, jestli budou data filtrována uvnitř funkce, nebo 
#   již vstupují filtovaná. False znamená, že vstupují filtrovaná.
#   nObj: označuje kolik největších objektů budeme hledat
    return segmentation


# TODO vyrobit nevim co
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
# při vývoji si necháme vypisovat všechny hlášky
    logger.setLevel(logging.DEBUG)

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


    #pdb.set_trace();
    #mat = scipy.io.loadmat(args.filename)
#   load all 
    mat = scipy.io.loadmat(args.filename)
    logger.debug( mat.keys())

    # load specific variable
    matthreshold = scipy.io.loadmat(args.filename, variable_names=['threshold'])

    logger.debug(matthreshold['threshold'][0][0])


    # zastavení chodu programu pro potřeby debugu
    pdb.set_trace();

    vesselSegmentation(mat['data'],mat['segmentation'], mat['threshold'] )

