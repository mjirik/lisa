#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Module for experiment support """


# import funkcí z jiného adresáře
import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
#import featurevector
import argparse

import logging
logger = logging.getLogger(__name__)

#import numpy as np
#import scipy.ndimage


import os
import os.path

# ----------------- my scripts --------
import misc


def get_subdirs(dirpath, wildcard='*', outputfile='experiment_data.yaml'):

    dirlist = []
    if os.path.exists(dirpath):
        logger.info('dirpath = ' + dirpath)
        #print completedirpath
    else:
        logger.error('Wrong path: ' + dirpath)
        raise Exception('Wrong path : ' + dirpath)

    dirpath = os.path.abspath(dirpath)
    #print 'copmpletedirpath = ', completedirpath
    #import pdb; pdb.set_trace()
    dirlist = {
            o:{'abspath': os.path.abspath(os.path.join(dirpath, o))}
            for o in os.listdir(dirpath) if os.path.isdir(
                os.path.join(dirpath, o))
            }
    #import pdb; pdb.set_trace()

   #print [o for o in os.listdir(dirpath) if os.path.isdir(os.path.abspath(o))]


    #    dirlist.append(infile)
    #    #print "current file is: " + infile

    misc.obj_to_file(dirlist, 'experiment_data.yaml', 'yaml')
    return dirlist

## Funkce vrati cast 3d dat. Funkce ma tri parametry :
## 	data - puvodni 3d data
##  sp - vektor udavajici zacatek oblasti, kterou chceme nacist napr [10,20,2]
##       Poradi : [z,y,x]
##  area - (area size) udava velikost oblasti, kterou chceme vratit. Opet
##           vektor stejne jako u sp
## Funkce kontroluje prekroceni velikosti obrazku.


def getArea(data, sp, area):
    if((sp[0] + area[0]) > data.shape[0]):
        sp[0] = data.shape[0] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose Z"
    if((sp[1] + area[1]) > data.shape[1]):
        sp[1] = data.shape[1] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose Y"
    if((sp[2] + area[2]) > data.shape[2]):
        sp[2] = data.shape[2] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose X"
    return data[sp[0]:sp[0] + area[0],
                sp[1]:sp[1] + area[1],
                sp[2]:sp[2] + area[2]]


def setArea(data, sp, area, value):

    if((sp[0] + area[0]) > data.shape[0]):
        sp[0] = data.shape[0] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose Z"
    if((sp[1] + area[1]) > data.shape[1]):
        sp[1] = data.shape[1] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose Y"
    if((sp[2] + area[2]) > data.shape[2]):
        sp[2] = data.shape[2] - area[0] - 1
        print "Funkce getArea() : Byla prekrocena velikost dat v ose X"

    print area, sp
    import ipdb; ipdb.set_trace() # BREAKPOINT

    data[sp[0]:sp[0] + area[0], sp[1]:sp[1] + area[1], sp[2]:sp[2] + area[2]] = value
    return data

if __name__ == "__main__":
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(description=
            'Experiment support')
    parser.add_argument('--get_subdirs', action='store_true',
            default=None,
            help='path to data dir')
    parser.add_argument('-o', '--output', default = None,
            help='output file name')
    parser.add_argument('-i', '--input', default=None,
            help='input')
    args = parser.parse_args()

    if args.get_subdirs:
        if args.output == None:
            args.output =  'experiment_data.yaml'
        get_subdirs(dirpath = args.input, outputfile = args.output)


#    SectorDisplay2__()

