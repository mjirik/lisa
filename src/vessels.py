#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import sys
sys.path.append("./src/")
import pdb
#  pdb.set_trace();

import scipy.io

import logging
logger = logging.getLogger(__name__)


import argparse
import numpy as np

#Ahooooooj


def vesselSegmentation(data, segmentation, threshold=1185, dataFiltering=False, nObj=1, voxelsizemm=[[1],[1],[1]]):
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
    if data.shape != segmentation.shape:
        raise Exception('Input size error','Shape if input data and segmentation must be same')

    return segmentation

# --------------------------tests-----------------------------
class Tests(unittest.TestCase):
    def test_t(self):
        pass
    def setUp(self):
        """ Nastavení společných proměnných pro testy  """
        datashape = [220,115,30]
        self.datashape = datashape
        self.rnddata = np.random.rand(datashape[0], datashape[1], datashape[2])
        self.segmcube = np.zeros(datashape)
        self.segmcube[130:190, 40:90,5:15] = 1

    def test_same_size_input_and_output(self):
        """Funkce testuje stejnost vstupních a výstupních dat"""
        outputdata = vesselSegmentation(self.rnddata,self.segmcube)
        self.assertEqual(outputdata.shape, self.rnddata.shape)


#
#    def test_different_data_and_segmentation_size(self):
#        """ Funkce ověřuje vyhození výjimky při různém velikosti vstpních
#        dat a segmentace """
#        pdb.set_trace();
#        self.assertRaises(Exception, vesselSegmentation, (self.rnddata, self.segmcube[2:,:,:]) )
#
        
        
# --------------------------main------------------------------
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
            help='*.mat file with variables "data", "segmentation" and "threshod"')
    parser.add_argument('-d', '--debug', action='store_true',
            help='run in debug mode')
    parser.add_argument('-t', '--tests', action='store_true', 
            help='run unittest')
    parser.add_argument('-o', '--outputfile', type=str,
        default='output.mat',help='output file name')
    args = parser.parse_args()


    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.tests:
        # hack for use argparse and unittest in one module
        sys.argv[1:]=[]
        unittest.main()


#   load all 
    mat = scipy.io.loadmat(args.filename)
    logger.debug( mat.keys())

    # load specific variable
    matthreshold = scipy.io.loadmat(args.filename, variable_names=['threshold'])

    logger.debug(matthreshold['threshold'][0][0])


    # zastavení chodu programu pro potřeby debugu, 
    # ovládá se klávesou's','c',... 
    # zakomentovat
    pdb.set_trace();

    # zde by byl prostor pro ruční (interaktivní) zvolení prahu z klávesnice 
    #tě ebo jinak

    output = vesselSegmentation(mat['data'],mat['segmentation'], mat['threshold'], mat['voxelsizemm'] )
    scipy.io.savemat(args.outputfile,{'vesselSegm':output})

