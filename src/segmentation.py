# -*- coding: utf-8 -*-
"""
================================================================================
Name:        segmentation
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky (volkovinsky.pavel@gmail.com)

Created:     08.11.2012
Copyright:   (c) Pavel Volkovinsky 2012
================================================================================
"""

import unittest
import sys
sys.path.append("../src/")
sys.path.append("../extern/")

import uiThreshold
import uiBinaryClosingAndOpening

import numpy
import scipy.io
import scipy.misc
import scipy.ndimage

import logging
logger = logging.getLogger(__name__)

import argparse

"""
================================================================================
vessel segmentation
================================================================================
"""

## data - cela data
## segmentation - zakladni oblast pro segmentaci
## threshold - prah
## voxelsizemm - (vektor o hodnote 3) rozmery jednoho voxelu
## inputSigma - pocatecni hodnota pro prahovani
## dilationIterations - pocet operaci dilation nad zakladni oblasti pro segmentaci ("segmantation")
## dataFiltering - PROZATIM NEPOUZITO - oznacuje, jestli maji data byt filtrovana nebo zda uz jsou filtrovana
## nObj - PROZATIM NEPOUZITO - oznacuje, kolik nejvetsich objektu se ma vyhledat
def vesselSegmentation(data, segmentation, threshold=1185, voxelsizemm=[[1],[1],[1]], inputSigma = -1, dilationIterations = 20, dataFiltering=False, nObj=1):
    """ Volumetric vessel segmentation from liver.
    data: CT (or MRI) 3D data
    segmentation: labeled image with same size as data where label:
    1 mean liver pixels,
    -1 interesting tissuse (bones)
    0 otherwise
    """
    """
    Funkce pracuje z počátku na principu jednoduchého prahování. Nalezne se
    největší souvislý objekt nad stanoveným prahem, Průběžně bude segmentace
    zpřesňována. Bude nutné hledat cévy, které se spojují mimo játra, ale
    ignorovat žebra.
    Proměnné threshold, dataFiltering a nObj se postupně pokusíme eliminovat a
    navrhnout je automaticky.
    """
    ## Kalkulace objemove jednotky (voxel) (V = a*b*c)
    voxel1 = voxelsizemm[0][0]
    voxel2 = voxelsizemm[1][0]
    voxel3 = voxelsizemm[2][0]
    voxelV = voxel1 * voxel2 * voxel3
    print('Voxel size: ', voxelV)
    
    print('Dimenze vstupu: ', numpy.ndim(data))
    ## number je zaokrohleny 1,5 nasobek objemove jednotky na 2 desetinna mista
    number = (numpy.round((1.5 * voxelV), 2))

    ## number stanovi doporucenou horni hranici parametru gauss. filtru
    print('Doporucena horni hranice gaussianskeho filtru: ', number)
    
    ## operace eroze nad samotnymi jatry
    if(dilationIterations >= 0.0):
        segmentation = scipy.ndimage.binary_dilation(input = segmentation, iterations = dilationIterations)
    
    ## Ziskani dat (jater)
    preparedData = data * (segmentation == 1)    
            
    print('Nasleduje filtrovani (rozmazani) a prahovani dat.')
    if(inputSigma == -1):
        inputSigma = number
    if(inputSigma > 2 * number):
        inputSigma = 2 * number
    uiT = uiThreshold.uiThreshold(preparedData, number, inputSigma, voxelV)
    filteredData = uiT.showPlot()
        
    print('Nasleduje binarni otevreni a uzavreni.')
    uiB = uiBinaryClosingAndOpening.uiBinaryClosingAndOpening(filteredData)
    output = uiB.showPlot()
    
    return output

"""
================================================================================
tests
================================================================================

class Tests(unittest.TestCase):
    def test_t(self):
        pass
    def setUp(self):
        #Nastavení společných proměnných pro testy
        datashape = [220,115,30]
        self.datashape = datashape
        self.rnddata = np.random.rand(datashape[0], datashape[1], datashape[2])
        self.segmcube = np.zeros(datashape)
        self.segmcube[130:190, 40:90,5:15] = 1

    def test_same_size_input_and_output(self):
        #Funkce testuje stejnost vstupních a výstupních dat
        outputdata = vesselSegmentation(self.rnddata,self.segmcube)
        self.assertEqual(outputdata.shape, self.rnddata.shape)

"""
#
#    def test_different_data_and_segmentation_size(self):
#        """ Funkce ověřuje vyhození výjimky při různém velikosti vstpních
#        dat a segmentace """
#        pdb.set_trace();
#        self.assertRaises(Exception, vesselSegmentation, (self.rnddata, self.segmcube[2:,:,:]) )
#

"""
================================================================================
main
================================================================================
"""
if __name__ == "__main__":
    
    print('Byl spusten skript.')
    print('Probiha nastavovani...')
    
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    ch = logging.StreamHandler()
    logging.basicConfig(format='%(message)s')

    formatter = logging.Formatter("%(levelname)-5s [%(module)s:%(funcName)s:%(lineno)d] %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    parser = argparse.ArgumentParser(description='Segment vessels from liver')
    parser.add_argument('-f','--filename', type=str,
            default = 'lena',
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
        sys.argv[1:]=[]
        unittest.main()

    print('Nacitam vstup...')
    
    if args.filename == 'lena':
        data = scipy.misc.lena()
    else:
        mat = scipy.io.loadmat(args.filename)
        logger.debug(mat.keys())        
        
    print('Hotovo.')
        
    #import pdb; pdb.set_trace()
    output = vesselSegmentation(mat['data'], mat['segmentation'], mat['threshold'], mat['voxelsizemm'])
    
    try:
        cislo = input('Chcete ulozit vystup?\n1 jako ano\n0 jako ne\n')
        if(cislo == '1'):
            print('Ukladam vystup...')
            scipy.io.savemat(args.outputfile, {'data':output})
            print('Vystup ulozen.')
    
    except Exception:
        print('Stala se chyba!')
        raise Exception
        
    print('Vypinam skript.')    
    
    sys.exit()
    
