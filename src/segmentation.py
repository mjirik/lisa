# -*- coding: utf-8 -*-
"""
================================================================================
Name:        uiThreshold
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky (volkovinsky.pavel@gmail.com)

Created:     08.11.2012
Copyright:   (c) Pavel Volkovinsky 2012
Licence:     <your licence>
================================================================================
"""

import unittest
import sys
sys.path.append("../src/")
sys.path.append("../extern/")
import uiThreshold

import scipy.io
import scipy.misc

import logging
logger = logging.getLogger(__name__)

import argparse

"""
================================================================================
vessel segmentation
================================================================================
"""
def vesselSegmentation(data, segmentation, threshold=1185, dataFiltering=False, nObj=1, voxelsizemm=[[1],[1],[1]]):
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
    threshold: ručně určený práh
    dataFiltering: označuje, jestli budou data filtrována uvnitř funkce, nebo
    již vstupují filtovaná. False znamená, že vstupují filtrovaná.
    nObj: označuje kolik největších objektů budeme hledat
    """
    
    # 1> data rozmazat gaussian filtrem
    # 2> data upravit prahovanim (uiThreshold)
    # 3> data upravit binary closing a opening
    
    sigma = float(input('Zvolte prosim smerodatnou odchylku gaussianskeho filtru (doporuceny interval je mezi 0.0 do 2.0): '))
    print('Zvoleno: ', sigma)

    preparedData = data * (segmentation == 1)
    filteredData = preparedData
    
    print('Filtruji...')
    scipy.ndimage.filters.gaussian_filter(preparedData, sigma, order = 0, output = filteredData, mode='reflect', cval=0.0)
    print('Filtrovani OK.')
    
    print('Nasleduje prahovani dat.')
    ui = uiThreshold.uiThreshold(filteredData)
    output = ui.showPlot()
    
    print('Nasleduje binarni otevreni a uzavreni.')
    print('Ktere prozatim neni implementovano :-P Paja se musi ucit ;-)')
    print('!! DODELAT !!')

    if data.shape != output.shape:
        raise Exception('Input size error','Shape of input data and segmentation must be same')
    
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
    output = vesselSegmentation(mat['data'], mat['segmentation'], mat['threshold'], mat['voxelsizemm'] )
    
    print('Ukladam vystup...')
    
    scipy.io.savemat(args.outputfile, {'data':output})
        
    print('Vystup ulozen.')
    print('Vypinam skript.')    
    
    sys.exit()
    
