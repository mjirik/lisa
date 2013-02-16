#
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
""" 
Vessel segmentation z jater.
    data - CT (nebo MRI) 3D data
    segmentation - zakladni oblast pro segmentaci, oznacena struktura veliksotne totozna s "data", 
        kde je oznaceni (label) jako:
            1 jatra,
            -1 zajimava tkan (kosti, ...)
            0 jinde
    ====PROZATIM NEPOUZITO - threshold - prah
    voxelsizemm - (vektor o hodnote 3) rozmery jednoho voxelu
    inputSigma - pocatecni hodnota pro prahovani
    dilationIterations - pocet operaci dilation nad zakladni oblasti pro segmentaci ("segmantation")
    dilationStructure - struktura pro operaci dilation
    nObj - oznacuje, kolik nejvetsich objektu se ma vyhledat - pokud je rovno 0 (nule), vraci cela data
    dataFiltering - oznacuje, jestli maji data byt filtrovana (True) nebo nemaji byt
        nebo filtrovana (False) (== uz jsou filtrovana)
"""
def vesselSegmentation(data, segmentation = -1, threshold = 1185, voxelsizemm = [[1],[1],[1]], inputSigma = -1, 
dilationIterations = 0, dilationStructure = None, nObj = 0, dataFiltering = True):
    """
    Funkce pracuje z počátku na principu jednoduchého prahování. Nalezne se
    největší souvislý objekt nad stanoveným prahem, Průběžně bude segmentace
    zpřesňována. Bude nutné hledat cévy, které se spojují mimo játra, ale
    ignorovat žebra.
    Proměnné threshold, dataFiltering a nObj se postupně pokusíme eliminovat a
    navrhnout je automaticky.
    """
    
    if(dataFiltering):
        voxel = voxelsizemm
        
        ## Kalkulace objemove jednotky (voxel) (V = a*b*c)
        voxel1 = voxel[0][0]
        voxel2 = voxel[1][0]
        voxel3 = voxel[2][0]
        voxelV = voxel1 * voxel2 * voxel3
        #print('Voxel size: ', voxelV)
        
        #print('Dimenze vstupu: ', numpy.ndim(data))
        ## number je zaokrohleny 2x nasobek objemove jednotky na 2 desetinna mista
        ## number stanovi doporucenou horni hranici parametru gauss. filtru
        number = (numpy.round((2 * voxelV), 2))
    
        #print('Doporucena horni hranice gaussianskeho filtru: ', number)
        
        ## Operace dilatace (dilation) nad samotnymi jatry ("segmentation")
        if(dilationIterations > 0.0):
            segmentation = scipy.ndimage.binary_dilation(input = segmentation, structure = dilationStructure, 
                                                                          iterations = dilationIterations)
        
        ## Dokumentace k dilataci ("dilation")
        """
        scipy.ndimage.morphology.binary_dilation:
        
            scipy.ndimage.morphology.binary_dilation(input, structure=None, iterations=1, mask=None, 
                output=None, border_value=0, origin=0, brute_force=False)
            ================================================================================
            input : array_like
                Binary array_like to be dilated. Non-zero (True) elements form the subset to be dilated.
            structure : array_like, optional
                Structuring element used for the dilation. Non-zero elements are considered True. 
                If no structuring element is provided an element is generated with a square connectivity equal to one.
            iterations : {int, float}, optional
                The dilation is repeated iterations times (one, by default). If iterations is less than 1, the dilation 
                is repeated until the result does not change anymore.
            mask : array_like, optional
                If a mask is given, only those elements with a True value at the corresponding mask element are 
                modified at each iteration.
            output : ndarray, optional
                Array of the same shape as input, into which the output is placed. By default, a new array is created.
            origin : int or tuple of ints, optional
                Placement of the filter, by default 0.
            border_value : int (cast to 0 or 1)
                Value at the border in the output array.
        """
        
        ## Ziskani datove oblasti jater (bud pouze jater nebo i jejich okoli - zalezi, jakym zpusobem bylo nalozeno
        ## s operaci dilatace dat)
        preparedData = data * (segmentation == 1)
                
        ## Filtrovani (rozmazani) a prahovani dat
        if(inputSigma == -1):
            inputSigma = number
        if(inputSigma > number):
            inputSigma = number
        uiT = uiThreshold.uiThreshold(preparedData, voxel, number, inputSigma)
        filteredData = uiT.showPlot()
            
        ## Binarni otevreni a uzavreni
        uiB = uiBinaryClosingAndOpening.uiBinaryClosingAndOpening(filteredData)
        output = uiB.showPlot()
    
    else:
        ## Binarni otevreni a uzavreni
        uiB = uiBinaryClosingAndOpening.uiBinaryClosingAndOpening(data)
        output = uiB.showPlot()
    
    ## Operace zjisteni poctu N nejvetsich objektu a jejich nasledne vraceni
    if(nObj > 0):
        return getBiggestObject(output, nObj)
    elif(nObj == 0):
        return output
    elif(nObj < 0):
        print('Chyba! Chcete vracet zaporny pocet objektu, coz neni mozne!')
        print('Nasleduje vraceni upravenych dat (vsech objektu)!')
        return output
    
"""
Vraceni N nejvetsich objektu.
    data - data, ve kterych chceme zachovat pouze nejvetsi objekty
    N - pocet nejvetsich objektu k vraceni
"""
def getBiggestObject(data, N):
    lab, num = scipy.ndimage.label(data)

    
    maxlab = maxAreaIndex(lab, num, N)

    data = (lab == maxlab)
    return data
    
"""
Zjisti cetnosti jednotlivych oznacenych ploch (labeled areas).
Return index of maximum labeled area.
    labels - data s aplikovanymi oznacenimi
    num - pocet pouzitych oznaceni
    N - pocet nejvetsich objektu k vraceni
"""
def maxAreaIndex(labels, num, N):
    
    print(num)
    arrayLabels = []
    arrayLabelsSum = []
    print(len(arrayLabels))
    
    for index in range(0, num):
        arrayLabels.append(index)
        sumOfLabel = numpy.sum(labels == index)
        arrayLabelsSum.append(sumOfLabel)
        
    print(len(arrayLabels))
    print(arrayLabels)
    print(len(arrayLabelsSum))
    
    
    
    max = 0
    maxIndex = -1
    for index in range(1, num):
        maxtmp = numpy.sum(labels == index)
        if(maxtmp > max):
            max = maxtmp
            maxIndex = index

    return maxIndex

"""
================================================================================
tests
================================================================================
"""
"""class Tests(unittest.TestCase):
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
    
    #print('Byl spusten skript.')
    #print('Probiha nastavovani...')
    
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

    #print('Nacitam vstup...')
    
    op3D = True
    
    if args.filename == 'lena':
        op3D = False
        mat = scipy.misc.lena()
    else:
        mat = scipy.io.loadmat(args.filename)
        logger.debug(mat.keys())        
        
    #print('Hotovo.')
        
    """
    structure = mat['segmentation']
    structure[structure <= 0.0] = False    
    structure[structure > 0.0] = True
    print(structure)
    """
    
    #import pdb; pdb.set_trace()
    if(op3D):
        structure = None
        output = vesselSegmentation(mat['data'], mat['segmentation'], mat['threshold'], 
                                                 mat['voxelsizemm'], inputSigma = 0.15, dilationIterations = 1, 
                                                 dilationStructure = structure, nObj = 1, dataFiltering = True) 
    else:
        output = vesselSegmentation(data = mat, segmentation = mat) 
    
    uiB = uiBinaryClosingAndOpening.uiBinaryClosingAndOpening(output)
    outputTmp = uiB.showPlot()
    
    import inspector
    inspect = inspector.inspector(outputTmp)
    output = inspect.showPlot()
    
    try:
        cislo = input('Chcete ulozit vystup?\n1 jako ano\n0 jako ne\n')
        if(cislo == '1'):
            print('Ukladam vystup...')
            scipy.io.savemat(args.outputfile, {'data':output})
            print('Vystup ulozen.')
    
    except Exception:
        print('Stala se chyba!')
        raise Exception
        
    #print('Vypinam skript.')    
    
    sys.exit()
    
