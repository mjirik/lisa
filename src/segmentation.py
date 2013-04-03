#
# -*- coding: utf-8 -*-
"""
================================================================================
Name:        segmentation
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky
Email:		 volkovinsky.pavel@gmail.com

Created:     08.11.2012
Copyright:   (c) Pavel Volkovinsky
================================================================================
"""

import unittest
import sys
sys.path.append("../src/")
sys.path.append("../extern/")

import uiThreshold
import uiBinaryClosingAndOpening

import numpy
import numpy.matlib
import scipy
import scipy.io
import scipy.misc
import scipy.ndimage

import logging
logger = logging.getLogger(__name__)

import argparse

# Import garbage collector
import gc as garbage

"""
Vessel segmentation z jater.
    input:
        data - CT (nebo MRI) 3D data
        segmentation - zakladni oblast pro segmentaci, oznacena struktura veliksotne totozna s "data",
            kde je oznaceni (label) jako:
                1 jatra,
                -1 zajimava tkan (kosti, ...)
                0 jinde
        threshold - prah
        voxelsizemm - (vektor o hodnote 3) rozmery jednoho voxelu
        inputSigma - pocatecni hodnota pro prahovani
        dilationIterations - pocet operaci dilation nad zakladni oblasti pro segmentaci ("segmantation")
        dilationStructure - struktura pro operaci dilation
        nObj - oznacuje, kolik nejvetsich objektu se ma vyhledat - pokud je rovno 0 (nule), vraci cela data
        dataFiltering - oznacuje, jestli maji data byt filtrovana (True) nebo nemaji byt
            nebo filtrovana (False) (== uz jsou filtrovana)

    returns:
        ---
"""
def vesselSegmentation(data, segmentation = -1, threshold = -1, voxelsizemm = [1,1,1], inputSigma = -1,
dilationIterations = 0, dilationStructure = None, nObj = 1, dataFiltering = True,
interactivity = True, binaryClosingIterations = 0, binaryOpeningIterations = 0):

    print('Pripravuji data...')

    ## Pokud jsou data nefiltrovana, tak nasleduje filtrovani.
    if(dataFiltering):
        voxel = numpy.array(voxelsizemm)

        ## Kalkulace objemove jednotky (voxel) (V = a*b*c).
        voxel1 = voxel[0]#[0]
        voxel2 = voxel[1]#[0]
        voxel3 = voxel[2]#[0]
        voxelV = voxel1 * voxel2 * voxel3

        ## number je zaokrohleny 2x nasobek objemove jednotky na 2 desetinna mista.
        ## number stanovi doporucenou horni hranici parametru gauss. filtru.
        number = (numpy.round((2 * voxelV**(1.0/3.0)), 2))

        ## Operace dilatace (dilation) nad samotnymi jatry ("segmentation").
        if(dilationIterations > 0.0):
            segmentation = scipy.ndimage.binary_dilation(input = segmentation,
                structure = dilationStructure, iterations = dilationIterations)

        ## Ziskani datove oblasti jater (bud pouze jater nebo i jejich okoli - zalezi,
        ## jakym zpusobem bylo nalozeno s operaci dilatace dat).
        preparedData = data * (segmentation == 1)
        del(data)
        del(segmentation)

        ## Filtrovani (rozmazani) a prahovani dat.
        if(inputSigma == -1):
            inputSigma = number
        if(inputSigma > number):
            inputSigma = number
        uiT = uiThreshold.uiThreshold(preparedData, voxel, threshold, interactivity, number, inputSigma, nObj)
        filteredData = uiT.run()
        del(uiT)
        garbage.collect()

    ## Binarni otevreni a uzavreni.
    uiB = uiBinaryClosingAndOpening.uiBinaryClosingAndOpening(filteredData, binaryClosingIterations,
        binaryOpeningIterations, interactivity)
    output = uiB.run()
    del(uiB)
    garbage.collect()


    ## Vraceni matice
    if(dataFiltering == False):
        ## Data vstoupila jiz filtrovana, tudiz neprosly nalezenim nejvetsich objektu.
        return getBiggestObjects(data = output, N = nObj)
    else:
        ## Data vstoupila nefiltrovana, tudiz jiz prosly nalezenim nejvetsich objektu.
        return output

"""
Vraceni N nejvetsich objektu.
    input:
        data - data, ve kterych chceme zachovat pouze nejvetsi objekty
        N - pocet nejvetsich objektu k vraceni

    returns:
        data s nejvetsimi objekty
"""
def getBiggestObjects(data, N):

    ## Oznaceni dat.
    ## labels - oznacena data.
    ## length - pocet rozdilnych oznaceni.
    labels, length = scipy.ndimage.label(data)

    ## Podminka mnozstvi objektu.
    maxN = 250
    if(length > maxN):
        print('Varovani: Existuje prilis mnoho objektu! (' + str(length) + ')')

    ## Soucet oznaceni z dat.
    arrayLabelsSum, arrayLabels = areaIndexes(labels, length)

    ## Serazeni poli pro vyber nejvetsich objektu.
    ## Pole arrayLabelsSum je serazeno od nejvetsi k nejmensi cetnosti.
    ## Pole arrayLabels odpovida prislusnym oznacenim podle pole arrayLabelsSum.
    arrayLabelsSum, arrayLabels = selectSort(list1 = arrayLabelsSum, list2 = arrayLabels)

    ## Osetreni neocekavane situace.
    if(N > len(arrayLabels)):
        print('Pocet nejvetsich objektu k vraceni chcete vetsi nez je oznacenych oblasti!')
        print('Redukuji pocet nejvetsich objektu k vraceni...')
        N = len(arrayLabels)

    for index1 in range(0, len(arrayLabelsSum)):
        print(str(arrayLabels[index1]) + " " + str(arrayLabelsSum[index1]))

    return data == 1

"""
    ## Upraveni dat (ziskani N nejvetsich objektu).
    search = N
    if (sys.version_info[0] < 3):
        import copy
        newData = copy.copy(data) # TODO: udelat lepe vynulovani
        newData = newData * 0
    else:
        newData = data.copy()
        newData = newData * 0

    for index in range(0, len(arrayLabels)):
        newData -= (labels == arrayLabels[index])
        if(arrayLabels[index] != 0):
            search -= 1
            if search <= 0:
                break

    ## Uvolneni pameti
    del(labels)
    del(arrayLabels)
    del(arrayLabelsSum)
    garbage.collect()

    return data - newData - 1
"""

"""
Zjisti cetnosti jednotlivych oznacenych ploch (labeled areas)
    input:
        labels - data s aplikovanymi oznacenimi
        num - pocet pouzitych oznaceni

    returns:
        dve pole - prvni sumy, druhe indexy
"""
def areaIndexes(labels, num):

    arrayLabelsSum = []
    arrayLabels = []

    for index in range(0, num):
        arrayLabels.append(index)
        sumOfLabel = numpy.sum(labels == index)
        arrayLabelsSum.append(sumOfLabel)

    return arrayLabelsSum, arrayLabels

"""
Razeni 2 poli najednou (list) pomoci metody select sort
    input:
        list1 - prvni pole (hlavni pole pro razeni)
        list2 - druhe pole (vedlejsi pole) (kopirujici pozice pro razeni podle hlavniho pole list1)

    returns:
        dve serazena pole - hodnoty se ridi podle prvniho pole, druhe "kopiruje" razeni
"""
def selectSort(list1, list2):

    length = len(list1)
    for index in range(0, length):
        min = index
        for index2 in range(index + 1, length):
            if list1[index2] > list1[min]:
                min = index2
        ## Prohozeni hodnot hlavniho pole
        list1[index], list1[min] = list1[min], list1[index]
        ## Prohozeni hodnot vedlejsiho pole
        list2[index], list2[min] = list2[min], list2[index]

    return list1, list2

"""
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

    def test_different_data_and_segmentation_size(self):
        # Funkce ověřuje vyhození výjimky při různém velikosti vstpních
        # dat a segmentace
        pdb.set_trace();
        self.assertRaises(Exception, vesselSegmentation, (self.rnddata, self.segmcube[2:,:,:]) )
"""

"""
Main
"""
if __name__ == "__main__":

    #print('Byl spusten skript.')
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

    op3D = True

    if args.filename == 'lena':
        mat = scipy.misc.lena()
        op3D = False
    else:
        mat = scipy.io.loadmat(args.filename)
        logger.debug(mat.keys())

    #print('Hotovo.')

    #import pdb; pdb.set_trace()
    if(op3D):
        structure = None
        outputTmp = vesselSegmentation(mat['data'], mat['segmentation'], threshold = -1,
            voxelsizemm = mat['voxelsizemm'], inputSigma = 0.15, dilationIterations = 2,
            nObj = 1, dataFiltering = True, interactivity = True, binaryClosingIterations = 1,
            binaryOpeningIterations = 1)
    else:
        outputTmp = vesselSegmentation(data = mat, segmentation = mat)

    import inspector
    inspect = inspector.inspector(outputTmp)
    output = inspect.run()
    del(inspect)
    garbage.collect()

    try:
        cislo = input('Chcete ulozit vystup?\n1 jako ano\ncokoliv jineho jako ne\n')
        if(cislo == '1'):
            print('Ukladam vystup...')
            scipy.io.savemat(args.outputfile, {'data':output})
            print('Vystup ulozen.')

    except Exception:
        print('Nastala chyba!')
        raise Exception

    garbage.collect()
    sys.exit()

# TODO: vraceni nastavenych parametru
# TODO: priprava na testy
# TODO: testy
# TODO: pridani cev (graficke)
