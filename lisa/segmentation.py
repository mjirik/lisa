# -*- coding: utf-8 -*-
"""
    Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

    Author:      Pavel Volkovinsky
    Email:       volkovinsky.pavel@gmail.com

    Created:     2012/11/08
    Copyright:   (c) Pavel Volkovinsky

"""

import sys
sys.path.append("../src/")
sys.path.append("../extern/")
import imtools
from imtools import uiThreshold, thresholding_functions
# import uiThreshold
# import thresholding_functions
import logging
logger = logging.getLogger(__name__)
import numpy
import scipy
import scipy.ndimage


def vesselSegmentation(data, segmentation=-1, threshold=-1,
                       voxelsize_mm=[1, 1, 1],
                       inputSigma=-1, dilationIterations=0,
                       dilationStructure=None, nObj=10, biggestObjects=False,
                       useSeedsOfCompactObjects=False, seeds=None,
                       interactivity=True, binaryClosingIterations=2,
                       binaryOpeningIterations=0,
                       smartInitBinaryOperations=True, returnThreshold=False,
                       binaryOutput=True, returnUsedData=False,
                       qapp=None):
    """

    Vessel segmentation z jater.

    Input:
        data - CT (nebo MRI) 3D data
        segmentation - zakladni oblast pro segmentaci, oznacena struktura se
        stejnymi rozmery jako "data",
            kde je oznaceni (label) jako:
                1 jatra,
                -1 zajimava tkan (kosti, ...)
                0 jinde
        threshold - prah
        voxelsize_mm - (vektor o hodnote 3) rozmery jednoho voxelu
        inputSigma - pocatecni hodnota pro prahovani
        dilationIterations - pocet operaci dilation nad zakladni oblasti pro
            segmentaci ("segmantation")
        dilationStructure - struktura pro operaci dilation
        nObj - oznacuje, kolik nejvetsich objektu se ma vyhledat - pokud je
            rovno 0 (nule), vraci cela data
        biggestObjects - moznost, zda se maji vracet nejvetsi objekty nebo ne
        seeds - moznost zadat pocatecni body segmentace na vstupu. Je to matice
            o rozmerech jako data. Vsude nuly, tam kde je oznaceni jsou jednicky
        interactivity - nastavi, zda ma nebo nema byt pouzit interaktivni mod
            upravy dat
        binaryClosingIterations - vstupni binary closing operations
        binaryOpeningIterations - vstupni binary opening operations
        smartInitBinaryOperations - logicka hodnota pro smart volbu pocatecnich
            hodnot binarnich operaci (bin. uzavreni a bin. otevreni)
        returnThreshold - jako druhy parametr funkce vrati posledni hodnotu
            prahu
        binaryOutput - zda ma byt vystup vracen binarne nebo ne (binarnim
            vystupem se rozumi: cokoliv jineho nez hodnota 0 je hodnota 1)
        returnUsedData - vrati pouzita data

    Output:
        filtrovana data

    """
    # self.qapp = qapp

    dim = numpy.ndim(data)
    logger.debug('Dimenze vstupnich dat: ' + str(dim))
    if (dim < 2) or (dim > 3):
        logger.debug('Nepodporovana dimenze dat!')
        logger.debug('Ukonceni funkce!')
        return None

    if seeds == None:
        logger.debug('Funkce spustena bez prioritnich objektu!')

    if biggestObjects:
        logger.debug(
            'Funkce spustena s vracenim nejvetsich objektu => nebude mozne\
vybrat prioritni objekty!')

    if (nObj < 1):
        nObj = 1

    if biggestObjects:
        logger.debug('Vybrano objektu k vraceni: ' + str(nObj))

    logger.debug('Pripravuji data...')

    voxel = numpy.array(voxelsize_mm)

    # Kalkulace objemove jednotky (voxel) (V = a*b*c).
    voxel1 = voxel[0]  # [0]
    voxel2 = voxel[1]  # [0]
    voxel3 = voxel[2]  # [0]
    voxelV = voxel1 * voxel2 * voxel3

    # number je zaokrohleny 2x nasobek objemove jednotky na 2 desetinna mista.
    # number stanovi doporucenou horni hranici parametru gauss. filtru.
    number = (numpy.round((2 * voxelV ** (1.0 / 3.0)), 2))

    # Operace dilatace (dilation) nad samotnymi jatry ("segmentation").
    if(dilationIterations > 0.0):
        segmentation = scipy.ndimage.binary_dilation(
            input=segmentation, structure=dilationStructure,
            iterations=dilationIterations)

    # Ziskani datove oblasti jater (bud pouze jater nebo i jejich okoli -
    # zalezi, jakym zpusobem bylo nalozeno s operaci dilatace dat).
    preparedData = (data * (segmentation == 1))  # .astype(numpy.float)
    logger.debug('Typ vstupnich dat: ' + str(preparedData.dtype))

#    if preparedData.dtype != numpy.uint8:
#        print 'Data nejsou typu numpy.uint8 => muze dojit k errorum'

    if not numpy.can_cast(preparedData.dtype, numpy.float):
        logger.debug(
            'ERROR: (debug message) Data nejsou takoveho typu, aby se daly \
prevest na typ "numpy.float" => muze dojit k errorum')
        logger.debug('Ukoncuji funkci!')
        return None

    if (preparedData == False).all():
        logger.debug(
            'ERROR: (debug message) Jsou spatna data nebo segmentacni matice: \
all is true == data is all false == bad segmentation matrix (if data matrix is \
ok)')
        logger.debug('Ukoncuji funkci!')
        return None

    del(data)
    del(segmentation)

    # Nastaveni rozmazani a prahovani dat.
    if(inputSigma == -1):
        inputSigma = number
    if(inputSigma > number):
        inputSigma = number

    # seeds = None
    if biggestObjects == False and\
            seeds == None and interactivity == True and threshold == -1:

        logger.debug(
            ('Nyni si levym nebo pravym tlacitkem mysi (klepnutim nebo tazenim)\
 oznacte specificke oblasti k vraceni.'))


        import sed3
        pyed = sed3.sed3qt(preparedData, windowW=400, windowC=50)
        # pyed.show()
        pyed.exec_()

        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        
        seeds = pyed.seeds

        # Zkontrolovat, jestli uzivatel neco vybral - nejaky item musi byt
        # ruzny od nuly.
        if (seeds != 0).any() == False:

            seeds = None
            logger.debug('Zadne seedy nezvoleny => nejsou prioritni objekty.')

        else:

            # seeds * (seeds != 0) ## seeds je n-tice poli indexu nenulovych
            # prvku => item krychle je == krychle[ seeds[0][x], seeds[1][x],
            # seeds[2][x] ]
            seeds = seeds.nonzero()
            logger.debug('Seedu bez nul: ' + str(len(seeds[0])))

    closing = binaryClosingIterations
    opening = binaryOpeningIterations

    if (smartInitBinaryOperations and interactivity):

        if (seeds == None):  # noqa

            closing = 5
            opening = 1

        else:

            closing = 2
            opening = 0

    # Samotne filtrovani
    if interactivity:
        uiT = uiThreshold.uiThresholdQt(
            preparedData, voxel=voxel, threshold=threshold,
            interactivity=interactivity, number=number, inputSigma=inputSigma,
            nObj=nObj, biggestObjects=biggestObjects,
            useSeedsOfCompactObjects=useSeedsOfCompactObjects,
            binaryClosingIterations=closing, binaryOpeningIterations=opening,
            seeds=seeds)

        output = uiT.run()
    else:
        uiT = uiThreshold.uiThreshold(
            preparedData, voxel=voxel, threshold=threshold,
            interactivity=interactivity, number=number, inputSigma=inputSigma,
            nObj=nObj, biggestObjects=biggestObjects,
            useSeedsOfCompactObjects=useSeedsOfCompactObjects,
            binaryClosingIterations=closing, binaryOpeningIterations=opening,
            seeds=seeds)
        output = uiT.run()

    # Vypocet binarni matice.
    if output == None:  # noqa

        logger.debug('Zadna data k vraceni! (output == None)')

    elif binaryOutput:

        output[output != 0] = 1

    # Vraceni matice.
    if returnThreshold:

        if returnUsedData:

            return preparedData, output, uiT.returnLastThreshold()

        else:

            return output, uiT.returnLastThreshold()

    else:

        if returnUsedData:

            return preparedData, output

        else:

            return output
