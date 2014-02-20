# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky
Email:       volkovinsky.pavel@gmail.com

Created:     2012/11/08
Copyright:   (c) Pavel Volkovinsky
-------------------------------------------------------------------------------
"""

import unittest
import sys
sys.path.append("../src/")
sys.path.append("../extern/")

import uiThreshold

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

def vesselSegmentation(data, segmentation = -1, threshold = -1, voxelsize_mm = [1,1,1], inputSigma = -1, 
                       dilationIterations = 0, dilationStructure = None, nObj = 10, biggestObjects = False, 
                       seeds = None, interactivity = True, binaryClosingIterations = 2, 
                       binaryOpeningIterations = 0, smartInitBinaryOperations = True):

    """

    Vessel segmentation z jater.

    Input:
        data - CT (nebo MRI) 3D data
        segmentation - zakladni oblast pro segmentaci, oznacena struktura se stejnymi rozmery jako "data",
            kde je oznaceni (label) jako:
                1 jatra,
                -1 zajimava tkan (kosti, ...)
                0 jinde
        threshold - prah
        voxelsize_mm - (vektor o hodnote 3) rozmery jednoho voxelu
        inputSigma - pocatecni hodnota pro prahovani
        dilationIterations - pocet operaci dilation nad zakladni oblasti pro segmentaci ("segmantation")
        dilationStructure - struktura pro operaci dilation
        nObj - oznacuje, kolik nejvetsich objektu se ma vyhledat - pokud je rovno 0 (nule), vraci cela data
        getBiggestObjects - moznost, zda se maji vracet nejvetsi objekty nebo ne
        seeds - moznost zadat pocatecni body segmentace na vstupu. Je to matice o rozmerech jako data. Vsude nuly, tam kde je oznaceni jsou jednicky.
        interactivity - nastavi, zda ma nebo nema byt pouzit interaktivni mod upravy dat
        binaryClosingIterations - vstupni binary closing operations
        binaryOpeningIterations - vstupni binary opening operations
        smartInitBinaryOperations - logicka hodnota pro smart volbu pocatecnich hodnot binarnich operaci (bin. uzavreni a bin. otevreni)

    Output:
        filtrovana data

    """

    dim = numpy.ndim(data)
    print 'Dimenze vstupnich dat: ' + str(dim)
    if (dim < 2) or (dim > 3):
        print 'Nepodporovana dimenze dat!'
        print 'Ukonceni funkce!'
        return None

    if seeds == None:
        print 'Funkce spustena bez prioritnich objektu!'

    if biggestObjects:
        print 'Funkce spustena s vracenim nejvetsich objektu => nebude mozne vybrat prioritni objekty!'

    if ( nObj < 1 ) :
        print 'K vraceni vybran 1 objekt.'
        nObj = 1

    print('Pripravuji data...')

    voxel = numpy.array(voxelsize_mm)

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
    preparedData = (data * (segmentation == 1))#.astype(numpy.float)
    print 'Typ vstupnich dat: ' + str(preparedData.dtype)

#    if preparedData.dtype != numpy.uint8:
#        print 'Data nejsou typu numpy.uint8 => muze dojit k errorum'

    if not numpy.can_cast(preparedData.dtype, numpy.float):
       print 'ERROR: (debug message) Data nejsou takoveho typu, aby se daly prevest na typ "numpy.float" => muze dojit k errorum'
       print 'Ukoncuji funkci!'
       return None

    if (preparedData == False).all():
       print 'ERROR: (debug message) Jsou spatna data nebo segmentacni matice: all is true == data is all false == bad segmentation matrix (if data matrix is ok)'
       print 'Ukoncuji funkci!'
       return None

    del(data)
    del(segmentation)

    ## Nastaveni rozmazani a prahovani dat.
    if(inputSigma == -1):
        inputSigma = number
    if(inputSigma > number):
        inputSigma = number

    if biggestObjects == False and seeds == None:
        print('Nyni si levym nebo pravym tlacitkem mysi (klepnutim nebo tazenim) oznacte specificke oblasti k vraceni.')
        import py3DSeedEditor
        pyed = py3DSeedEditor.py3DSeedEditor(preparedData)
        pyed.show()
        seeds = pyed.seeds

        ## Zkontrolovat, jestli uzivatel neco vybral - nejaky item musi byt ruzny od nuly.
        if (seeds != 0).any() == False:
            print 'Zadne seedy nezvoleny => nejsou prioritni objekty.'
            seeds = None
        else:
            seeds = seeds.nonzero()#seeds * (seeds != 0) ## seeds je n-tice poli indexu nenulovych prvku => item krychle je == krychle[ seeds[0][x], seeds[1][x], seeds[2][x] ]
            print 'Seedu bez nul: ' + str(len(seeds[0]))

    closing = binaryClosingIterations
    opening = binaryOpeningIterations

    if (smartInitBinaryOperations):

        if (seeds == None):

            closing = 5
            opening = 1

        else:

            closing = 2
            opening = 0

    ## Samotne filtrovani.
    uiT = uiThreshold.uiThreshold(preparedData, voxel, threshold,
        interactivity, number, inputSigma, nObj, biggestObjects, closing,
        opening, seeds)
    output = uiT.run()

    del(preparedData)
    del(uiT)
    garbage.collect()

    ## Vypocet binarni matice.
    if output == None:

        print 'Zadna data k vraceni! (output == None)'

    else:

        output[output != 0] = 1

    ## Vraceni matice.
    return output

def getPriorityObjects(data, nObj = 1, seeds = None, debug = False):

    """

    Vraceni N nejvetsich objektu.
        input:
            data - data, ve kterych chceme zachovat pouze nejvetsi objekty
            nObj - pocet nejvetsich objektu k vraceni
            seeds - dvourozmerne pole s umistenim pixelu, ktere chce uzivatel vratit (odpovidaji matici "data")

        returns:
            data s nejvetsimi objekty

    """

    ## Oznaceni dat.
    ## labels - oznacena data.
    ## length - pocet rozdilnych oznaceni.
    dataLabels, length = scipy.ndimage.label(data)

    print 'Olabelovano oblasti: ' + str(length)

    if debug:
       print 'data labels:'
       print dataLabels

    ## Podminka maximalniho mnozstvi objektu.
    maxN = 250
#    if ( length > maxN ) :
#        print('Varovani: Existuje prilis mnoho objektu! (' + str ( length ) + ')')

    ## Uzivatel si nevybral specificke objekty.
    if (seeds == None) :

        ## Zjisteni nejvetsich objektu.
        arrayLabelsSum, arrayLabels = areaIndexes(dataLabels, length)
        ## Serazeni labelu podle velikosti oznacenych dat (prvku / ploch).
        arrayLabelsSum, arrayLabels = selectSort(arrayLabelsSum, arrayLabels)

        returning = None
        label = 0
        stop = nObj - 1

        ## Budeme postupne prochazet arrayLabels a postupne pridavat jednu oblast za druhou (od te nejvetsi - mimo nuloveho pozadi) dokud nebudeme mit dany pocet objektu (nObj).
        while label <= stop :

            if label >= len(arrayLabels):
                break

            if arrayLabels[label] != 0:
                if returning == None:
                    ## "Prvni" iterace
                    returning = data * (dataLabels == arrayLabels[label])
                else:
                    ## Jakakoli dalsi iterace
                    returning = returning + data * (dataLabels == arrayLabels[label])
            else:
                ## Musime prodlouzit hledany interval, protoze jsme narazili na nulove pozadi.
                stop = stop + 1

            label = label + 1

            if debug:
                print (str(label - 1)) + ':'
                print returning

        if returning == None:
           print 'Zadna validni olabelovana data! (DEBUG: returning == None)'

        return returning
        # Function exit
        # Function return: Priority objects

    ## Uzivatel si vybral specificke objekty (seeds != None).
    else:

        ## Zalozeni pole pro ulozeni seedu
        arrSeed = []
        ## Zjisteni poctu seedu.
        stop = seeds[0].size
        tmpSeed = 0
        dim = numpy.ndim(dataLabels)
        for index in range(0, stop):
            ## Tady se ukladaji labely na mistech, ve kterych kliknul uzivatel.
            if dim == 3:
                ## 3D data.
                tmpSeed = dataLabels[ seeds[0][index], seeds[1][index], seeds[2][index] ]
            elif dim == 2:
                ## 2D data.
                tmpSeed = dataLabels[ seeds[0][index], seeds[1][index] ]

            ## Tady opet pocitam s tim, ze oznaceni nulou pripada cerne oblasti (pozadi).
            if tmpSeed != 0:
                ## Pokud se nejedna o pozadi (cernou oblast), tak se novy seed ulozi do pole "arrSeed"
                arrSeed.append(tmpSeed)

        ## Pokud existuji vhodne labely, vytvori se nova data k vraceni.
        ## Pokud ne, vrati se "None" typ. { Deprecated: Pokud ne, vrati se cela nafiltrovana data, ktera do funkce prisla (nedojde k vraceni specifickych objektu). }
        if len(arrSeed) > 0:

            ## Zbaveni se duplikatu.
            arrSeed = list( set ( arrSeed ) )
            if debug:
                print 'seed list:'
                print arrSeed

            print 'Ruznych prioritnich objektu k vraceni: ' + str(len(arrSeed))

            ## Vytvoreni vystupu - postupne pricitani dat prislunych specif. labelu.
            returning = None
            for index in range ( 0, len ( arrSeed ) ) :

                if returning == None:
                    returning = data * (dataLabels == arrSeed[index])
                else:
                    returning = returning + data * (dataLabels == arrSeed[index])

                if debug:
                    print (str(index)) + ':'
                    print returning

            return returning
            # Function exit
            # Function return: Priority objects

        else:

            print 'Zadna validni data k vraceni - zadne prioritni objekty nenalezeny (DEBUG: function getPriorityObjects: len(arrSeed) == 0)'
            return None
            # Function exit
            # Function return: None

def areaIndexes(labels, num):

    """

    Zjisti cetnosti jednotlivych oznacenych ploch (labeled areas)
        input:
            labels - data s aplikovanymi oznacenimi
            num - pocet pouzitych oznaceni

        returns:
            dve pole - prvni sumy, druhe indexy

    """

    arrayLabelsSum = []
    arrayLabels = []
    for index in range(0, num):
        arrayLabels.append(index)
        sumOfLabel = numpy.sum(labels == index)
        arrayLabelsSum.append(sumOfLabel)

    return arrayLabelsSum, arrayLabels

def selectSort(list1, list2):

    """
    Razeni 2 poli najednou (list) pomoci metody select sort
        input:
            list1 - prvni pole (hlavni pole pro razeni)
            list2 - druhe pole (vedlejsi pole) (kopirujici pozice pro razeni podle hlavniho pole list1)

        returns:
            dve serazena pole - hodnoty se ridi podle prvniho pole, druhe "kopiruje" razeni
    """

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

def _main():

    """

    Main

    """

    print('Deprecated - volejte metodu "segmentation.vesselSegmentation()" primo!')
    return

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

    """
    import py3DSeedEditor
    pyed = py3DSeedEditor.py3DSeedEditor(mat['data'], mat['segmentation'])
    pyed.show()
    seeds = pyed.seeds
    for i in seeds:
        if i == 1:
           print 'hell yeah'
    """

    structure = None
    outputTmp = vesselSegmentation(mat['data'], mat['segmentation'], threshold = -1,
        voxelsize_mm = mat['voxelsizemm'], inputSigma = 0.15, dilationIterations = 2,
        nObj = 2, biggestObjects = False, interactivity = True, binaryClosingIterations = 5, binaryOpeningIterations = 1)

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

if __name__ == "__main__":

    _main()





