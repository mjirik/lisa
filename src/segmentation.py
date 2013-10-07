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

# TODO: Podpora "seeds" - vraceni specifickych objektu
# TODO: Udelat lepe vraceni nejvetsich (nejvetsiho) objektu (muze vzniknout problem s cernou oblasti)
# TODO: Bylo by dobre zavest paralelizmus - otazka jak a kde - neda se udelat vsude, casem si to zjistit - urcite pred bakalarskou praci - asi v ramci PRJ5 nebo az mi bude o prazdninach chybet projekt

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

"""
Vessel segmentation z jater.
    input:
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

    returns:
        filtrovana data
"""
def vesselSegmentation(
        data,
        segmentation = -1,
        threshold = -1,
        voxelsize_mm = [1,1,1],
        inputSigma = -1,
        dilationIterations = 0,
        dilationStructure = None,
        nObj = 1,
        biggestObjects = True,
        seeds = None,
        interactivity = True,
        binaryClosingIterations = 1,
        binaryOpeningIterations = 1
        ):

    print('Pripravuji data...')

    if ( nObj < 1 ) :
        nObj = 1

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
    preparedData = data * (segmentation == 1)
    del(data)
    del(segmentation)

    ## Nastaveni rozmazani a prahovani dat.
    if(inputSigma == -1):
        inputSigma = number
    if(inputSigma > number):
        inputSigma = number

    if biggestObjects is not True:
        print('Nyni si levym nebo pravym tlacitkem mysi (klepnutim nebo tazenim) oznacte specificke oblasti k vraceni.')
        import py3DSeedEditor
        pyed = py3DSeedEditor.py3DSeedEditor(preparedData)
        pyed.show()
        seeds = pyed.seeds

        ## Zkontrolovat, jestli uzivatel neco vybral - nejaky item musi byt ruzny od nuly
        if (seeds != 0).any() == False:
            seeds = None
        else:
            seeds = seeds.nonzero() ## seeds je n-tice poli indexu nenulovych prvku   =>   item krychle je == krychle[ seeds[0][x], seeds[1][x], seeds[2][x] ]

    ## Samotne filtrovani.
    uiT = uiThreshold.uiThreshold(preparedData, voxel, threshold,
        interactivity, number, inputSigma, nObj, biggestObjects, binaryClosingIterations,
        binaryOpeningIterations, seeds)
    output = uiT.run()
    del(uiT)
    garbage.collect()

    ## Vraceni matice
    return output

    """
    if(dataFiltering == False):
        ## Data vstoupila jiz filtrovana, tudiz neprosly nalezenim nejvetsich objektu.
        return getPriorityObjects(data = output, N = nObj)
    else:
        ## Data vstoupila nefiltrovana, tudiz jiz prosly nalezenim nejvetsich objektu.
        return output
    """

"""
Vraceni N nejvetsich objektu.
    input:
        data - data, ve kterych chceme zachovat pouze nejvetsi objekty
        nObj - pocet nejvetsich objektu k vraceni
        seeds - dvourozmerne pole s umistenim pixelu, ktere chce uzivatel vratit (odpovidaji matici "data")

    returns:
        data s nejvetsimi objekty
"""
def getPriorityObjects(data, nObj, seeds = None):

    ## Oznaceni dat.
    ## labels - oznacena data.
    ## length - pocet rozdilnych oznaceni.
    dataLabels, length = scipy.ndimage.label(data)

    ## Podminka maximalniho mnozstvi objektu.
    maxN = 250
    if ( length > maxN ) :
        print('Varovani: Existuje prilis mnoho objektu! (' + str ( length ) + ')')

    ## Uzivatel si nevybral specificke objekty.
    if (seeds == None) :

        ## Zjisteni nejvetsich objektu
        arrayLabelsSum, arrayLabels = areaIndexes(dataLabels, length)
        arrayLabelsSum, arrayLabels = selectSort(arrayLabelsSum, arrayLabels)

        returning = None
        label = 0
        stop = nObj - 1
        while label <= stop :
            if label >= len(arrayLabels):
               break
            if arrayLabels[label] != 0:
               if returning == None:
                  returning = dataLabels == arrayLabels[label]
               else:
                  returning = returning + dataLabels == arrayLabels[label]
            else:
               stop = stop + 1
            label = label + 1

        return returning
        # Function exit

    else:

        ## Uzivatel si vybral specificke objekty.
        seed = []
        stop = seeds[0].size ## Zjisteni poctu seedu.
        for index in range(0, stop):
            ## Tady se ukladaji labely na mistech, ve kterych kliknul uzivatel.
            tmp = dataLabels[ seeds[0][index], seeds[1][index], seeds[2][index] ]
            if tmp != 0: ## Tady opet pocitam s tim, ze oznaceni nulou pripada cerne oblasti.
                seed.append(tmp)

        ## Pokud existuji vhodne labely, vytvori se nova data k vraceni.
        ## Pokud ne, vrati se cela nafiltrovana data, ktera do funkce prisla (nedojde k vraceni specifickych objektu).
        if len ( seed ) > 0:

            ## Zbaveni se duplikatu.
            seed = list( set ( seed ) )

            ## Vytvoreni vystupu - postupne pricitani dat prislunych specif. labelu.
            returning = dataLabels == seed[0]
            # Debug:
            import inspector
            inspect = inspector.inspector(returning)
            inspect.run()
            del(inspect)
            garbage.collect()
            #======
            for index in range ( 1, len ( seed ) ) :
                returning = returning + dataLabels == seed[index]
                # Debug:
                import inspector
                inspect = inspector.inspector(returning)
                inspect.run()
                del(inspect)
                garbage.collect()
                #======

            return returning
            # Function exit

        else:

            return data
            # Function exit

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
def _main():

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





