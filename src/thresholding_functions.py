# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky
Email:       volkovinsky.pavel@gmail.com

Created:     2014/02/22
Copyright:   (c) Pavel Volkovinsky
-------------------------------------------------------------------------------
"""

import numpy
import scipy
import scipy.ndimage

def gaussFilter(self):

        """

        Aplikace gaussova filtru.

        """

        ## Zjisteni jakou sigmu pouzit
        if(self.firstRun == True and self.inputSigma >= 0):
            sigma = numpy.round(self.inputSigma, 2)
        else:
            sigma = numpy.round(self.ssigma.val, 2)
        sigmaNew = self.calculateSigma(sigma)

        ## Filtrovani
        scipy.ndimage.filters.gaussian_filter(self.imgFiltering, sigmaNew, order = 0, output = self.imgFiltering, mode = 'nearest')

        if (self.lastSigma != sigma) or (self.threshold < 0):

            if not self.firstRun:

                self.calculateAutomaticThreshold()

            self.lastSigma = sigma

        else:

            self.threshold = self.smin.val

        del(sigmaNew)

def thresholding(self):

        """

        Prahovani podle minimalniho a maximalniho prahu.

        """

        self.imgFiltering = self.imgFiltering * (self.imgFiltering >= self.threshold)
        self.imgFiltering = self.imgFiltering * (self.imgFiltering <= self.smax.val)

        if (self.interactivity == True) :

            self.smin.val = (numpy.round(self.threshold, 2))
            self.smin.valtext.set_text('{}'.format(self.smin.val))
            self.smax.val = (numpy.round(self.smax.val, 2))
            self.smax.valtext.set_text('{}'.format(self.smax.val))

def binaryClosingOpening(self):

        """

        Aplikace binarniho uzavreni a pote binarniho otevreni.

        """

        ## Nastaveni hodnot slideru.
        if (self.interactivity == True) :

            closeNum = int(numpy.round(self.sclose.val, 0))
            openNum = int(numpy.round(self.sopen.val, 0))
            self.sclose.valtext.set_text('{}'.format(closeNum))
            self.sopen.valtext.set_text('{}'.format(openNum))

        else:

            closeNum = self.ICBinaryClosingIterations
            openNum = self.ICBinaryOpeningIterations


        if (closeNum >= 1) or (openNum >= 1):

            ## Vlastni binarni uzavreni.
            if (closeNum >= 1):

                self.imgFiltering = self.numpyDataOnes * scipy.ndimage.binary_closing(self.imgFiltering, iterations = closeNum)

            ## Vlastni binarni otevreni.
            if (openNum >= 1):

                self.imgFiltering = self.numpyDataOnes * scipy.ndimage.binary_opening(self.imgFiltering, iterations = openNum)

def calculateSigma(self, input):

        """

        Spocita novou hodnotu sigma pro gaussovo filtr.

        """

        if (self.voxel[0] == self.voxel[1] == self.voxel[2]):
            return ((5 / self.voxel[0]) * input) / self.voxelV
        else:
            sigmaX = (5.0 / self.voxel[0]) * input
            sigmaY = (5.0 / self.voxel[1]) * input
            sigmaZ = (5.0 / self.voxel[2]) * input

            return (sigmaX, sigmaY, sigmaZ) / self.voxelV

def calculateAutomaticThreshold(self):

        """

        Automaticky vypocet prahu - pokud jsou data bez oznacenych objektu, tak vraci nekolik nejvetsich objektu.
        Pokud jsou ale definovany prioritni seedy, tak je na jejich zaklade vypocitan prah.

        """

        if self.arrSeed != None:

            self.threshold = numpy.round(min(self.arrSeed), 2) - 1
            print('Zjisten automaticky threshold ze seedu (o 1 zmenseny): ' + str(self.threshold))
            return self.threshold

        self.imgUsed = self.data

        ## Hustota hist
        hist_points = 1300
        ## Pocet bodu v primce 1 ( klesajici od maxima )
        pointsFrom = 20 #(int)(hist_points * 0.05)
        ## Pocet bodu v primce 2 ( stoupajici k okoli spravneho thresholdu)
        pointsTo = 20 #(int)(hist_points * 0.1)
        ## Pocet bodu k preskoceni od konce hist
        pointsSkip = (int)(hist_points * 0.025)
        ## hledani maxima: zacina se od 'start'*10 procent delky histu (aby se
        ## preskocili prvni oscilace)
        start = 0.1

        ## hist: funkce(threshold)
        hist, bin_edges = numpy.histogram(self.imgUsed, bins = hist_points)
        ## bin_centers: threshold
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ## last je maximum z hist
        ## init_index je index "last"
        init_index = 0
        last = hist[(int)(len(hist) * start)]
        for index in range((int)(len(hist) * start), len(hist)):
            if(last < hist[index]):
                last = hist[index] ## maximum histu
                init_index = index ## pozice maxima histu

        ## muj_histogram_temp(x+1) == { f(x+1) = hist[x+1] + hist[x] }
        ## stoupajici tendence histogramu
        muj_histogram_temp = []
        muj_histogram_temp.insert(0, hist[0])
        for index in range(1, len(hist)):
            muj_histogram_temp.insert(index, hist[index] + muj_histogram_temp[index - 1])

        ## reverse muj_histogram_temp do muj_histogram
        ## klesajici tendence histogramu
        muj_histogram = muj_histogram_temp[::-1]

        """
        1. primka (od maxima)
        """

        ## Pridani bodu to poli x1 a y1
        ## (klesajici tendence)
        x1 = []
        y1 = []
        place = 0
        for index in range(init_index, init_index + pointsFrom):
            x1.insert(place, bin_centers[index])
            y1.insert(place, muj_histogram[index])
##            print("[ " + str(x1[place]) + ", " + str(y1[place]) + " ]")
            place += 1

        ## Linearni regrese nad x1 a y1
        ## slope == smernice
        ## intercept == posuv
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)

        """
        2. primka (k thresholdu)
        """

        x2 = []
        y2 = []
        place = 0
        for index in range(init_index + pointsFrom + pointsSkip, init_index + pointsFrom + pointsSkip + pointsTo): # len(muj_histogram) - 5 - int(0.1 * len(muj_histogram)), len(muj_histogram) -
                                                                                                                   # 5
            x2.insert(place, bin_centers[index])
            y2.insert(place, muj_histogram[index])
##            print("[ " + str(x2[place]) + ", " + str(y2[place]) + " ]")
            place += 1

        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)

        """

        print("=============")
        print("DEBUG vypisy:\n")

        print("start = " + str(start))
        print("pointsFrom = " + str(pointsFrom))
        print("pointsTo = " + str(pointsTo))
        print("pointsSkip = " + str(pointsSkip) + "\n")

        print("max = " + str(last))
        print("max index = " + str(init_index) + "\n")

        print("slope1 = " + str(slope1))
        print("intercept1 = " + str(intercept1))
        print("r_value1 = " + str(r_value1))
        print("p_value1 = " + str(p_value1))
        print("std_err1 = " + str(std_err1))
        print(str(slope1) + "x + " + str(intercept1) + "\n\n")

        ## //

        print("slope2 = " + str(slope2))
        print("intercept2 = " + str(intercept2))
        print("r_value2 = " + str(r_value2))
        print("p_value2 = " + str(p_value2))
        print("std_err2 = " + str(std_err2))
        print(str(slope2) + "x + " + str(intercept2))

        print("=============")

        """

        self.threshold = (intercept2 - intercept1) / (slope1 - slope2)
        self.threshold = numpy.round(self.threshold, 2)

        print('Zjisten threshold: ' + str(self.threshold))

        """
        muj_histogram_graph = []
        bin_centers_graph = []
        place = 0
        for index in range(len(muj_histogram) / 2, len(muj_histogram)):
            bin_centers_graph.insert(place, bin_centers[index])
            muj_histogram_graph.insert(place, muj_histogram[index])
            place += 1


        matpyplot.figure(figsize = (11, 4))
        matpyplot.plot(bin_centers_graph, muj_histogram_graph, lw = 2)
        #matpyplot.plot([1100*slope1], [1200*slope1], label='one', color='green')
        #matpyplot.plot([1100*slope2], [1200*slope2], label='two', color='blue')
        matpyplot.axvline(self.threshold, color = 'r', ls = '--', lw = 2)
        matpyplot.show()
        """

        self.newThreshold = True

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

        else:

            print 'Zadna validni data k vraceni - zadne prioritni objekty nenalezeny (DEBUG: function getPriorityObjects: len(arrSeed) == 0)'
            return None

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