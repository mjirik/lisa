#
# -*- coding: utf-8 -*-
"""
================================================================================
Name:        uiThreshold
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky
Email:		 volkovinsky.pavel@gmail.com

Created:     08.11.2012
Copyright:   (c) Pavel Volkovinsky
================================================================================
"""

import sys
sys.path.append("../src/")
sys.path.append("../extern/")

import logging
logger = logging.getLogger(__name__)

import numpy
import scipy.ndimage
from scipy import stats
import sys

import segmentation

"""
import scipy.misc
import scipy.io

import unittest
import argparse
"""

import matplotlib
import matplotlib.pyplot as matpyplot
from matplotlib.widgets import Slider, Button#, RadioButtons

# Import garbage collector
import gc as garbage

"""
================================================================================
uiThreshold
================================================================================
"""
class uiThreshold:

    """
    Metoda init.
        data - data pro prahovani, se kterymi se pracuje
        number - maximalni hodnota slideru pro gauss. filtrovani (max sigma)
        inputSigma - pocatecni hodnota pro gauss. filtr
        voxelV - objemova jednotka jednoho voxelu
        initslice - PROZATIM NEPOUZITO
        cmap - grey
    """
    def __init__(self, data, voxel, threshold = -1, interactivity = True,
    number = 100.0, inputSigma = -1, nObj = 10, binaryClosingIterations = 1,
    binaryOpeningIterations = 1, cmap = matplotlib.cm.Greys_r):

        print('Spoustim prahovani dat...')

        self.inputDimension = numpy.ndim(data)
        if(self.inputDimension != 3):

            print('Vstup nema 3 dimenze! Ma jich ' + str(self.inputDimension) + '.')
            self.errorsOccured = True
            return

        else:

            self.errorsOccured = False

        self.interactivity = interactivity
        self.cmap = cmap
        self.number = number
        self.inputSigma = inputSigma
        self.threshold = threshold
        self.nObj = nObj
        self.ICBinaryClosingIterations = binaryClosingIterations
        self.ICBinaryOpeningIterations = binaryOpeningIterations

        if (sys.version_info[0] < 3):

            import copy
            self.data = copy.copy(data)
            self.voxel = copy.copy(voxel)
            self.imgChanged = copy.copy(data)
            self.imgFiltering = copy.copy(data)

        else:

            self.data = data.copy()
            self.voxel = voxel.copy()
            self.imgUsed = data.copy()
            self.imgChanged = data.copy()
            self.imgFiltering = data.copy()

        ## Kalkulace objemove jednotky (voxel) (V = a*b*c)
        voxel1 = self.voxel[0]
        voxel2 = self.voxel[1]
        voxel3 = self.voxel[2]
        self.voxelV = voxel1 * voxel2 * voxel3

        if self.interactivity == True:

            ## Minimalni pouzita hodnota prahovani v obrazku
            self.min0 = numpy.amin(self.data)
            ## Maximalni pouzita hodnota prahovani v obrazku
            self.max0 = numpy.amax(self.data)

            self.fig = matpyplot.figure()

            ## Pridani subplotu do okna (do figure)
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)

            ## Upraveni subplotu
            self.fig.subplots_adjust(left = 0.1, bottom = 0.3)

            ## Vykreslit obrazek
            self.im1 = self.ax1.imshow(numpy.amax(self.data, 0), self.cmap)
            self.im2 = self.ax2.imshow(numpy.amax(self.data, 1), self.cmap)
            self.im3 = self.ax3.imshow(numpy.amax(self.data, 2), self.cmap)

            ## Zalozeni mist pro slidery
            self.axcolor = 'white' # lightgoldenrodyellow
            self.axmin = self.fig.add_axes([0.20, 0.24, 0.55, 0.03], axisbg = self.axcolor)
            self.axmax  = self.fig.add_axes([0.20, 0.20, 0.55, 0.03], axisbg = self.axcolor)
            self.axclosing = self.fig.add_axes([0.20, 0.16, 0.55, 0.03], axisbg = self.axcolor)
            self.axopening = self.fig.add_axes([0.20, 0.12, 0.55, 0.03], axisbg = self.axcolor)
            self.axsigma = self.fig.add_axes([0.20, 0.08, 0.55, 0.03], axisbg = self.axcolor)

            ## Vlastni vytvoreni slideru
            self.smin = Slider(self.axmin, 'Minimal threshold', self.min0, self.max0, valinit = self.min0, dragging = False)
            self.smax = Slider(self.axmax, 'Maximal threshold', self.min0, self.max0, valinit = self.max0, dragging = False)
            if(self.ICBinaryClosingIterations >= 1):
                self.sclose = Slider(self.axclosing, 'Binary closing', 0, 100, valinit = self.ICBinaryClosingIterations, dragging = False)
            else:
                self.sclose = Slider(self.axclosing, 'Binary closing', 0, 100, valinit = 0, dragging = False)
            if(self.ICBinaryOpeningIterations >= 1):
                self.sopen = Slider(self.axopening, 'Binary opening', 0, 100, valinit = self.ICBinaryOpeningIterations, dragging = False)
            else:
                self.sopen = Slider(self.axopening, 'Binary opening', 0, 100, valinit = 0, dragging = False)
            self.ssigma = Slider(self.axsigma, 'Sigma', 0.00, self.number, valinit = self.inputSigma, dragging = False)

            ## Funkce slideru pri zmene jeho hodnoty
            self.smin.on_changed(self.updateImage)
            self.smax.on_changed(self.updateImage)
            self.sclose.on_changed(self.updateImage)
            self.sopen.on_changed(self.updateImage)
            self.ssigma.on_changed(self.updateImage)

            ## Zalozeni mist pro tlacitka
            self.axbuttnext1 = self.fig.add_axes([0.81, 0.24, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttprev1 = self.fig.add_axes([0.86, 0.24, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttnext2 = self.fig.add_axes([0.81, 0.20, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttprev2 = self.fig.add_axes([0.86, 0.20, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttnextclosing = self.fig.add_axes([0.81, 0.16, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttprevclosing = self.fig.add_axes([0.86, 0.16, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttnextopening = self.fig.add_axes([0.81, 0.12, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttprevopening = self.fig.add_axes([0.86, 0.12, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttreset = self.fig.add_axes([0.79, 0.08, 0.06, 0.03], axisbg = self.axcolor)
            self.axbuttcontinue = self.fig.add_axes([0.86, 0.08, 0.06, 0.03], axisbg = self.axcolor)

            ## Zalozeni tlacitek
            self.bnext1 = Button(self.axbuttnext1, '+1.0')
            self.bprev1 = Button(self.axbuttprev1, '-1.0')
            self.bnext2 = Button(self.axbuttnext2, '+1.0')
            self.bprev2 = Button(self.axbuttprev2, '-1.0')
            self.bnextclosing = Button(self.axbuttnextclosing, '+1.0')
            self.bprevclosing = Button(self.axbuttprevclosing, '-1.0')
            self.bnextopening = Button(self.axbuttnextopening, '+1.0')
            self.bprevopening = Button(self.axbuttprevopening, '-1.0')
            self.breset = Button(self.axbuttreset, 'Reset')
            self.bcontinue = Button(self.axbuttcontinue, 'Next UI')

            ## Funkce tlacitek pri jejich aktivaci
            self.bnext1.on_clicked(self.buttonNext1)
            self.bprev1.on_clicked(self.buttonPrev1)
            self.bnext2.on_clicked(self.buttonNext2)
            self.bprev2.on_clicked(self.buttonPrev2)
            self.bnextclosing.on_clicked(self.buttonNextClosing)
            self.bprevclosing.on_clicked(self.buttonPrevClosing)
            self.bnextopening.on_clicked(self.buttonNextOpening)
            self.bprevopening.on_clicked(self.buttonPrevOpening)
            self.breset.on_clicked(self.buttonReset)
            self.bcontinue.on_clicked(self.buttonContinue)

    def run(self):

        if(self.errorsOccured == True):

            return self.data

        self.lastSigma = -42
        self.lastCloseNum = -42
        self.lastOpenNum = -42
        self.lastMinThresNum = -42

        self.overrideSigma = False
        self.overrideThres = False

        self.firstRun = True
        self.newThreshold = False

        if self.interactivity == False:

             self.updateImage(0)
             garbage.collect()

        else:

             self.updateImage(0)
             garbage.collect()
             matpyplot.show()

        del(self.imgBeforeBinaryClosing)
        del(self.imgBeforeBinaryOpening)
        del(self.imgUsed)
        del(self.data)

        garbage.collect()

        return self.imgChanged

    """
    ================================================================
    ================================================================
    Update metoda.
    ================================================================
    ================================================================
    """

    def updateImage(self, val):

        garbage.collect()

        ## Zjisteni jakou sigmu pouzit
        if(self.firstRun == True and self.inputSigma >= 0):
            sigma = numpy.round(self.inputSigma, 2)
        else:
            sigma = numpy.round(self.ssigma.val, 2)
        sigmaNew = self.calculateSigma(sigma)

        ## Filtrovani
        if(self.lastSigma != sigma):
            scipy.ndimage.filters.gaussian_filter(self.data, sigmaNew,
                0, self.imgFiltering, 'reflect', 0.0)
            self.calculateAutomaticThreshold()
            ## Ulozeni posledni hodnoty sigma pro neopakovani stejne operace
            self.lastSigma = sigma
            self.overrideSigma = True

        del(sigmaNew)

        ## Prahovani (smin, smax)
        if ( self.firstRun == True ) :

            if ( self.threshold < 0 ) :

                self.calculateAutomaticThreshold()

            self.imgBeforeBinaryClosing = (self.imgFiltering > self.threshold)

            if ( self.interactivity == True ) :

                self.smin.val = (numpy.round(self.threshold, 2))
                self.smin.valtext.set_text('{}'.format(self.smin.val))
                self.lastMinThresNum = self.smin.val

            self.overrideThres = True

        else:

            if ( self.newThreshold == True ) :

                if ( (self.lastMinThresNum != self.threshold) or (self.overrideSigma == True) ):

                    self.imgBeforeBinaryClosing = (self.imgFiltering > self.threshold)

                    if ( self.interactivity == True ) :

                        self.smin.val = (numpy.round(self.threshold, 2))
                        self.smin.valtext.set_text('{}'.format(self.smin.val))

                    self.lastMinThresNum = self.threshold
                    self.overrideThres = True

            elif ( (self.lastMinThresNum != self.smin.val) or (self.overrideSigma == True) ) :

                self.imgBeforeBinaryClosing = (self.imgFiltering > self.smin.val)
                self.lastMinThresNum = self.smin.val
                self.overrideThres = True

##        self.imgChanged = self.imgBeforeBinaryClosing

        ## Operace binarni otevreni a uzavreni.
        ## Nastaveni hodnot slideru.
        if ( self.interactivity == True ) :

            closeNum = int(numpy.round(self.sclose.val, 0))
            openNum = int(numpy.round(self.sopen.val, 0))
            self.sclose.valtext.set_text('{}'.format(closeNum))
            self.sopen.valtext.set_text('{}'.format(openNum))

        else:

            closeNum = self.ICBinaryClosingIterations
            openNum = self.ICBinaryOpeningIterations

        ## Vlastni binarni uzavreni.
        if(self.firstRun == True and self.ICBinaryClosingIterations >= 1):
            self.imgBeforeBinaryOpening = scipy.ndimage.binary_closing(self.imgBeforeBinaryClosing,
                    structure = None, iterations = closeNum)
        else:
            if( ((closeNum >= 1) and (self.lastCloseNum != closeNum)) or (self.overrideThres == True) ):
                self.imgBeforeBinaryOpening = scipy.ndimage.binary_closing(self.imgBeforeBinaryClosing,
                    structure = None, iterations = closeNum)
            else:
                self.imgBeforeBinaryOpening = self.imgBeforeBinaryClosing
        self.lastCloseNum = closeNum

        ## Vlastni binarni otevreni.
        if(self.firstRun == True and self.ICBinaryOpeningIterations >= 1):
            self.imgChanged = scipy.ndimage.binary_opening(self.imgBeforeBinaryOpening,
                    structure = None, iterations = openNum)
        else:
            if( ((openNum >= 1) and (self.lastOpenNum != openNum)) or (self.overrideThres == True) ):
                self.imgChanged = scipy.ndimage.binary_opening(self.imgBeforeBinaryOpening,
                    structure = None, iterations = openNum)
            else:
                self.imgChanged = self.imgBeforeBinaryOpening
        self.lastOpenNum = openNum

        ## Zjisteni nejvetsich objektu
        self.imgChanged = segmentation.getBiggestObjects(self.imgChanged, self.nObj)

        if ( self.interactivity == True ) :
            ## Predani obrazku k vykresleni
            self.im1 = self.ax1.imshow(numpy.amax(self.imgChanged, 0), self.cmap)
            self.im2 = self.ax2.imshow(numpy.amax(self.imgChanged, 1), self.cmap)
            self.im3 = self.ax3.imshow(numpy.amax(self.imgChanged, 2), self.cmap)

##            ## Minimalni pouzitelna hodnota prahovani v obrazku
##            self.min0 = numpy.amin(self.imgChanged)
##            ## Maximalni pouzitelna hodnota prahovani v obrazku
##            self.max0 = numpy.amax(self.imgChanged)

            ## Prekresleni
            self.fig.canvas.draw()

        if ( self.firstRun == True ) :
            self.firstRun = False

        if ( self.newThreshold == True ) :
            self.newThreshold = False

        self.overrideSigma = False
        self.overrideThres = False

        garbage.collect()

    """
    ================================================================
    ================================================================
    Vypocetni metody.
    ================================================================
    ================================================================
    """

    def calculateSigma(self, input):

        if ( self.voxel[0] == self.voxel[1] == self.voxel[2] ):
            return ((5 / self.voxel[0]) * input) / self.voxelV
        else:
            sigmaX = (5.0 / self.voxel[0]) * input
            sigmaY = (5.0 / self.voxel[1]) * input
            sigmaZ = (5.0 / self.voxel[2]) * input

            return (sigmaX, sigmaY, sigmaZ) / self.voxelV

    ## Automaticky vypocet vhodneho prahu
    def calculateAutomaticThreshold(self):

        print("Hledani prahu...")

        self.imgUsed = self.data

        ## Hustota hist
        hist_points = 1300
        ## Pocet bodu v primce 1 ( klesajici od maxima )
        pointsFrom = 20 #(int)(hist_points * 0.05)
        ## Pocet bodu v primce 2 ( stoupajici k okoli spravneho thresholdu)
        pointsTo = 20 #(int)(hist_points * 0.1)
        ## Pocet bodu k preskoceni od konce hist
        pointsSkip = (int)(hist_points * 0.025)
        ## hledani maxima: zacina se od 'start'*10 procent delky histu (aby se preskocili prvni oscilace)
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
        for index in range(init_index + pointsFrom + pointsSkip, init_index + pointsFrom + pointsSkip + pointsTo): # len(muj_histogram) - 5 - int(0.1 * len(muj_histogram)), len(muj_histogram) - 5
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

    """
    ================================================================
    ================================================================
    Obsluha udalosti (buttons).
    ================================================================
    ================================================================
    """

    def buttonReset(self, event):

        self.sclose.valtext.set_text('{}'.format(self.ICBinaryClosingIterations))
        self.sopen.valtext.set_text('{}'.format(self.ICBinaryOpeningIterations))
        self.ssigma.valtext.set_text('{}'.format(self.inputSigma))

        self.firstRun = True
        self.lastSigma = -1
        self.threshold = -1

        self.updateImage(0)

    def buttonContinue(self, event):

        matpyplot.clf()
        matpyplot.close()

    def buttonNext1(self, event):

        self.smin.val += 1.0
        self.smin.val = (numpy.round(self.smin.val, 2))
        self.smin.valtext.set_text('{}'.format(self.smin.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonPrev1(self, event):

        if(self.smin.val - 1.0 >= 0):
            self.smin.val -= 1.0
            self.smin.val = (numpy.round(self.smin.val, 2))
            self.smin.valtext.set_text('{}'.format(self.smin.val))
            self.fig.canvas.draw()
            self.updateImage(0)

    def buttonNext2(self, event):

        self.smax.val += 1.0
        self.smax.val = (numpy.round(self.smax.val, 2))
        self.smax.valtext.set_text('{}'.format(self.smax.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonPrev2(self, event):

        if(self.smax.val - 1.0 >= 0):
            self.smax.val -= 1.0
            self.smax.val = (numpy.round(self.smax.val, 2))
            self.smax.valtext.set_text('{}'.format(self.smax.val))
            self.fig.canvas.draw()
            self.updateImage(0)

    def buttonNextOpening(self, event):

        self.sopen.val += 1.0
        self.sopen.val = (numpy.round(self.sopen.val, 2))
        self.sopen.valtext.set_text('{}'.format(self.sopen.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonPrevOpening(self, event):

        if(self.sopen.val >= 1.0):
            self.sopen.val -= 1.0
            self.sopen.val = (numpy.round(self.sopen.val, 2))
            self.sopen.valtext.set_text('{}'.format(self.sopen.val))
            self.fig.canvas.draw()
            self.updateImage(0)

    def buttonNextClosing(self, event):

        self.sclose.val += 1.0
        self.sclose.val = (numpy.round(self.sclose.val, 2))
        self.sclose.valtext.set_text('{}'.format(self.sclose.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonPrevClosing(self, event):

        if(self.sclose.val >= 1.0):
            self.sclose.val -= 1.0
            self.sclose.val = (numpy.round(self.sclose.val, 2))
            self.sclose.valtext.set_text('{}'.format(self.sclose.val))
            self.fig.canvas.draw()
            self.updateImage(0)














