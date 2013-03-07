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
import sys

"""
import scipy.misc
import scipy.io

import unittest
import argparse
"""

import matplotlib
import matplotlib.pyplot as matpyplot
from matplotlib.widgets import Slider, Button#, RadioButtons

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
    def __init__(self, data, voxel, threshold, interactivity, number = 100.0, inputSigma = -1,
    initslice = 0, cmap = matplotlib.cm.Greys_r):

        print('Spoustim prahovani dat.')

        self.interactivity = interactivity

        inputDimension = numpy.ndim(data)
        #print('Dimenze vstupu: ',  inputDimension)
        self.cmap = cmap
        self.number = number
        self.inputSigma = inputSigma
        self.threshold = threshold

        if (sys.version_info[0] < 3):
            import copy
            self.data = copy.copy(data)
        else:
            self.data = data.copy()

        if(inputDimension == 2):

            self.voxel = voxel
            self.imgUsed = data
            self.imgChanged = data

            ## Ziskani okna (figure)
            self.fig = matpyplot.figure()

            ## Pridani subplotu do okna (do figure)
            self.ax1 = self.fig.add_subplot(111)

            ## Upraveni subplotu
            self.fig.subplots_adjust(left = 0.1, bottom = 0.25)

            ## Vykresli obrazek
            self.im1 = self.ax1.imshow(self.imgChanged, self.cmap)

            ## Zakladni informace o slideru
            axcolor = 'white' # lightgoldenrodyellow
            axmin = self.fig.add_axes([0.25, 0.16, 0.495, 0.03], axisbg = axcolor)
            axmax  = self.fig.add_axes([0.25, 0.12, 0.495, 0.03], axisbg = axcolor)

            ## Vytvoreni slideru
                ## Minimalni pouzita hodnota v obrazku
            min0 = data.min()
                ## Maximalni pouzita hodnota v obrazku
            max0 = data.max()
                ## Vlastni vytvoreni slideru
            self.smin = Slider(axmin, 'Minimal threshold', min0, max0, valinit = min0)
            self.smax = Slider(axmax, 'Maximal threshold', min0, max0, valinit = max0)

            ## Udalost pri zmene hodnot slideru - volani updatu
            self.smin.on_changed(self.updateImg2D)
            self.smax.on_changed(self.updateImg2D)

        elif(inputDimension == 3):

            if (sys.version_info[0] < 3):
                import copy
                self.voxel = copy.copy(voxel)
                self.imgUsed = copy.copy(data)
                self.imgChanged = copy.copy(self.imgUsed)
            else:
                self.voxel = voxel.copy()
                self.imgUsed = data.copy()
                self.imgChanged = self.imgUsed.copy()

            self.lastSigma = -1.0

            ## Kalkulace objemove jednotky (voxel) (V = a*b*c)
            voxel1 = self.voxel[0][0]
            voxel2 = self.voxel[1][0]
            voxel3 = self.voxel[2][0]
            self.voxelV = voxel1 * voxel2 * voxel3

            ## Minimalni pouzita hodnota prahovani v obrazku
            self.min0 = numpy.amin(self.imgUsed)
            ## Maximalni pouzita hodnota prahovani v obrazku
            self.max0 = numpy.amax(self.imgUsed)

            if(self.interactivity == True):

                self.fig = matpyplot.figure()

                ## Pridani subplotu do okna (do figure)
                self.ax1 = self.fig.add_subplot(131)
                self.ax2 = self.fig.add_subplot(132)
                self.ax3 = self.fig.add_subplot(133)

                ## Upraveni subplotu
                self.fig.subplots_adjust(left = 0.1, bottom = 0.3)

                ## Vykreslit obrazek
                self.im1 = self.ax1.imshow(numpy.amax(self.imgUsed, 0), self.cmap)
                self.im2 = self.ax2.imshow(numpy.amax(self.imgUsed, 1), self.cmap)
                self.im3 = self.ax3.imshow(numpy.amax(self.imgUsed, 2), self.cmap)

                ## Zalozeni mist pro slidery
                self.axcolor = 'white' # lightgoldenrodyellow
                self.axmin = self.fig.add_axes([0.20, 0.20, 0.55, 0.03], axisbg = self.axcolor)
                self.axmax  = self.fig.add_axes([0.20, 0.16, 0.55, 0.03], axisbg = self.axcolor)
                self.axsigma = self.fig.add_axes([0.20, 0.08, 0.55, 0.03], axisbg = self.axcolor)

                ## Vlastni vytvoreni slideru
                self.smin = Slider(self.axmin, 'Minimal threshold', self.min0, self.max0, valinit = self.min0)
                self.smax = Slider(self.axmax, 'Maximal threshold', self.min0, self.max0, valinit = self.max0)
                self.ssigma = Slider(self.axsigma, 'Sigma', 0.00, self.number, valinit = self.inputSigma)

                ## Funkce slideru pri zmene jeho hodnoty
                self.smin.on_changed(self.updateImg1Threshold3D)
                self.smax.on_changed(self.updateImg1Threshold3D)
                self.ssigma.on_changed(self.updateImgFilter)

                ## Zalozeni mist pro tlacitka
                self.axbuttnext1 = self.fig.add_axes([0.81, 0.20, 0.04, 0.03], axisbg = self.axcolor)
                self.axbuttprev1 = self.fig.add_axes([0.86, 0.20, 0.04, 0.03], axisbg = self.axcolor)
                self.axbuttnext2 = self.fig.add_axes([0.81, 0.16, 0.04, 0.03], axisbg = self.axcolor)
                self.axbuttprev2 = self.fig.add_axes([0.86, 0.16, 0.04, 0.03], axisbg = self.axcolor)
                self.axbuttreset = self.fig.add_axes([0.79, 0.08, 0.06, 0.03], axisbg = self.axcolor)
                self.axbuttcontinue = self.fig.add_axes([0.86, 0.08, 0.06, 0.03], axisbg = self.axcolor)

                ## Zalozeni tlacitek
                self.bnext1 = Button(self.axbuttnext1, '+1.0')
                self.bprev1 = Button(self.axbuttprev1, '-1.0')
                self.bnext2 = Button(self.axbuttnext2, '+1.0')
                self.bprev2 = Button(self.axbuttprev2, '-1.0')
                self.breset = Button(self.axbuttreset, 'Reset')
                self.bcontinue = Button(self.axbuttcontinue, 'Next UI')

                ## Funkce tlacitek pri jejich aktivaci
                self.bnext1.on_clicked(self.button3DNext1)
                self.bprev1.on_clicked(self.button3DPrev1)
                self.bnext2.on_clicked(self.button3DNext2)
                self.bprev2.on_clicked(self.button3DPrev2)
                self.breset.on_clicked(self.button3DReset)
                self.bcontinue.on_clicked(self.button3DContinue)

        else:

            print('Spatny vstup.\nDimenze vstupu neni 2 ani 3.\nUkoncuji prahovani.')

    def run(self):

        if(self.interactivity == True):
            ## Provedeni pocatecniho gauss. filtrovani
            self.firstRun = True
            self.updateImgFilter(self)
            ## Zobrazeni plot (figure)
            matpyplot.show()
        else:
            self.autoWork()

        del(self.imgUsed)
        del(self.data)

        return self.imgChanged

    def autoWork(self):

        if(self.threshold == -1 and self.interactivity == False):
            self.calculateAutomaticThreshold()
        self.firstRun = False
        self.updateImgFilter(self)

    ## Automaticky vypocet vhodneho prahu - Otsu´s method
    def calculateAutomaticThreshold(self):

        # TODO: automaticky vypocet prahu

        hist, bin_edges = numpy.histogram(self.data, bins=60)
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

##        print('bin_edges')
##        print(bin_edges)
##        print('bin_centers')
##        print(bin_centers)
##        print('hist')
##        print(hist)

        # hist jsou udaje na ose x
        # bin_centers jsou udaje na ose y (asi pocet zmen)
        # navrh algoritmu: jit od posledni pozice v hist, najit paralelne
        #   nejvyssi honotu, dokud hodnoty v bin_centers nezacnou klesat
        #   nebo hlidat vzrust hodnot - pokud nova hodnota bude vetsi
        #   nez kvadrat (nutno zvolit mocninu) puvodni hodnoty, tak
        #   puvodni hodnota je ta spravna

        counter = 1
        oldNum = hist[len(bin_centers) - 1]
        print('len(hist) == ' + str(len(hist)))
        for index in range(1, len(hist)):
            indexBack = len(hist) - 1 - index
            print('========')
            print('index == ' + str(indexBack))
            newNum = hist[indexBack]
            print('bin_edge == ' + str(bin_edges[indexBack]))
            print('hist == ' + str(hist[indexBack]))
            if(newNum >= oldNum):
                print('found: ' + str(bin_centers[indexBack + 1]))
                self.threshold = bin_centers[indexBack + 1]
                oldNum = newNum
            else:
                if(counter > 10):
                    break
                counter += 1

##        oldNum = hist[len(bin_centers) - 1]
##        superNum = hist[len(bin_centers) - 1]
##        print('len(hist) == ' + str(len(hist)))
##        for index in range(1, len(hist)):
##            indexBack = len(hist) - 1 - index
##            print(indexBack)
##            newNum = hist[indexBack]
##            print('newNum == ' + str(newNum))
##            print('oldNum == ' + str(oldNum))
##            if(newNum >= (3.5 * oldNum) and oldNum > superNum):
##                print('found: ' + str(bin_centers[indexBack + 1]))
##                self.threshold = bin_centers[indexBack + 1]
##            oldNum = newNum

##        matpyplot.figure(figsize=(11,4))
##        matpyplot.plot(bin_centers, hist, lw=2)
##        matpyplot.axvline(self.threshold, color='r', ls='--', lw=2)
##        matpyplot.show()

    def updateImgFilter(self, val):

        if(self.interactivity == True):
            sigma = float(self.ssigma.val)
        else:
            sigma = float(self.threshold)

        ## Filtrovani
        if(self.lastSigma != sigma):
            scipy.ndimage.filters.gaussian_filter(self.data.copy(), self.calculateSigma(sigma), 0, self.imgUsed, 'reflect', 0.0)
            ## Ulozeni posledni hodnoty sigma pro neopakovani stejne operace
            self.lastSigma = sigma

        ## Vykresleni novych pohledu z filtrovani
        if(self.firstRun == False):
            self.updateImg1Threshold3D(self)
        else:
            self.firstRun = False
            if(self.interactivity == True):
                self.im1 = self.ax1.imshow(numpy.amax(self.imgUsed, 0), self.cmap)
                self.im2 = self.ax2.imshow(numpy.amax(self.imgUsed, 1), self.cmap)
                self.im3 = self.ax3.imshow(numpy.amax(self.imgUsed, 2), self.cmap)

        if(self.interactivity == True):
            ## Prekresleni
            self.fig.canvas.draw()


    def calculateSigma(self, input):

        if ( self.voxel[0][0] == self.voxel[1][0] == self.voxel[2][0] ):
            return ((5 / self.voxel[0][0]) * input) / self.voxelV
        else:

            sigmaX = (5.0 / self.voxel[0][0]) * input
            sigmaY = (5.0 / self.voxel[1][0]) * input
            sigmaZ = (5.0 / self.voxel[2][0]) * input

            return (sigmaX, sigmaY, sigmaZ) / self.voxelV

    def button3DReset(self, event):

        ## Vykresleni novych pohledu z originalnich dat
        self.im1 = self.ax1.imshow(numpy.amax(self.data, 0), self.cmap)
        self.im2 = self.ax2.imshow(numpy.amax(self.data, 1), self.cmap)
        self.im3 = self.ax3.imshow(numpy.amax(self.data, 2), self.cmap)

        ## Prevzeti originalnich dat
        self.imgUsed = self.data.copy()

        ## Nastaveni hodnot slideru
        self.smin.val = (numpy.round(self.min0, 2))
        self.smin.valtext.set_text('{}'.format(self.smin.val))
        self.smax.val = (numpy.round(self.max0, 2))
        self.smax.valtext.set_text('{}'.format(self.smax.val))
        self.ssigma.val = (numpy.round(0.00, 2))
        self.ssigma.valtext.set_text('{}'.format(self.ssigma.val))

        ## Minimalni pouzita hodnota prahovani v obrazku
        self.min0 = numpy.amin(self.imgUsed)
        ## Maximalni pouzita hodnota prahovani v obrazku
        self.max0 = numpy.amax(self.imgUsed)

        ## Prekresleni
        self.fig.canvas.draw()

    def button3DContinue(self, event):

        matpyplot.clf()
        matpyplot.close()

    def button3DNext1(self, event):

        if(self.smin.val + 1.0 <= self.max0):
            self.smin.val += 1.0
            self.smin.val = (numpy.round(self.smin.val, 2))
            self.smin.valtext.set_text('{}'.format(self.smin.val))
            self.fig.canvas.draw()
            self.updateImg1Threshold3D(self)

    def button3DPrev1(self, event):

        if(self.smin.val - 1.0 >= self.min0):
            self.smin.val -= 1.0
            self.smin.val = (numpy.round(self.smin.val, 2))
            self.smin.valtext.set_text('{}'.format(self.smin.val))
            self.fig.canvas.draw()
            self.updateImg1Threshold3D(self)

    def button3DNext2(self, event):

        if(self.smax.val + 1.0 <= self.max0):
            self.smax.val += 1.0
            self.smax.val = (numpy.round(self.smax.val, 2))
            self.smax.valtext.set_text('{}'.format(self.smax.val))
            self.fig.canvas.draw()
            self.updateImg1Threshold3D(self)

    def button3DPrev2(self, event):

        if(self.smax.val - 1.0 >= self.min0):
            self.smax.val -= 1.0
            self.smax.val = (numpy.round(self.smax.val, 2))
            self.smax.valtext.set_text('{}'.format(self.smax.val))
            self.fig.canvas.draw()
            self.updateImg1Threshold3D(self)

    def updateImg1Threshold3D(self, val):

        ## Prahovani (smin, smax)
        if(self.interactivity == True):
            self.imgChanged = (self.imgUsed > self.smin.val) & (self.imgUsed < self.smax.val)
        else:
            self.imgChanged = (self.imgUsed > self.threshold)

        if(self.interactivity == True):
            ## Predani obrazku k vykresleni
            self.im1 = self.ax1.imshow(numpy.amax(self.imgChanged, 0), self.cmap)
            self.im2 = self.ax2.imshow(numpy.amax(self.imgChanged, 1), self.cmap)
            self.im3 = self.ax3.imshow(numpy.amax(self.imgChanged, 2), self.cmap)

        ## Minimalni pouzitelna hodnota prahovani v obrazku
        self.min0 = numpy.amin(self.imgChanged)
        ## Maximalni pouzitelna hodnota prahovani v obrazku
        self.max0 = numpy.amax(self.imgChanged)

        if(self.interactivity == True):
            ## Prekresleni
            self.fig.canvas.draw()

    def updateImg2D(self, val):

        ## Prahovani (smin, smax)
        img1 = self.imgUsed.copy() > self.smin.val
        self.imgChanged = img1 #< self.smax.val

        ## Predani obrazku k vykresleni
        self.im1 = self.ax1.imshow(self.imgChanged, self.cmap)
        ## Prekresleni
        self.fig.canvas.draw()




