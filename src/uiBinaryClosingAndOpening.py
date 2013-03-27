#
# -*- coding: utf-8 -*-
"""
================================================================================
Name:        uiBinaryClosingAndOpening
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
#import scipy.misc
#import scipy.io
import scipy.ndimage

#import unittest
#import argparse

import matplotlib.pyplot as matpyplot
import matplotlib
from matplotlib.widgets import Slider, Button#, RadioButtons

# Import garbage collector
import gc as garbage

"""
================================================================================
uiBinaryClosingAndOpening
================================================================================
"""
class uiBinaryClosingAndOpening:

    """
    Metoda init.
        data - data pro operace binary closing a opening, se kterymi se pracuje
        initslice - PROZATIM NEPOUZITO
        cmap - grey
    """
    def __init__(self, data, binaryClosingIterations, binaryOpeningIterations, interactivity,
    initslice = 0, cmap = matplotlib.cm.Greys_r):

        print('Spoustim binarni otevreni a uzavreni dat...')

        if(binaryClosingIterations >= 1 or binaryOpeningIterations >= 1):
            self.interactivity = False
        else:
            self.interactivity = interactivity

        inputDimension = numpy.ndim(data)
        self.cmap = cmap
        self.imgUsed = data
        self.imgChanged = self.imgUsed
        self.imgChanged1 = self.imgUsed

        if(self.interactivity == False):
            self.binaryClosingIterations = binaryClosingIterations
            self.binaryOpeningIterations = binaryOpeningIterations

        if(self.interactivity == False):
            return

        if(inputDimension == 2):

            self.imgUsed = self.imgUsed
            self.imgChanged = self.imgUsed

            # Ziskani okna (figure)
            self.fig = matpyplot.figure()
            # Pridani subplotu do okna (do figure)
            self.ax1 = self.fig.add_subplot(111)

            # Upraveni subplotu
            self.fig.subplots_adjust(left = 0.1, bottom = 0.25)

            # Vykresli obrazek
            self.im1 = self.ax1.imshow(self.imgChanged, self.cmap)

            # Zakladni informace o slideru
            axcolor = 'white' # lightgoldenrodyellow
            axmin = self.fig.add_axes([0.25, 0.16, 0.495, 0.03], axisbg = axcolor)
            axmax  = self.fig.add_axes([0.25, 0.12, 0.495, 0.03], axisbg = axcolor)
            # Vytvoreni slideru
                # Minimalni pouzita hodnota v obrazku
            min0 = self.imgUsed.min()
                # Maximalni pouzita hodnota v obrazku
            max0 = self.imgUsed.max()
                # Vlastni vytvoreni slideru
            self.smin = Slider(axmin, 'Minimal threshold', min0, max0, valinit = min0)
            self.smax = Slider(axmax, 'Maximal threshold', min0, max0, valinit = max0)

            # Udalost pri zmene hodnot slideru - volani updatu
            self.smin.on_changed(self.updateImg2D)
            self.smax.on_changed(self.updateImg2D)

        elif(inputDimension == 3):

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
                axopening1 = self.fig.add_axes([0.25, 0.14, 0.55, 0.03], axisbg = self.axcolor)
                axclosing1 = self.fig.add_axes([0.25, 0.18, 0.55, 0.03], axisbg = self.axcolor)

                ## Vlastni vytvoreni slideru
                self.sopen1 = Slider(axopening1, 'Binary opening', 0, 100, valinit = 0)
                self.sclose1 = Slider(axclosing1, 'Binary closing', 0, 100, valinit = 0)

                ## Funkce slideru pri zmene jeho hodnoty
                self.sopen1.on_changed(self.updateImg1Binary3D)
                self.sclose1.on_changed(self.updateImg1Binary3D)

                self.sopen1.valtext.set_text('{}'.format(int(self.sopen1.val)))
                self.sclose1.valtext.set_text('{}'.format(int(self.sclose1.val)))

                ## Zalozeni mist pro tlacitka
                self.axbuttnextopening = self.fig.add_axes([0.83, 0.14, 0.04, 0.03], axisbg = self.axcolor)
                self.axbuttprevopening = self.fig.add_axes([0.88, 0.14, 0.04, 0.03], axisbg = self.axcolor)
                self.axbuttnextclosing = self.fig.add_axes([0.83, 0.18, 0.04, 0.03], axisbg = self.axcolor)
                self.axbuttprevclosing = self.fig.add_axes([0.88, 0.18, 0.04, 0.03], axisbg = self.axcolor)
                self.axbuttreset = self.fig.add_axes([0.80, 0.08, 0.07, 0.03], axisbg = self.axcolor)
                self.axbuttcontinue = self.fig.add_axes([0.88, 0.08, 0.07, 0.03], axisbg = self.axcolor)

                ## Zalozeni tlacitek
                self.bnextopening = Button(self.axbuttnextopening, '+1.0')
                self.bprevopening = Button(self.axbuttprevopening, '-1.0')
                self.bnextclosing = Button(self.axbuttnextclosing, '+1.0')
                self.bprevclosing = Button(self.axbuttprevclosing, '-1.0')
                self.breset = Button(self.axbuttreset, 'Reset')
                self.bcontinue = Button(self.axbuttcontinue, 'End editing')

                ## Funkce tlacitek pri jejich aktivaci
                self.bnextopening.on_clicked(self.button3DNextOpening)
                self.bprevopening.on_clicked(self.button3DPrevOpening)
                self.bnextclosing.on_clicked(self.button3DNextClosing)
                self.bprevclosing.on_clicked(self.button3DPrevClosing)
                self.breset.on_clicked(self.button3DReset)
                self.bcontinue.on_clicked(self.button3DContinue)

        else:
            print('Spatny vstup.\nDimenze vstupu neni 2 ani 3.\nUkoncuji prahovani.')

    def run(self):

        if(self.interactivity == True):
            ## Zobrazeni plot (figure)
            matpyplot.show()
        else:
            self.autoWork()

        del(self.imgUsed)
        del(self.imgChanged)

        garbage.collect()

        return self.imgChanged1

    def autoWork(self):

        self.updateImg1Binary3D(self)

    def button3DReset(self, event):

        self.sopen1.val = 0.0
        self.sopen1.valtext.set_text('{}'.format(int(self.sopen1.val)))
        self.sclose1.val = 0.0
        self.sclose1.valtext.set_text('{}'.format(int(self.sclose1.val)))

        ## Vykreslit obrazek
        self.im1 = self.ax1.imshow(numpy.amax(self.imgChanged, 0), self.cmap)
        self.im2 = self.ax2.imshow(numpy.amax(self.imgChanged, 1), self.cmap)
        self.im3 = self.ax3.imshow(numpy.amax(self.imgChanged, 2), self.cmap)

        ## Prekresleni
        self.fig.canvas.draw()

    def button3DContinue(self, event):

        matpyplot.clf()
        matpyplot.close()

    def button3DNextOpening(self, event):

        self.sopen1.val += 1.0
        self.sopen1.val = (numpy.round(self.sopen1.val, 2))
        self.sopen1.valtext.set_text('{}'.format(self.sopen1.val))
        self.fig.canvas.draw()
        self.updateImg1Binary3D(self)

    def button3DPrevOpening(self, event):

        if(self.sopen1.val >= 1.0):
            self.sopen1.val -= 1.0
            self.sopen1.val = (numpy.round(self.sopen1.val, 2))
            self.sopen1.valtext.set_text('{}'.format(self.sopen1.val))
            self.fig.canvas.draw()
            self.updateImg1Binary3D(self)

    def button3DNextClosing(self, event):

        self.sclose1.val += 1.0
        self.sclose1.val = (numpy.round(self.sclose1.val, 2))
        self.sclose1.valtext.set_text('{}'.format(self.sclose1.val))
        self.fig.canvas.draw()
        self.updateImg1Binary3D(self)

    def button3DPrevClosing(self, event):

        if(self.sclose1.val >= 1.0):
            self.sclose1.val -= 1.0
            self.sclose1.val = (numpy.round(self.sclose1.val, 2))
            self.sclose1.valtext.set_text('{}'.format(self.sclose1.val))
            self.fig.canvas.draw()
            self.updateImg1Binary3D(self)

    def updateImg2D(self, val):

        ## Prahovani (smin, smax)
        img1 = self.imgUsed.copy() > self.smin.val
        self.imgChanged = img1 #< self.smax.val

        ## Predani obrazku k vykresleni
        self.im1 = self.ax1.imshow(self.imgChanged, self.cmap)
        ## Prekresleni
        self.fig.canvas.draw()

    def updateImg1Binary3D(self, val):

        ## Nastaveni hodnot slideru
        if(self.interactivity == True):
            self.sopen1.valtext.set_text('{}'.format(int(numpy.round(self.sopen1.val, 0))))
            self.sclose1.valtext.set_text('{}'.format(int(numpy.round(self.sclose1.val, 0))))
            openDil = int(numpy.round(self.sopen1.val, 0))
            closeDil = int(numpy.round(self.sclose1.val, 0))
        else:
            openDil = int(numpy.round(self.binaryOpeningIterations, 0))
            closeDil = int(numpy.round(self.binaryClosingIterations, 0))

        ## Prekresleni
        if(self.interactivity == True):
            self.fig.canvas.draw()

        imgChanged1 = self.imgChanged

        if(closeDil >= 0.1):
            imgChanged1 = scipy.ndimage.binary_closing(self.imgChanged, structure = None, iterations = closeDil)
        else:
            imgChanged1 = self.imgChanged

        if(openDil >= 0.1):
            self.imgChanged1 = scipy.ndimage.binary_opening(imgChanged1, structure = None, iterations = openDil)
        else:
            self.imgChanged1 = imgChanged1

        if(self.interactivity == True):
            ## Predani obrazku k vykresleni
            ## Vykreslit obrazek
            self.im1 = self.ax1.imshow(numpy.amax(self.imgChanged1, 0), self.cmap)
            self.im2 = self.ax2.imshow(numpy.amax(self.imgChanged1, 1), self.cmap)
            self.im3 = self.ax3.imshow(numpy.amax(self.imgChanged1, 2), self.cmap)

            ## Prekresleni
            self.fig.canvas.draw()

        del(imgChanged1)
        garbage.collect()
