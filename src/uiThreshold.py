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
    def __init__(self, data, voxel, threshold, interactivity, number = 100.0, inputSigma = -1, nObj = 1,
    initslice = 0, cmap = matplotlib.cm.Greys_r):

        print('Spoustim prahovani dat...')

        self.interactivity = interactivity

        inputDimension = numpy.ndim(data)
        #print('Dimenze vstupu: ',  inputDimension)
        self.cmap = cmap
        self.number = number
        self.inputSigma = inputSigma
        self.threshold = threshold
        self.nObj = nObj

        if (sys.version_info[0] < 3):
            import copy
            self.data = copy.copy(data)
            self.voxel = copy.copy(voxel)
            self.imgUsed = copy.copy(data)
            self.imgChanged = copy.copy(self.imgUsed)
        else:
            self.data = data.copy()
            self.voxel = voxel.copy()
            self.imgUsed = data.copy()
            self.imgChanged = self.imgUsed.copy()

        ## Kalkulace objemove jednotky (voxel) (V = a*b*c)
        voxel1 = self.voxel[0]
        voxel2 = self.voxel[1]
        voxel3 = self.voxel[2]
        self.voxelV = voxel1 * voxel2 * voxel3

        if(self.interactivity == False):
            return

        if(inputDimension == 3):

            self.lastSigma = -1.0

            ## Minimalni pouzita hodnota prahovani v obrazku
            self.min0 = numpy.amin(self.imgUsed)
            ## Maximalni pouzita hodnota prahovani v obrazku
            self.max0 = numpy.amax(self.imgUsed)

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

    def Initialization(self):

        self.calculateAutomaticThreshold()
        self.firstRun = False

        self.smin.val = (numpy.round(self.threshold, 2))
        self.smin.valtext.set_text('{}'.format(self.smin.val))
        self.ssigma.val = (numpy.round(self.lastSigma, 2))
        self.ssigma.valtext.set_text('{}'.format(self.ssigma.val))

        scipy.ndimage.filters.gaussian_filter(self.data, self.calculateSigma(self.inputSigma), 0, self.imgUsed, 'reflect', 0.0)
        imgThres = self.imgUsed > self.threshold
        self.imgChanged = segmentation.getBiggestObjects(imgThres, self.nObj)

        del(imgThres)

        ## Predani obrazku k vykresleni
        self.im1 = self.ax1.imshow(numpy.amax(self.imgChanged, 0), self.cmap)
        self.im2 = self.ax2.imshow(numpy.amax(self.imgChanged, 1), self.cmap)
        self.im3 = self.ax3.imshow(numpy.amax(self.imgChanged, 2), self.cmap)

        ## Minimalni pouzitelna hodnota prahovani v obrazku
        self.min0 = numpy.amin(self.imgChanged)
        ## Maximalni pouzitelna hodnota prahovani v obrazku
        self.max0 = numpy.amax(self.imgChanged)

        ## Prekresleni
        self.fig.canvas.draw()

    def run(self):

        if(self.interactivity == True):
            self.Initialization()
            ## Zobrazeni plot (figure)
            garbage.collect()
            matpyplot.show()
        else:
            self.autoWork()

        del(self.imgUsed)
        del(self.data)
        garbage.collect()

        return self.imgChanged

    def autoWork(self):

        if(self.threshold == -1):
            print('Hledani prahu...')
            self.calculateAutomaticThreshold()
        print('Vlastni rozmazani a prahovani dat...')
        scipy.ndimage.filters.gaussian_filter(self.data, self.calculateSigma(self.inputSigma), 0, self.imgUsed, 'reflect', 0.0)
        self.imgChanged = self.imgUsed > self.threshold

    ## Automaticky vypocet vhodneho prahu
    def calculateAutomaticThreshold(self):

        # TODO: automaticky vypocet prahu

        print("Hledani prahu...")

        self.imgUsed = self.data

        if(self.inputSigma >= 0):
            sigma = float(self.inputSigma)
            self.lastSigma = sigma
            scipy.ndimage.filters.gaussian_filter(self.data, self.calculateSigma(sigma), 0, self.imgUsed, 'reflect', 0.0)

        hist_points = 1300
        ## hist: funkce(threshold)
        hist, bin_edges = numpy.histogram(self.imgUsed, bins = hist_points)
        ## bin_centers: threshold
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        """
        start = 0
        for index in range(0, len(hist)):
			if(hist[index] > 50):
				start = index
				break

        suma_hist = 0
        for index in range(start, len(hist)):
			suma_hist += hist[index]

        percent_hist = 0.70 ## !!!!!!!!!!!! 70% !!!!!!!!!!!
        crop_hist = suma_hist * (1 - percent_hist)

        control_suma = 0
        init_index = 0 ## pocatek
        for index in range(start, len(hist)):
			control_suma += hist[index]
			if(control_suma > crop_hist):
				init_index = index
				break
        """

        ## last je maximum z hist
        ## init_index je index "last"
        init_index = 0
        last = hist[(int)(len(hist) / 10)]
        for index in range((int)(len(hist) / 10), len(hist)):
            if(last < hist[index]):
                last = hist[index]
                init_index = index

        ## muj_histogram_temp(x+1) = f(x+1) = hist[x+1] + hist[x]
        muj_histogram_temp = []
        muj_histogram_temp.insert(0, hist[0])
        for index in range(1, len(hist)):
            muj_histogram_temp.insert(index, hist[index] + muj_histogram_temp[index - 1])

        ## reverse muj_histogram_temp do muj_histogram
        muj_histogram = muj_histogram_temp[::-1]

        ## Pridani bodu to poli x1 a y1
        ## (klesajici tendence)
        x1 = []
        y1 = []
        place = 0
        for index in range(init_index, init_index + 20):
            x1.insert(place, bin_centers[index])
            y1.insert(place, muj_histogram[index])
            print("[ " + str(x1[place]) + ", " + str(y1[place]) + " ]")
            place += 1

        ## Linearni regrese nad x1 a y1
        ## slope == smernice
        ## intercept == posuv
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)

        print("slope1 = " + str(slope1))
        print("intercept1 = " + str(intercept1))
        print("r_value1 = " + str(r_value1))
        print("p_value1 = " + str(p_value1))
        print("std_err1 = " + str(std_err1))
        print(str(slope1) + "x + " + str(intercept1))

        x2 = []
        y2 = []
        place = 0
        for index in range(len(muj_histogram) - 45, len(muj_histogram) - 5):
            x2.insert(place, bin_centers[index])
            y2.insert(place, muj_histogram[index])
            print("[ " + str(x2[place]) + ", " + str(y2[place]) + " ]")
            place += 1

        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)

        print("slope2 = " + str(slope2))
        print("intercept2 = " + str(intercept2))
        print("r_value2 = " + str(r_value2))
        print("p_value2 = " + str(p_value2))
        print("std_err2 = " + str(std_err2))
        print(str(slope2) + "x + " + str(intercept2))

        self.threshold = (intercept2 - intercept1) / (slope1 - slope2)

        print('Zjisten threshold: ' + str(self.threshold))
        print('====================================')
        print('!- ANO, jeste to porad nefunguje => pracuje se na tom ;-)')

        muj_histogram_graph = []
        bin_centers_graph = []
        place = 0
        for index in range(len(muj_histogram) / 2, len(muj_histogram)):
            bin_centers_graph.insert(place, bin_centers[index])
            muj_histogram_graph.insert(place, muj_histogram[index])
            place += 1

        matpyplot.figure(figsize = (11, 4))
        matpyplot.plot(bin_centers_graph, muj_histogram_graph, lw = 2)
        matpyplot.axvline(self.threshold, color = 'r', ls = '--', lw = 2)
        matpyplot.show()

        garbage.collect()

        """
        counter = 1
        oldNum = hist[len(bin_centers) - 1]
        print('====================================')
        print('Toto jsou debug vypisy, ktere budou (casem) vymazany ;-)')
        print('len(hist) == ' + str(len(hist)))

        suma_hist = 0
        start = numpy.round(len(hist) / 10, 0)
        for index in range(start, len(hist)):
            suma_hist += hist[index]

        print('moje suma hist == ' + str(suma_hist))
        print('suma hist == ' + str(sum(hist)))

        mark = -1
        mark_suma = 0
        self.threshold_percent = 0.30
        stop_suma = suma_hist * self.threshold_percent
        print('Upominka: Threshold se pocita pro ' + str((int)(self.threshold_percent * 100)) + '% dat.')
        end = start - 1
        start = len(hist) - 1
        index = start
        while(index > end):
            mark_suma += hist[index]
            if(mark_suma >= stop_suma):
                mark = index
                break
            index = index - 1

        self.threshold = (bin_centers[mark - 1] + bin_centers[mark]
                            + bin_centers[mark + 1]) / 3.0

		print('Zjisten threshold: ' + str(self.threshold))
        print('====================================')
        print('!- ANO, jeste to porad nefunguje => pracuje se na tom ;-)')

        matpyplot.figure(figsize = (11, 4))
        matpyplot.plot(bin_centers, hist, lw = 2)
        matpyplot.axvline(self.threshold, color = 'r', ls = '--', lw = 2)
        matpyplot.show()

        garbage.collect()
		"""

    def updateImgFilter(self, val):

        if(self.interactivity == True):
            sigma = float(self.ssigma.val)
        else:
            sigma = float(self.inputSigma)

        ## Filtrovani
        if(self.lastSigma != sigma):
            scipy.ndimage.filters.gaussian_filter(self.data, self.calculateSigma(sigma), 0, self.imgUsed, 'reflect', 0.0)
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

        if ( self.voxel[0] == self.voxel[1] == self.voxel[2] ):
            return ((5 / self.voxel[0]) * input) / self.voxelV
        else:
            sigmaX = (5.0 / self.voxel[0]) * input
            sigmaY = (5.0 / self.voxel[1]) * input
            sigmaZ = (5.0 / self.voxel[2]) * input

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
            imgThres = (self.imgUsed > self.smin.val) & (self.imgUsed < self.smax.val)
        else:
            imgThres = (self.imgUsed > self.threshold)

        self.imgChanged = segmentation.getBiggestObjects(imgThres, self.nObj)

        del(imgThres)

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

        garbage.collect()














