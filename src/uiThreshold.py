# -*- coding: utf-8 -*-
"""
================================================================================
Name:        uiThreshold
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky (volkovinsky.pavel@gmail.com)

Created:     08.11.2012
Copyright:   (c) Pavel Volkovinsky 2012
Licence:     <your licence>
================================================================================
"""

import sys
sys.path.append("../src/")
sys.path.append("../extern/")

import logging
logger = logging.getLogger(__name__)

import numpy

from scipy import ndimage
from scipy import misc
import scipy.io

import unittest
import argparse
#import pylab

import matplotlib.pyplot as matpyplot
import matplotlib
from matplotlib.widgets import Slider#, Button, RadioButtons

"""
================================================================================
uiThreshold
================================================================================
"""
class uiThreshold:

    def __init__(self, imgUsed, initslice = 0, cmap = matplotlib.cm.Greys_r):

        inputDimension = numpy.ndim(imgUsed)
        print('Dimension of input: ',  inputDimension)
        self.cmap = cmap
        
        if(inputDimension == 2):
            
            self.imgUsed = imgUsed
            self.imgChanged = imgUsed
                
            """
            self.imgChanged1 = self.imgUsed
            self.imgChanged2 = self.imgUsed
            self.imgChanged3 = self.imgUsed
            """
            
            # Zakladni informace o obrazku (+ statisticke)
            """
            print('Image dtype: ', imgUsed.dtype)
            print('Image size: ', imgUsed.size)
            print('Image shape: ', imgUsed.shape[0], ' x ',  imgUsed.shape[1])
            print('Max value: ', imgUsed.max(), ' at pixel ',  imgUsed.argmax())
            print('Min value: ', imgUsed.min(), ' at pixel ',  imgUsed.argmin())
            print('Variance: ', imgUsed.var())
            print('Standard deviation: ', imgUsed.std())
            """
            
            # Ziskani okna (figure)
            self.fig = matpyplot.figure()
            # Pridani subplotu do okna (do figure)
            self.ax1 = self.fig.add_subplot(111)
            """
    #        self.ax0 = self.fig.add_subplot(232)
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)
            """
            # Upraveni subplotu
            self.fig.subplots_adjust(left = 0.1, bottom = 0.25)
            # Vykresli obrazek
    #        self.im0 = self.ax0.imshow(imgUsed)
            self.im1 = self.ax1.imshow(self.imgUsed, self.cmap)
            """
            self.im2 = self.ax2.imshow(imgUsed)
            self.im3 = self.ax3.imshow(imgUsed)
            """
     #       self.fig.colorbar(self.im1)
    
            # Zakladni informace o slideru
            axcolor = 'white' # lightgoldenrodyellow
            axmin = self.fig.add_axes([0.25, 0.16, 0.495, 0.03], axisbg = axcolor)
            axmax  = self.fig.add_axes([0.25, 0.12, 0.495, 0.03], axisbg = axcolor)
            """
            axopening = self.fig.add_axes([0.25, 0.08, 0.495, 0.03], axisbg = axcolor)
            axclosing = self.fig.add_axes([0.25, 0.04, 0.495, 0.03], axisbg = axcolor)
            """
            
            # Vytvoreni slideru
                # Minimalni pouzita hodnota v obrazku
            min0 = imgUsed.min()
                # Maximalni pouzita hodnota v obrazku
            max0 = imgUsed.max()
                # Vlastni vytvoreni slideru
            self.smin = Slider(axmin, 'Minimal threshold', min0, max0, valinit = min0)
            self.smax = Slider(axmax, 'Maximal threshold', min0, max0, valinit = max0)
            """
            self.sopen = Slider(axopening, 'Binary opening', 0, 10, valinit = 0)
            self.sclose = Slider(axclosing, 'Binary closing', 0, 10, valinit = 0)
            """
            
            # Udalost pri zmene hodnot slideru - volani updatu
            """
            self.smin.on_changed(self.updateMinThreshold)
            self.smax.on_changed(self.updateMaxThreshold)
            self.sopen.on_changed(self.updateBinOpening)
            self.sclose.on_changed(self.updateBinClosing)
            """
            """"""
            self.smin.on_changed(self.updateImg2D)
            self.smax.on_changed(self.updateImg2D)
            """
            self.sopen.on_changed(self.updateImg)
            self.sclose.on_changed(self.updateImg)
            """
        
        elif(inputDimension == 3):
            
            # Zakladni informace o obrazcich (+ statisticke)
            """
            print('Image dtype: ', imgUsed.dtype)
            print('Image size: ', imgUsed.size)
            print('Image shape: ', imgUsed.shape[0], ' x ',  imgUsed.shape[1], ' x ',  imgUsed.shape[2])
            print('Max value: ', imgUsed.max(), ' at pixel ',  imgUsed.argmax())
            print('Min value: ', imgUsed.min(), ' at pixel ',  imgUsed.argmin())
            print('Variance: ', imgUsed.var())
            print('Standard deviation: ', imgUsed.std())
            """
            
            self.imgUsed = imgUsed
            self.imgOutput = self.imgUsed
            
            #self.imgMin = numpy.min(self.imgUsed)
            #self.imgMax = numpy.max(self.imgUsed)
            
            self.imgShape = list(self.imgUsed.shape)
            
            imgShowPlace = numpy.round(self.imgShape[2] / 2).astype(int)
            self.imgShow = self.imgUsed[:, :, imgShowPlace]
            
            self.fig = matpyplot.figure()
            # Pridani subplotu do okna (do figure)
            self.ax1 = self.fig.add_subplot(111)
            
            # Upraveni subplotu
            self.fig.subplots_adjust(left = 0.1, bottom = 0.25)
            # Vykresli obrazek
    #        self.im0 = self.ax0.imshow(imgUsed)
            self.im1 = self.ax1.imshow(self.imgShow, self.cmap)

     #       self.fig.colorbar(self.im1)
    
            # Zakladni informace o slideru
            axcolor = 'white' # lightgoldenrodyellow
            axmin = self.fig.add_axes([0.25, 0.16, 0.495, 0.03], axisbg = axcolor)
            axmax  = self.fig.add_axes([0.25, 0.12, 0.495, 0.03], axisbg = axcolor)
            
            # Vytvoreni slideru
                # Minimalni pouzita hodnota v obrazku
            min0 = numpy.amin(self.imgUsed)
                # Maximalni pouzita hodnota v obrazku
            max0 = numpy.amax(self.imgUsed)
                # Vlastni vytvoreni slideru
                
            self.smin = Slider(axmin, 'Minimal threshold', min0, max0, valinit = min0)
            self.smax = Slider(axmax, 'Maximal threshold', min0, max0, valinit = max0)
            
            self.smin.on_changed(self.updateImg3D)
            self.smax.on_changed(self.updateImg3D)
            
        else:
            
            print('Wrong input.\nDimension of input is not 2 or 3.\nExiting.')

    def showPlot(self):
        
        # Zobrazeni plot (figure)
        matpyplot.show() 
        return self.imgOutput

    def updateImg2D(self, val):
        
        # Prahovani (smin, smax)
        img1 = self.imgUsed > self.smin.val
        self.imgChanged = img1 #< self.smax.val
        
        # Predani obrazku k vykresleni
        self.im1 = self.ax1.imshow(self.imgChanged, self.cmap)
        # Prekresleni
        self.fig.canvas.draw()
        
    def updateImg3D(self, val):
        
        # Prahovani (smin, smax)
        round = 0
        for round in range(self.imgShape[2]):
            im1 = self.imgUsed[:, :, round] > self.smin.val
            self.imgOutput[:, :, round] = (im1) #< self.smax.val
        
        # Predani obrazku k vykresleni
        self.imgShow = numpy.amax(self.imgOutput, 2) # self.imgOutput[:, :, self.imgShowPlace]
        self.im1 = self.ax1.imshow(self.imgShow, self.cmap)
        
        # Prekresleni
        self.fig.canvas.draw()
    
"""
================================================================================
main
================================================================================
"""
"""
    # Vyzve uzivatele k zadani jmena souboru.
#    fileName = input('Give me a filename: ')
    # Precteni souboru (obrazku)
#    imgLoaded = matplotlib.image.imread(fileName)
    imgLoaded = misc.lena()
    # Vytvoreni uiThreshold
    ui = uiThreshold(imgLoaded)
"""
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    ch = logging.StreamHandler()
    logging.basicConfig(format='%(message)s')

    formatter = logging.Formatter("%(levelname)-5s [%(module)s:%(funcName)s:%(lineno)d] %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    parser = argparse.ArgumentParser(description='Segment vessels from liver')
    parser.add_argument('-f','--filename',
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

    if args.filename == 'lena':
        data = misc.lena()
    else:
        mat = scipy.io.loadmat(args.filename)
        logger.debug(mat.keys())

        dataraw = scipy.io.loadmat(args.filename, variable_names=['data'])
        data = dataraw['data']

    ui = uiThreshold(data)
    output = ui.showPlot()

    scipy.io.savemat(args.outputfile, {'data':output})





