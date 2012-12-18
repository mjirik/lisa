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

import scipy.ndimage
"""
import scipy.misc
import scipy.io

import unittest
import argparse
"""

import matplotlib.pyplot as matpyplot
import matplotlib
from matplotlib.widgets import Slider, Button#, RadioButtons

"""
================================================================================
uiThreshold
================================================================================
"""
class uiThreshold:

    def __init__(self, data, number = 100.0, voxelV = 1.0, initslice = 0, cmap = matplotlib.cm.Greys_r):

        inputDimension = numpy.ndim(data)
        #print('Dimenze vstupu: ',  inputDimension)
        self.cmap = cmap
        self.number = number
        self.data = data
        
        if(inputDimension == 2):
            
            self.imgUsed = data
            self.imgChanged = data
                
            """
            self.imgChanged1 = self.imgUsed
            self.imgChanged2 = self.imgUsed
            self.imgChanged3 = self.imgUsed
            """
            
            ## Zakladni informace o obrazku (+ statisticke)
            """
            print('Image dtype: ', imgUsed.dtype)
            print('Image size: ', imgUsed.size)
            print('Image shape: ', imgUsed.shape[0], ' x ',  imgUsed.shape[1])
            print('Max value: ', imgUsed.max(), ' at pixel ',  imgUsed.argmax())
            print('Min value: ', imgUsed.min(), ' at pixel ',  imgUsed.argmin())
            print('Variance: ', imgUsed.var())
            print('Standard deviation: ', imgUsed.std())
            """
            
            ## Ziskani okna (figure)
            self.fig = matpyplot.figure()
            ## Pridani subplotu do okna (do figure)
            self.ax1 = self.fig.add_subplot(111)
            """
    #        self.ax0 = self.fig.add_subplot(232)
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)
            """
            ## Upraveni subplotu
            self.fig.subplots_adjust(left = 0.1, bottom = 0.25)
            ## Vykresli obrazek
    #        self.im0 = self.ax0.imshow(imgUsed)
            self.im1 = self.ax1.imshow(self.imgChanged, self.cmap)
            """
            self.im2 = self.ax2.imshow(imgUsed)
            self.im3 = self.ax3.imshow(imgUsed)
            """
     #       self.fig.colorbar(self.im1)
    
            ## Zakladni informace o slideru
            axcolor = 'white' # lightgoldenrodyellow
            axmin = self.fig.add_axes([0.25, 0.16, 0.495, 0.03], axisbg = axcolor)
            axmax  = self.fig.add_axes([0.25, 0.12, 0.495, 0.03], axisbg = axcolor)
            """
            axopening = self.fig.add_axes([0.25, 0.08, 0.495, 0.03], axisbg = axcolor)
            axclosing = self.fig.add_axes([0.25, 0.04, 0.495, 0.03], axisbg = axcolor)
            """
            
            ## Vytvoreni slideru
                ## Minimalni pouzita hodnota v obrazku
            min0 = data.min()
                ## Maximalni pouzita hodnota v obrazku
            max0 = data.max()
                ## Vlastni vytvoreni slideru
            self.smin = Slider(axmin, 'Minimal threshold', min0, max0, valinit = min0)
            self.smax = Slider(axmax, 'Maximal threshold', min0, max0, valinit = max0)
            """
            self.sopen = Slider(axopening, 'Binary opening', 0, 10, valinit = 0)
            self.sclose = Slider(axclosing, 'Binary closing', 0, 10, valinit = 0)
            """
            
            ## Udalost pri zmene hodnot slideru - volani updatu
            self.smin.on_changed(self.updateImg2D)
            self.smax.on_changed(self.updateImg2D)
        
        elif(inputDimension == 3):
            
            self.voxelV = voxelV
            
            ## Zakladni informace o obrazcich (+ statisticke)
            """
            print('Image dtype: ', imgUsed.dtype)
            print('Image size: ', imgUsed.size)
            print('Image shape: ', imgUsed.shape[0], ' x ',  imgUsed.shape[1], ' x ',  imgUsed.shape[2])
            print('Max value: ', imgUsed.max(), ' at pixel ',  imgUsed.argmax())
            print('Min value: ', imgUsed.min(), ' at pixel ',  imgUsed.argmin())
            print('Variance: ', imgUsed.var())
            print('Standard deviation: ', imgUsed.std())
            """
            
            self.imgUsed = data
            self.imgChanged = self.imgUsed
            
            #self.imgMin = numpy.min(self.imgUsed)
            #self.imgMax = numpy.max(self.imgUsed)
            
            self.imgShape = list(self.imgUsed.shape)
            
            self.fig = matpyplot.figure()
            ## Pridani subplotu do okna (do figure)
#            self.ax1n = self.fig.add_subplot(231)
#            self.ax2n = self.fig.add_subplot(232)
#            self.ax3n = self.fig.add_subplot(233)
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)
            
            ## Upraveni subplotu
            self.fig.subplots_adjust(left = 0.1, bottom = 0.3)
            
            ## Vykreslit obrazek
#            self.im1n = self.ax1n.imshow(numpy.amax(self.data, 0), self.cmap)
#            self.im2n = self.ax2n.imshow(numpy.amax(self.data, 1), self.cmap)
#            self.im3n = self.ax3n.imshow(numpy.amax(self.data, 2), self.cmap)
            self.im1 = self.ax1.imshow(numpy.amax(self.imgUsed, 0), self.cmap)
            self.im2 = self.ax2.imshow(numpy.amax(self.imgUsed, 1), self.cmap)
            self.im3 = self.ax3.imshow(numpy.amax(self.imgUsed, 2), self.cmap)
    
            ## Zakladni informace o slideru
            self.axcolor = 'white' # lightgoldenrodyellow
            self.axmin = self.fig.add_axes([0.20, 0.20, 0.55, 0.03], axisbg = self.axcolor)
            self.axmax  = self.fig.add_axes([0.20, 0.16, 0.55, 0.03], axisbg = self.axcolor)
            self.axsigma = self.fig.add_axes([0.20, 0.08, 0.55, 0.03], axisbg = self.axcolor)
            
            ## Vytvoreni slideru
                ## Minimalni pouzita hodnota prahovani v obrazku
            self.min0 = numpy.amin(self.imgUsed)
                ## Maximalni pouzita hodnota prahovani v obrazku
            self.max0 = numpy.amax(self.imgUsed)
                ## Vlastni vytvoreni slideru
                
            self.smin = Slider(self.axmin, 'Minimal threshold', self.min0, self.max0, valinit = self.min0)
            self.smax = Slider(self.axmax, 'Maximal threshold', self.min0, self.max0, valinit = self.max0)
            self.ssigma = Slider(self.axsigma, 'Sigma', 0.00, self.number * 2, valinit = self.number)
            
            self.smin.on_changed(self.updateImg1Threshold3D)
            self.smax.on_changed(self.updateImg1Threshold3D)
            self.ssigma.on_changed(self.updateImgFilter)
            
            self.axbuttnext = self.fig.add_axes([0.81, 0.20, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttprev = self.fig.add_axes([0.86, 0.20, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttreset = self.fig.add_axes([0.81, 0.08, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttcontinue = self.fig.add_axes([0.90, 0.04, 0.04, 0.03], axisbg = self.axcolor)
            
            self.bnext = Button(self.axbuttnext, '+1.0')
            self.bprev = Button(self.axbuttprev, '-1.0')
            self.breset = Button(self.axbuttreset, 'Reset')
            self.bcontinue = Button(self.axbuttcontinue, 'Next UI')
            
            self.bnext.on_clicked(self.button3DNext)
            self.bprev.on_clicked(self.button3DPrev)
            self.breset.on_clicked(self.button3DReset)
            self.bcontinue.on_clicked(self.button3DContinue)

        else:
            
            print('Spatny vstup.\nDimenze vstupu neni 2 ani 3.\nUkoncuji prahovani.')

    def showPlot(self):
        
        ## Zobrazeni plot (figure)
        matpyplot.show()
        
        return self.imgChanged 
        
    def updateImgFilter(self, val):
        
        ## Vypocet sigma pro gauss. filtr
        sigma = float(self.ssigma.val) / self.voxelV
        
        ## Filtrovani
        images = self.data
        imgUsed = images
        scipy.ndimage.filters.gaussian_filter(images, sigma, 0, imgUsed, 'reflect', 0.0)
        self.imgUsed = imgUsed
            
        ## Vykresleni novych pohledu
        self.im1 = self.ax1.imshow(numpy.amax(self.imgUsed, 0), self.cmap)
        self.im2 = self.ax2.imshow(numpy.amax(self.imgUsed, 1), self.cmap)
        self.im3 = self.ax3.imshow(numpy.amax(self.imgUsed, 2), self.cmap)
        
#        ## Zmena maximalnich a minimalnich hodnot os prahovani
#            ## Minimalni pouzita hodnota prahovani v obrazku
#        min0 = numpy.amin(self.imgUsed)
#            ## Maximalni pouzita hodnota prahovani v obrazku
#        max0 = numpy.amax(self.imgUsed)
#            ## Vlastni vytvoreni slideru 
#        self.smin = Slider(self.axmin, 'Minimal threshold', min0, max0, valinit = min0)
#        self.smax = Slider(self.axmax, 'Maximal threshold', min0, max0, valinit = max0)
#        self.smin.on_changed(self.updateImg1Threshold3D)
#        self.smax.on_changed(self.updateImg1Threshold3D)
        
        ## Prekresleni
        self.fig.canvas.draw()
        
    def button3DReset(self, event):
        
        self.imgUsed = self.data
        
        ## Vykresleni novych pohledu
        self.im1 = self.ax1.imshow(numpy.amax(self.imgUsed, 0), self.cmap)
        self.im2 = self.ax2.imshow(numpy.amax(self.imgUsed, 1), self.cmap)
        self.im3 = self.ax3.imshow(numpy.amax(self.imgUsed, 2), self.cmap)
        
#        ## Zmena maximalnich a minimalnich hodnot os prahovani
#            ## Minimalni pouzita hodnota prahovani v obrazku
#        min0 = numpy.amin(self.imgUsed)
#            ## Maximalni pouzita hodnota prahovani v obrazku
#        max0 = numpy.amax(self.imgUsed)
#            ## Vlastni vytvoreni slideru
#        self.smin = Slider(self.axmin, 'Minimal threshold', min0, max0, valinit = min0)
#        self.smax = Slider(self.axmax, 'Maximal threshold', min0, max0, valinit = max0)
#        self.ssigma = Slider(self.axsigma, 'Sigma', 0.00, self.number * 2, valinit = self.number)
#        self.smin.on_changed(self.updateImg1Threshold3D)
#        self.smax.on_changed(self.updateImg1Threshold3D)
#        self.ssigma.on_changed(self.updateImgFilter)

        self.smin.val = (numpy.round(self.min0, 2))
        self.smin.valtext.set_text('{}'.format(self.smin.val))
        self.smax.val = (numpy.round(self.max0, 2))
        self.smax.valtext.set_text('{}'.format(self.smax.val))
        self.ssigma.val = (numpy.round(0.00, 2))
        self.ssigma.valtext.set_text('{}'.format(self.ssigma.val))
        
        ## Prekresleni
        self.fig.canvas.draw()
        
    def button3DContinue(self, event):
        
        matpyplot.clf()
        matpyplot.close()
        
    def button3DNext(self, event):
        
        self.smin.val += 1.0
        self.smin.val = (numpy.round(self.smin.val, 2))
        self.smin.valtext.set_text('{}'.format(self.smin.val))
        self.fig.canvas.draw()
        self.updateImg1Threshold3D(self) #self, self.smin.val
        
    def button3DPrev(self, event):
        
        self.smin.val -= 1.0
        self.smin.val = (numpy.round(self.smin.val, 2))
        self.smin.valtext.set_text('{}'.format(self.smin.val))
        self.fig.canvas.draw()
        self.updateImg1Threshold3D(self) #self, self.smin.val

    def updateImg2D(self, val):
        
        ## Prahovani (smin, smax)
        img1 = self.imgUsed.copy() > self.smin.val
        self.imgChanged = img1 #< self.smax.val
        
        ## Predani obrazku k vykresleni
        self.im1 = self.ax1.imshow(self.imgChanged, self.cmap)
        ## Prekresleni
        self.fig.canvas.draw()
        
    def updateImg1Threshold3D(self, val):
        
        ## Prahovani (smin, smax)
        self.imgChanged = (self.imgUsed > self.smin.val) & (self.imgUsed < self.smax.val)
        
        ## Predani obrazku k vykresleni
        self.im1 = self.ax1.imshow(numpy.amax(self.imgChanged, 0), self.cmap)
        self.im2 = self.ax2.imshow(numpy.amax(self.imgChanged, 1), self.cmap)
        self.im3 = self.ax3.imshow(numpy.amax(self.imgChanged, 2), self.cmap)
        
        ## Prekresleni
        self.fig.canvas.draw()
    
"""
================================================================================
main
================================================================================
"""
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
        data = scipy.misc.lena()
    else:
        mat = scipy.io.loadmat(args.filename)
        logger.debug(mat.keys())
        
        data = mat['data'] * (mat['segmentation'] == 1)

    ui = uiThreshold(data)
    output = ui.showPlot()

    scipy.io.savemat(args.outputfile, {'data':output})
    sys.exit()
    
"""





