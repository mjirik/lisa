# -*- coding: utf-8 -*-
"""
================================================================================
Name:        uiBinaryClosingAndOpening
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
#import scipy.misc
#import scipy.io
import scipy.ndimage

#import unittest
#import argparse

import matplotlib.pyplot as matpyplot
import matplotlib
from matplotlib.widgets import Slider, Button#, RadioButtons

"""
================================================================================
uiBinaryClosingAndOpening
================================================================================
"""
class uiBinaryClosingAndOpening:

    ## data - data pro operace binary closing a opening, se kterymi se pracuje
    ## initslice - PROZATIM NEPOUZITO
    ## cmap - grey
    def __init__(self, data, initslice = 0, cmap = matplotlib.cm.Greys_r):

        inputDimension = numpy.ndim(data)
        #print('Dimenze vstupu: ',  inputDimension)
        self.cmap = cmap
        self.imgUsed = data
        self.imgChanged = self.imgUsed
        self.imgChanged1 = self.imgUsed
#        self.imgChanged2 = imgUsed
        
        if(inputDimension == 2):
            
            self.imgUsed = self.imgUsed
            self.imgChanged = self.imgUsed
                
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
            self.im1 = self.ax1.imshow(self.imgChanged, self.cmap)
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
            min0 = self.imgUsed.min()
                # Maximalni pouzita hodnota v obrazku
            max0 = self.imgUsed.max()
                # Vlastni vytvoreni slideru
            self.smin = Slider(axmin, 'Minimal threshold', min0, max0, valinit = min0)
            self.smax = Slider(axmax, 'Maximal threshold', min0, max0, valinit = max0)
            """
            self.sopen = Slider(axopening, 'Binary opening', 0, 10, valinit = 0)
            self.sclose = Slider(axclosing, 'Binary closing', 0, 10, valinit = 0)
            """
            
            # Udalost pri zmene hodnot slideru - volani updatu
            self.smin.on_changed(self.updateImg2D)
            self.smax.on_changed(self.updateImg2D)
        
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
            
            #self.imgMin = numpy.min(self.imgUsed)
            #self.imgMax = numpy.max(self.imgUsed)
            
            self.imgShape = list(self.imgUsed.shape)
            
            self.fig = matpyplot.figure()
            # Pridani subplotu do okna (do figure)
            self.ax1 = self.fig.add_subplot(111)
            #self.ax2 = self.fig.add_subplot(122)
            
            # Upraveni subplotu
            self.fig.subplots_adjust(left = 0.1, bottom = 0.3)
            
            # Nalezeni a pripraveni obrazku k vykresleni
     #       imgShowPlace = numpy.round(self.imgShape[2] / 2).astype(int)
     #       self.imgShow = self.imgUsed[:, :, imgShowPlace]
            self.imgShow = numpy.amax(self.imgChanged, 2)
            
            # Vykreslit obrazek
            self.im1 = self.ax1.imshow(self.imgShow, self.cmap)
            #self.im2 = self.ax2.imshow(self.imgShow, self.cmap)
    
            # Zakladni informace o slideru
            self.axcolor = 'white' # lightgoldenrodyellow
            axopening1 = self.fig.add_axes([0.25, 0.18, 0.55, 0.03], axisbg = self.axcolor)
            axclosing1 = self.fig.add_axes([0.25, 0.14, 0.55, 0.03], axisbg = self.axcolor)
            #axopening2 = self.fig.add_axes([0.25, 0.04, 0.495, 0.03], axisbg = self.axcolor)
            #axclosing2 = self.fig.add_axes([0.25, 0.08, 0.495, 0.03], axisbg = self.axcolor)
            
            # Vytvoreni slideru
            self.sopen1 = Slider(axopening1, 'Binary opening 1', 0, 100, valinit = 0)
            self.sclose1 = Slider(axclosing1, 'Binary closing 1', 0, 100, valinit = 0)
            #self.sopen2 = Slider(axopening2, 'Binary opening 2', 0, 100, valinit = 0)
            #self.sclose2 = Slider(axclosing2, 'Binary closing 2', 0, 100, valinit = 0)
            
            self.sopen1.on_changed(self.updateImg1Binary3D)
            self.sclose1.on_changed(self.updateImg1Binary3D)
            #self.sopen2.on_changed(self.updateImg2Binary3D)
            #self.sclose2.on_changed(self.updateImg2Binary3D)
            
            self.sopen1.valtext.set_text('{}'.format(int(self.sopen1.val)))
            self.sclose1.valtext.set_text('{}'.format(int(self.sclose1.val)))
            
            self.axbuttnextopening = self.fig.add_axes([0.83, 0.18, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttprevopening = self.fig.add_axes([0.88, 0.18, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttnextclosing = self.fig.add_axes([0.83, 0.14, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttprevclosing = self.fig.add_axes([0.88, 0.14, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttreset = self.fig.add_axes([0.83, 0.08, 0.04, 0.03], axisbg = self.axcolor)
            self.axbuttcontinue = self.fig.add_axes([0.88, 0.08, 0.06, 0.03], axisbg = self.axcolor)
            self.axbuttswap = self.fig.add_axes([0.05, 0.18, 0.09, 0.03], axisbg = self.axcolor)
            
            self.bnextopening = Button(self.axbuttnextopening, '+1.0')
            self.bprevopening = Button(self.axbuttprevopening, '-1.0')
            self.bnextclosing = Button(self.axbuttnextclosing, '+1.0')
            self.bprevclosing = Button(self.axbuttprevclosing, '-1.0')
            self.breset = Button(self.axbuttreset, 'Reset')
            self.bcontinue = Button(self.axbuttcontinue, 'End editing')
            self.bswap = Button(self.axbuttswap, 'Swap operations')
            
            self.bnextopening.on_clicked(self.button3DNextOpening)
            self.bprevopening.on_clicked(self.button3DPrevOpening)
            self.bnextclosing.on_clicked(self.button3DNextClosing)
            self.bprevclosing.on_clicked(self.button3DPrevClosing)
            self.breset.on_clicked(self.button3DReset)
            self.bcontinue.on_clicked(self.button3DContinue)
            self.bswap.on_clicked(self.buttonSwap)
            
            self.state = 'firstOpening'
            self.text = matpyplot.figtext(0.05, 0.15, 'First: opening')
            
        else:
            
            print('Spatny vstup.\nDimenze vstupu neni 2 ani 3.\nUkoncuji prahovani.')

    def showPlot(self):
        
        # Zobrazeni plot (figure)
        matpyplot.show()
        
        return self.imgChanged1
        
    def buttonSwap(self, event):
        
        if(self.state == 'firstOpening'):
            self.state = 'firstClosing'
            #matpyplot.figtext(0.05, 0.15, 'First: closing')
        elif(self.state == 'firstClosing'):
            self.state = 'firstOpening'
            #matpyplot.figtext(0.05, 0.15, 'First: opening')
        
        self.fig.canvas.draw()
        
    def button3DReset(self, event):
        
        self.sopen1.val = 0.0
        self.sopen1.valtext.set_text('{}'.format(int(self.sopen1.val)))
        self.sclose1.val = 0.0
        self.sclose1.valtext.set_text('{}'.format(int(self.sclose1.val)))
        
        self.imgShow = numpy.amax(self.imgChanged, 2)
            
        ## Vykreslit obrazek
        self.im1 = self.ax1.imshow(self.imgShow, self.cmap)
        
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
        self.sopen1.valtext.set_text('{}'.format(int(numpy.round(self.sopen1.val, 0))))
        self.sclose1.valtext.set_text('{}'.format(int(numpy.round(self.sclose1.val, 0))))
        
        ## Prekresleni
        self.fig.canvas.draw()
        
        imgChanged1 = self.imgChanged
        
        ## Prvni operace opening, pote closing
        if(self.state == 'firstOpening'):
            
            if(self.sopen1.val >= 0.1):
                imgChanged1 = scipy.ndimage.binary_opening(self.imgChanged, structure = None, iterations = int(numpy.round(self.sopen1.val, 0)))
            else:
                imgChanged1 = self.imgChanged
                
            if(self.sclose1.val >= 0.1):
                self.imgChanged1 = scipy.ndimage.binary_closing(imgChanged1, structure = None, iterations = int(numpy.round(self.sclose1.val, 0)))
            else:
                self.imgChanged1 = imgChanged1
            
        ## Prvni operace closing, pote opening
        elif(self.state == 'firstClosing'):
            
            if(self.sclose1.val >= 0.1):
                imgChanged1 = scipy.ndimage.binary_closing(self.imgChanged, structure = None, iterations = int(numpy.round(self.sclose1.val, 0)))
            else:
                imgChanged1 = self.imgChanged
            
            if(self.sopen1.val >= 0.1):
                self.imgChanged1 = scipy.ndimage.binary_opening(imgChanged1, structure = None, iterations = int(numpy.round(self.sopen1.val, 0)))
            else:
                self.imgChanged1 = imgChanged1
            
        ## Predani obrazku k vykresleni
        self.imgShow1 = numpy.amax(self.imgChanged1, 2)
        self.im1 = self.ax1.imshow(self.imgShow1, self.cmap)
        
        ## Prekresleni
        self.fig.canvas.draw()
        
"""
    def updateImg2Binary3D(self, val):
        
        self.sclose2.valtext.set_text('{}'.format(int(self.sclose2.val)))
        self.sopen2.valtext.set_text('{}'.format(int(self.sopen2.val)))
        
        self.fig.canvas.draw()
        
        imgChanged2 = self.imgChanged
        
        if(self.sclose2.val >= 0.5):
            imgChanged2 = scipy.ndimage.binary_closing(self.imgChanged, structure = None, iterations = int(numpy.round(self.sclose2.val, 0)))
        else:
            imgChanged2 = self.imgChanged
        
        if(self.sopen2.val >= 0.5):
            self.imgChanged2 = scipy.ndimage.binary_opening(imgChanged2, structure = None, iterations = int(numpy.round(self.sopen2.val, 0)))
        else:
            self.imgChanged2 = imgChanged2
            
        # Predani obrazku k vykresleni
        self.imgShow2 = numpy.amax(self.imgChanged2, 2)
        self.im2 = self.ax2.imshow(self.imgShow2, self.cmap)
        
        # Prekresleni
        self.fig.canvas.draw()
"""

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

        dataraw = scipy.io.loadmat(args.filename)
        
        data = dataraw['data'] * (dataraw['segmentation'] == 1)

    ui = uiThreshold(data)
    output = ui.showPlot()

    scipy.io.savemat(args.outputfile, {'data':output})
"""




