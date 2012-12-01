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
#import argparse
sys.path.append("../src/")
sys.path.append("../extern/")
import logging
logger = logging.getLogger(__name__)
import scipy.io
import unittest
import argparse
#import pylab
import matplotlib.pyplot as matpyplot
#import numpy
import matplotlib
from matplotlib.widgets import Slider#, Button, RadioButtons
from scipy import ndimage
from scipy import misc
import numpy

"""
================================================================================
uiThreshold
================================================================================
"""
class uiThreshold:

    def __init__(self, imgUsed):

        self.imgUsed = imgUsed
        self.imgChanged = imgUsed
        
        inputDimension = numpy.ndim(imgUsed)
        print('Dimension of input: ',  inputDimension)
        if(inputDimension == 2):
            
            """
            self.imgChanged1 = self.imgUsed
            self.imgChanged2 = self.imgUsed
            self.imgChanged3 = self.imgUsed
            """
            
            # Zakladni informace o obrazku (+ statisticke)
            print('Image dtype: ', imgUsed.dtype)
            print('Image size: ', imgUsed.size)
            print('Image shape: ', imgUsed.shape[0], ' x ',  imgUsed.shape[1])
            print('Max value: ', imgUsed.max(), ' at pixel ',  imgUsed.argmax())
            print('Min value: ', imgUsed.min(), ' at pixel ',  imgUsed.argmin())
            print('Variance: ', imgUsed.var())
            print('Standard deviation: ', imgUsed.std())
            
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
            self.im1 = self.ax1.imshow(imgUsed)
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
            self.smin.on_changed(self.updateImg)
            self.smax.on_changed(self.updateImg)
            """
            self.sopen.on_changed(self.updateImg)
            self.sclose.on_changed(self.updateImg)
            """
            
            # Zobrazeni plot (figure)
            matpyplot.show() 
            
        
        elif(inputDimension == 3):
            
            print('Sorry! Not yet implemented for 3 dimensions!\nExiting.')
            
        else:
            print('Wrong input.\nDimension of input is not 2 or 3.\nExiting.')

    def updateImg(self, val):
        
        # Prahovani (smin, smax)
        img1 = self.imgUsed > self.smin.val
        self.imgChanged = img1# < self.smax.val
        
        """
        # Binary opening a binary closing
        img3 = ndimage.binary_opening(img2)
        self.imgChanged = ndimage.binary_closing(img3)
        """
        
        # Predani obrazku k vykresleni
        self.im1 = self.ax1.imshow(self.imgChanged)
        # Prekresleni
        self.fig.canvas.draw()
    """
    def updateMinThreshold(self, val):

        # Prahovani
        self.imgChanged1 = self.imgUsed > val
        # Predani obrazku k vykresleni
        self.im1 = self.ax1.imshow(self.imgChanged1)
        # Prekresleni
        self.fig.canvas.draw()
    
    def updateMaxThreshold(self, val):

        # Prahovani
        self.imgChanged1 = self.imgUsed < val
        # Predani obrazku k vykresleni
        self.im1 = self.ax1.imshow(self.imgChanged1)
        # Prekresleni
        self.fig.canvas.draw()
    
    def updateBinOpening(self, val):

        # Prahovani
        self.imgChanged2 = ndimage.binary_opening(self.imgChanged1, None, int(round(val, 0)))
        # Predani obrazku k vykresleni
        self.im2 = self.ax2.imshow(self.imgChanged2)
        # Prekresleni
        self.fig.canvas.draw()
        
    def updateBinClosing(self, val):

        # Prahovani
        self.imgChanged3 = ndimage.binary_closing(self.imgChanged2, None, int(round(val, 0)))
        # Predani obrazku k vykresleni
        self.im3 = self.ax3.imshow(self.imgChanged3)
        # Prekresleni
        self.fig.canvas.draw()
    """
"""
================================================================================
main
================================================================================
"""
"""
if __name__ == '__main__':

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
    #   load all
        mat = scipy.io.loadmat(args.filename)
        logger.debug(mat.keys())

        dataraw = scipy.io.loadmat(args.filename, variable_names=['data'])
        data = dataraw['data']

    ui = uiThreshold(data)
    output = ui.show()

    scipy.io.savemat(args.outputfile, {'data':output})





