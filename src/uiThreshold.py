# -*- coding: utf-8 -*-
"""
================================================================================
Name:        uiThreshold
Purpose:     (CZE-ZCU-FAV-KY) Liver medical project

Author:      Pavel Volkovinsky (volkovinsky.pavel@gmail.com)

Created:     08.11.2012
Copyright:   (c) Pavel Volkovinsky 2012
Licence:     <your licence>
================================================================================
"""
#VERSION = "0.0.2"

try:
    #import unittest
    import sys
    import argparse
    sys.path.append("../src/")
    sys.path.append("../extern/")

    import logging
    logger = logging.getLogger(__name__)

    import pylab as pylab
    import matplotlib.pyplot as matpyplot
    import numpy as nump
    import matplotlib
    from matplotlib.widgets import Slider, Button, RadioButtons

except ImportError, err:
    print "Critical error! Couldn't load module! %s" % (err)
    sys.exit(2)

"""
================================================================================
class uiThreshold
================================================================================
"""

class uiThreshold:

    def __init__(self, imgUsed):

        self.imgUsed = imgUsed
        self.imgChanged = self.imgUsed

        print 'Image dtype: %s' % (imgUsed.dtype)
        print 'Image size: %6d' % (imgUsed.size)
        print 'Image shape: %3dx%3d' % (imgUsed.shape[0], imgUsed.shape[1])
        print 'Max value %1.2f at pixel %6d' % (imgUsed.max(), imgUsed.argmax())
        print 'Min value %1.2f at pixel %6d' % (imgUsed.min(), imgUsed.argmin())
        print 'Variance: %1.5f' % (imgUsed.var())
        print 'Standard deviation: %1.5f' % (imgUsed.std())

        # Ziskani okna (figure)
        self.fig = matpyplot.figure()
        # Pridani subplotu do okna (do figure)
        self.ax = self.fig.add_subplot(111)
        # Upraveni subplotu
        self.fig.subplots_adjust(left = 0.05, bottom = 0.25)
        # Vykresli obrazek
        self.im1 = self.ax.imshow(imgUsed)
        #self.fig.colorbar(self.im1)

        # Zakladni informace o slideru
        axcolor = 'white' # lightgoldenrodyellow
        axmin = self.fig.add_axes([0.15, 0.1, 0.65, 0.03], axisbg = axcolor)
        axmax  = self.fig.add_axes([0.15, 0.15, 0.65, 0.03], axisbg = axcolor)

        # Vytvoreni slideru
            # Minimalni pouzita hodnota v obrazku
        min0 = imgUsed.min()
            # Maximalni pouzita hodnota v obrazku
        max0 = imgUsed.max()
            # Vlastni vytvoreni slideru
        self.smin = Slider(axmin, 'Min', min0, max0, valinit = min0)
        self.smax = Slider(axmax, 'Max', min0, max0, valinit = max0)

        # Udalost pri zmene hodnot slideru - volani updatu
        self.smin.on_changed(self.updateMin)
        self.smax.on_changed(self.updateMax)

        # Zobrazeni okna (plot)
        matpyplot.show()

    def updateMin(self, val):

        # Prahovani
        self.imgChanged = self.imgUsed > self.smin.val
        # Predani obrazku k vykresleni
        self.im1 = self.ax.imshow(self.imgChanged)
        # Prekresleni
        self.fig.canvas.draw()

    def updateMax(self, val):

        # Prahovani
        self.imgChanged = self.imgUsed < self.smax.val
        # Predani obrazku k vykresleni
        self.im1 = self.ax.imshow(self.imgChanged)
        # Prekresleni
        self.fig.canvas.draw()

"""
================================================================================
main
================================================================================
"""

if __name__ == '__main__':

    # Vyzve uzivatele k zadani jmena souboru.
    fileName = raw_input("Give me a filename: ")
    # Precteni souboru (obrazku)
    imgLoaded = matplotlib.image.imread(fileName)
    # Vytvoreni uiThreshold
    ui = uiThreshold(imgLoaded)
