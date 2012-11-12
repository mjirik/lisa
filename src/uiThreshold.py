# -*- coding: utf-8 -*-
#===============================================================================
# Name:        uiThreshold
# Purpose:
#
# Author:      Pavel Volkovinsky (volkovinsky.pavel@gmail.com)
#
# Created:     08.11.2012
# Copyright:   (c) PavelVolkovinsky 2012
# Licence:     <your licence>
#===============================================================================
VERSION = "0.0.1"

try:
    import unittest
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

    #from scipy import ndimage

except ImportError, err:
    print "Critical error! Couldn't load module! %s" % (err)
    sys.exit(2)

"""
================================================================================
tests
================================================================================
"""

class Tests(unittest.TestCase):
    def test_t(self):
        pass
    def setUp(self):
        """ Nastavení společných proměnných pro testy  """
        datashape = [220,115,30]
        self.datashape = datashape
        self.rnddata = nump.random.rand(datashape[0], datashape[1], datashape[2])
        self.segmcube = nump.zeros(datashape)
        self.segmcube[130:190, 40:90,5:15] = 1

    def test_same_size_inumput_and_output(self):
        """Funkce testuje stejnost vstupních a výstupních dat"""
        outputdata = vesselSegmentation(self.rnddata,self.segmcube)
        self.assertEqual(outputdata.shape, self.rnddata.shape)


#
#    def test_different_data_and_segmentation_size(self):
#        """ Funkce ověřuje vyhození výjimky při různém velikosti vstpních
#        dat a segmentace """
#        pdb.set_trace();
#        self.assertRaises(Exception, vesselSegmentation, (self.rnddata, self.segmcube[2:,:,:]) )
#

"""
================================================================================
main
================================================================================
"""

#raise Exception('Inumput size error','Shape if inumput data and segmentation must be same')

if __name__ == '__main__':
    main()

def main():

    # Vyzve uzivatele k zadani jmena souboru.
    info = raw_inumput("Give me a filename: ")
    if(info == 'x'):
        fileName = 'morpho.png'
    else:
        fileName = info

    imgLoaded = matplotlib.image.imread(fileName)
    print 'Image dtype: %s' % (imgLoaded.dtype)
    print 'Image size: %6d' % (imgLoaded.size)
    print 'Image shape: %3dx%3d' % (imgLoaded.shape[0], imgLoaded.shape[1])
    print 'Max value %1.2f at pixel %6d' % (imgLoaded.max(), imgLoaded.argmax())
    print 'Min value %1.2f at pixel %6d' % (imgLoaded.min(), imgLoaded.argmin())
    print 'Variance: %1.5f' % (imgLoaded.var())
    print 'Standard deviation: %1.5f' % (imgLoaded.std())

    fig = matpyplot.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left = 0.05, bottom = 0.25)

    #im = max0 * nump.random.random((10,10))
    im1 = ax.imshow(imgLoaded) # tady byl 'im' (dekl. o radek vyse)
    fig.colorbar(im1)

    axcolor = 'white' # lightgoldenrodyellow
    axmin = fig.add_axes([0.15, 0.1, 0.65, 0.03], axisbg = axcolor)
    axmax  = fig.add_axes([0.15, 0.15, 0.65, 0.03], axisbg = axcolor)

    min0 = 100
    max0 = 900
    smin = Slider(axmin, 'Min', 0, 1000, valinit = min0)
    smax = Slider(axmax, 'Max', 0, 1000, valinit = max0)

    def update(val):
        im1.set_clim([smin.val, smax.val])
        fig.canvas.draw()

    smin.on_changed(update)
    smax.on_changed(update)

    matpyplot.show()

























