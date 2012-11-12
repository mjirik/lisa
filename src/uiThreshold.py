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

def mainGauss(imgLoaded):

    from scipy import ndimage
    import numpy as np

    np.random.seed(1)
    n = 10
    l = 256
    im = np.zeros((l, l))

    mask = (im > im.mean()).astype(np.float)
    mask += 0.1 * im
    img = mask + 0.3*np.random.randn(*mask.shape)

    from sklearn.mixture import GMM
    classif = GMM(n_components=2)
    classif.fit(img.reshape((img.size, 1)))
    classif.means_
    array([[ 0.9353155 ], [-0.02966039]])
    np.sqrt(classif.covars_).ravel()
    array([ 0.35074631,  0.28225327])
    classif.weights_
    array([ 0.40989799,  0.59010201])
    threshold = np.mean(classif.means_)
    binary_img = img > threshold

    matpyplot.figure(figsize=(11,4))

    hist, bin_edges = np.histogram(img, bins=60)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    matpyplot.subplot(131)
    matpyplot.imshow(img)
    matpyplot.axis('off')
    matpyplot.subplot(132)
    matpyplot.plot(bin_centers, hist, lw=2)
    matpyplot.axvline(0.5, color='r', ls='--', lw=2)
    matpyplot.text(0.57, 0.8, 'histogram', fontsize=20, transform = matpyplot.gca().transAxes)
    matpyplot.yticks([])
    matpyplot.subplot(133)
    matpyplot.imshow(binary_img, cmap=matpyplot.cm.gray, interpolation='nearest')
    matpyplot.axis('off')

    matpyplot.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
    matpyplot.show()

def mainHist(imgLoaded):

    from scipy import ndimage
    import numpy as np

    np.random.seed(1)
    n = 10
    l = 256
    im = np.zeros((l, l))
    points = l*np.random.random((2, n**2))
    im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

    mask = (im > im.mean()).astype(np.float)

    mask += 0.1 * im

    img = mask + 0.2*np.random.randn(*mask.shape)

    hist, bin_edges = np.histogram(img, bins=60)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    binary_img = img > 0.5

    matpyplot.figure(figsize=(11,4))

    matpyplot.subplot(131)
    matpyplot.imshow(img)
    matpyplot.axis('off')
    matpyplot.subplot(132)
    matpyplot.plot(bin_centers, hist, lw=2)
    matpyplot.axvline(0.5, color='r', ls='--', lw=2)
    matpyplot.text(0.57, 0.8, 'histogram', fontsize=20, transform = matpyplot.gca().transAxes)
    matpyplot.yticks([])
    matpyplot.subplot(133)
    matpyplot.imshow(binary_img, cmap=matpyplot.cm.gray, interpolation='nearest')
    matpyplot.axis('off')

    matpyplot.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
    matpyplot.show()

def mainHistPlusModif(imgInput):

    img = imgInput
    """
    img = pylab.mean(img, 2) # to get a 2-D array
    #pylab.imshow(img)
    pylab.gray()
    """

    from scipy import ndimage
    import numpy as np

    hist, bin_edges = np.histogram(img, bins=60)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    print 'Image dtype: %s' % (img.dtype)
    print 'Image size: %6d' % (img.size)
    print 'Image shape: %3dx%3d' % (img.shape[0], img.shape[1])
    print 'Max value %1.2f at pixel %6d' % (img.max(), img.argmax())
    print 'Min value %1.2f at pixel %6d' % (img.min(), img.argmin())
    print 'Variance: %1.5f' % (img.var())
    print 'Standard deviation: %1.5f' % (img.std())

    binary_img = img > 0.5

    # Remove small white regions
    open_img = ndimage.binary_opening(binary_img)
    # Remove small black hole
    close_img = ndimage.binary_closing(open_img)

    matpyplot.figure(figsize=(12, 9))
    l = img.size
    mask = (img > img.mean()).astype(np.float)
    mask += 0.1 * img

    matpyplot.subplot(231)
    matpyplot.imshow(img)
    matpyplot.axis('off')

    matpyplot.subplot(232)
    matpyplot.plot(bin_centers, hist, lw = 2)
    matpyplot.axvline(img.std(), color='r', ls = '--', lw = 2)
    matpyplot.text(0.57, 0.8, 'histogram', fontsize = 20,
        transform = matpyplot.gca().transAxes)
    matpyplot.yticks([])

    matpyplot.subplot(233)
    matpyplot.imshow(binary_img[:l, :l], cmap = matpyplot.cm.gray)
    matpyplot.axis('off')

    matpyplot.subplot(234)
    matpyplot.imshow(open_img[:l, :l], cmap = matpyplot.cm.gray)
    matpyplot.axis('off')

    matpyplot.subplot(235)
    matpyplot.imshow(close_img[:l, :l], cmap = matpyplot.cm.gray)
    matpyplot.axis('off')

    matpyplot.subplot(236)
    matpyplot.imshow(mask[:l, :l], cmap = matpyplot.cm.gray)
    matpyplot.contour(close_img[:l, :l], [0.5], linewidths = 2, colors = 'r')
    matpyplot.axis('off')

    matpyplot.subplots_adjust(wspace = 0.02, hspace = 0.3, top = 0.9,
        bottom = 0.1, left = 0, right = 0.975)

    matpyplot.show()

def mainHistPlus(imgLoaded):

    from scipy import ndimage
    import numpy as np

    np.random.seed(1)
    n = 20
    l = 256
    im = np.zeros((l, l))
    points = l*np.random.random((2, n**2))
    im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

    mask = (im > im.mean()).astype(np.float)

    mask += 0.1 * im

    img = mask + 0.2*np.random.randn(*mask.shape)

    hist, bin_edges = np.histogram(img, bins=60)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    print 'Image dtype: %s' % (img.dtype)
    print 'Image size: %6d' % (img.size)
    print 'Image shape: %3dx%3d' % (img.shape[0], img.shape[1])
    print 'Max value %1.2f at pixel %6d' % (img.max(), img.argmax())
    print 'Min value %1.2f at pixel %6d' % (img.min(), img.argmin())
    print 'Variance: %1.5f' % (img.var())
    print 'Standard deviation: %1.5f' % (img.std())

    binary_img = img > 0.5

    # Remove small white regions
    open_img = ndimage.binary_opening(binary_img)
    # Remove small black hole
    close_img = ndimage.binary_closing(open_img)

    matpyplot.figure(figsize=(12, 9))

    l = 128

    matpyplot.subplot(231)
    matpyplot.imshow(img)
    matpyplot.axis('off')

    matpyplot.subplot(232)
    matpyplot.plot(bin_centers, hist, lw=2)
    matpyplot.axvline(img.std(), color='r', ls='--', lw=2)
    matpyplot.text(0.57, 0.8, 'histogram', fontsize=20,
        transform = matpyplot.gca().transAxes)
    matpyplot.yticks([])

    matpyplot.subplot(233)
    matpyplot.imshow(binary_img[:l, :l], cmap=matpyplot.cm.gray)
    matpyplot.axis('off')

    matpyplot.subplot(234)
    matpyplot.imshow(open_img[:l, :l], cmap=matpyplot.cm.gray)
    matpyplot.axis('off')

    matpyplot.subplot(235)
    matpyplot.imshow(close_img[:l, :l], cmap=matpyplot.cm.gray)
    matpyplot.axis('off')

    matpyplot.subplot(236)
    matpyplot.imshow(mask[:l, :l], cmap=matpyplot.cm.gray)
    matpyplot.contour(close_img[:l, :l], [0.5], linewidths=2, colors='r')
    matpyplot.axis('off')

    matpyplot.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1,
        left=0, right=1)

    matpyplot.show()

def main(imgLoaded):

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

def setupInput():
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    # při vývoji si necháme vypisovat všechny hlášky
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    #   output configureation
    #logging.basicConfig(format='%(asctime)s %(message)s')
    logging.basicConfig(format='%(message)s')

    formatter = logging.Formatter("%(levelname)-5s [%(module)s:%(funcName)s:%(lineno)d] %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    # input parser
    parser = argparse.ArgumentParser(description='Segment vessels from liver')
    parser.add_argument('filename', type=str,
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
        # hack for use argparse and unittest in one module
        sys.argv[1:]=[]
        unittest.main()

if __name__ == '__main__':

    #setupInput()
    number = raw_input("1: uiThreshold\n2: histogram based\n3: "
        +"histogram (plus) based\n4: Gaussian mixture\n5: "
        +"histogram (plus modif) based")

    # Vyzve uzivatele k zadani jmena souboru.
    info = raw_input("Give me a filename: ")
    if(info == '='):
        fileName = 'morpho.png'
    else:
        fileName = info

    imgLoaded = matplotlib.image.imread(fileName)

    if(number == '1'):
        main(imgLoaded)
    elif(number == '2'):
        mainHist(imgLoaded)
    elif(number == '3'):
        mainHistPlus(imgLoaded)
    elif(number == '4'):
        mainGauss(imgLoaded)
    elif(number == '5'):
        mainHistPlusModif(imgLoaded)

    #raise Exception('Input size error','Shape if inumput data and segmentation must be same')













