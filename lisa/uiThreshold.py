# -*- coding: utf-8 -*-
"""
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky
Email:       volkovinsky.pavel@gmail.com

Created:     2012/11/08
Copyright:   (c) Pavel Volkovinsky
"""

import sys

sys.path.append("../src/")
sys.path.append("../extern/")

import logging as logger
import traceback

import numpy
# import scipy.ndimage

import thresholding_functions

import matplotlib
import matplotlib.pyplot as matpyplot
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button  # , RadioButtons

import gc as garbage

from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

class uiThresholdQt(QtGui.QDialog):
    def __init__(self, *pars, **params):
    # def __init__(self,parent=None):
        parent = None


        QtGui.QDialog.__init__(self, parent)
        # super(Window, self).__init__(parent)
        # self.setupUi(self)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

# set the layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # layout.addWidget(self.button)
        self.setLayout(layout)

    # def set_params(self, *pars, **params):
        # import sed3.sed3

        params["figure"] = self.figure 
        self.uit = uiThreshold(*pars, **params)
        self.uit.on_close_fcn = self.callback_close
        self.uit.on_show_fcn = self.exec_
        # ed.show()
        self.output = None

    def run(self):
        # pass
        return self.uit.run()

    def callback_close(self, uit):
        # self.output = uit

        self.close()

    def get_values(self):
        pass

class uiThreshold:

    """

    UI pro prahovani 3D dat.

    """

    def __init__(self, data, voxel, threshold=-1, interactivity=True,
                 number=100.0, inputSigma=-1, nObj=10,  biggestObjects=True,
                 useSeedsOfCompactObjects=True,
                 binaryClosingIterations=2, binaryOpeningIterations=0,
                 seeds=None, cmap=matplotlib.cm.Greys_r, fillHoles=True, 
                 figure=None):
        """

        Inicialitacni metoda.
        Input:
            data - data pro prahovani, se kterymi se pracuje
            voxel - velikost voxelu
            threshold
            interactivity - zapnuti / vypnuti gui
            number - maximalni hodnota slideru pro gauss. filtrovani (max sigma)
            inputSigma - pocatecni hodnota pro gauss. filtr
            nObj - pocet nejvetsich objektu k vraceni
            biggestObjects - oznacuje, zda se maji vracet nejvetsi objekty
            binaryClosingIterations - iterace binary closing
            binaryOpeningIterations - iterace binary opening
            seeds - matice s kliknutim uzivatele- pokud se maji vracet
                specifikce objekty
            cmap - grey

        """

        logger.debug('Spoustim prahovani dat...')
        self.on_close_fcn = None

        self.inputDimension = numpy.ndim(data)
        if(self.inputDimension != 3):

            logger.debug(
                'Vstup nema 3 dimenze! Ma jich ' + str(self.inputDimension)
                + '.')
            self.errorsOccured = True
            return

        else:

            self.errorsOccured = False

        self.on_show_fcn = plt.show
        self.interactivity = interactivity
        self.cmap = cmap
        self.number = number
        self.inputSigma = inputSigma
        self.threshold = threshold
        self.nObj = nObj
        self.biggestObjects = biggestObjects
        self.ICBinaryClosingIterations = binaryClosingIterations
        self.ICBinaryOpeningIterations = binaryOpeningIterations
        self.seeds = seeds
        self.useSeedsOfCompactObjects = useSeedsOfCompactObjects
        self.fillHoles = fillHoles

        if (sys.version_info[0] < 3):

            import copy
            self.data = copy.copy(data)
            self.voxel = copy.copy(voxel)

        else:

            self.data = data.copy()
            self.voxel = voxel.copy()

        # Kalkulace objemove jednotky (voxel) (V = a*b*c)
        voxel1 = self.voxel[0]
        voxel2 = self.voxel[1]
        voxel3 = self.voxel[2]
        self.voxelV = voxel1 * voxel2 * voxel3

        self.numpyAMaxKeepDims = False

        self.arrSeed = None

        if self.interactivity == True:

            if figure is None:

                self.fig = matpyplot.figure()
            else:
                self.fig = figure

            # Maximalni a minimalni pouzita hodnota prahovani v datech (bud v
            # celych datech nebo vybranych seedu)
            self.min0 = numpy.amin(numpy.amin(self.data, axis=0))
            if self.seeds == None:

                self.max0 = numpy.amax(numpy.amax(self.data, axis=0))
                self.max0 = self.max0 + \
                    abs(abs(self.min0) - abs(self.max0)) / 10

            else:

                self.arrSeed = thresholding_functions.getSeeds(
                    data, self.seeds)

                # Pokud existuji vhodne labely, vytvori se nova data k
                # vraceni.
                # Pokud ne, vrati se "None" typ.  { Deprecated: Pokud ne,
                # vrati se cela nafiltrovana data, ktera do funkce prisla
                # (nedojde k vraceni specifickych objektu).  }
                if len(self.arrSeed) > 0:

                    # Zbaveni se duplikatu.
                    self.arrSeed = list(set(self.arrSeed))
                    logger.debug('Hodnoty seedu: ')
                    logger.debug(self.arrSeed)

                    self.max0 = max(self.arrSeed)
                    self.max0 = self.max0 + \
                        abs(abs(self.min0) - abs(self.max0)) / 10

                    # Prahy
                    logger.debug('')
                    logger.debug(
                        'Minimalni doporucena hodnota prahu: ' +
                        str(min(self.arrSeed)))
                    logger.debug(
                        'Maximalni doporucena hodnota prahu: ' +
                        str(max(self.arrSeed)))
                    logger.debug('')

                else:

                    self.max0 = numpy.amax(numpy.amax(self.data, axis=0))
                    self.max0 = self.max0 + \
                        abs(abs(self.min0) - abs(self.max0)) / 10

            # Pridani subplotu do okna (do figure)
            self.ax1 = self.fig.add_subplot(222)
            self.ax2 = self.fig.add_subplot(223)
            self.ax3 = self.fig.add_subplot(224)
            self.ax4 = self.fig.add_subplot(221)

            # Upraveni subplotu
            self.fig.subplots_adjust(left=0.1, bottom=0.3)

            # Vykreslit obrazek
            self.ax1.imshow(
                numpy.amax(self.data, axis=0, keepdims=self.numpyAMaxKeepDims),
                self.cmap)
            self.ax2.imshow(
                numpy.amax(self.data, axis=1, keepdims=self.numpyAMaxKeepDims),
                self.cmap)
            self.ax3.imshow(
                numpy.amax(self.data, axis=2, keepdims=self.numpyAMaxKeepDims),
                self.cmap)

            # Zalozeni mist pro slidery
            self.axcolor = 'white'  # lightgoldenrodyellow
            self.axmin = self.fig.add_axes(
                [0.20, 0.24, 0.55, 0.03], axisbg=self.axcolor)
            self.axmax = self.fig.add_axes(
                [0.20, 0.20, 0.55, 0.03], axisbg=self.axcolor)
            self.axclosing = self.fig.add_axes(
                [0.20, 0.16, 0.55, 0.03], axisbg=self.axcolor)
            self.axopening = self.fig.add_axes(
                [0.20, 0.12, 0.55, 0.03], axisbg=self.axcolor)
            self.axsigma = self.fig.add_axes(
                [0.20, 0.08, 0.55, 0.03], axisbg=self.axcolor)

            # Vlastni vytvoreni slideru

            minBinaryClosing = 0
            minBinaryOpening = 0
            minSigma = 0.00

            self.firstRun = True

            thres = self.threshold
            if thres == -1:
                try:
                    thres = thresholding_functions.calculateAutomaticThreshold(
                        self.data, self.arrSeed)
                except:
                    logger.info(traceback.format_exc())
                    thres = (self.max0 + self.min0) / 2

            self.smin = Slider(
                self.axmin, 'Minimal threshold   ' + str(self.min0),
                self.min0, self.max0, valinit=thres, dragging=True)
            self.smax = Slider(
                self.axmax, 'Maximal threshold   ' + str(self.min0),
                self.min0, self.max0, valinit=self.max0, dragging=True)

            if(self.ICBinaryClosingIterations >= 1):
                self.sclose = Slider(
                    self.axclosing, 'Binary closing', minBinaryClosing,
                    100, valinit=self.ICBinaryClosingIterations, dragging=False)
            else:
                self.sclose = Slider(
                    self.axclosing, 'Binary closing', minBinaryClosing, 100,
                    valinit=0, dragging=False)

            if(self.ICBinaryOpeningIterations >= 1):
                self.sopen = Slider(
                    self.axopening, 'Binary opening', minBinaryOpening,
                    100, valinit=self.ICBinaryOpeningIterations, dragging=False)
            else:
                self.sopen = Slider(
                    self.axopening, 'Binary opening', minBinaryOpening, 100,
                    valinit=0, dragging=False)

            self.ssigma = Slider(
                self.axsigma, 'Sigma', 0.00, self.number,
                valinit=self.inputSigma, dragging=False)

            # Funkce slideru pri zmene jeho hodnoty
            self.smin.on_changed(self.updateImage)
            self.smax.on_changed(self.updateImage)
            self.sclose.on_changed(self.updateImage)
            self.sopen.on_changed(self.updateImage)
            self.ssigma.on_changed(self.updateImage)

            # Zalozeni mist pro tlacitka
            self.axbuttnext1 = self.fig.add_axes(
                [0.86, 0.24, 0.04, 0.03], axisbg=self.axcolor)
            self.axbuttprev1 = self.fig.add_axes(
                [0.81, 0.24, 0.04, 0.03], axisbg=self.axcolor)
            self.axbuttnext2 = self.fig.add_axes(
                [0.86, 0.20, 0.04, 0.03], axisbg=self.axcolor)
            self.axbuttprev2 = self.fig.add_axes(
                [0.81, 0.20, 0.04, 0.03], axisbg=self.axcolor)
            self.axbuttnextclosing = self.fig.add_axes(
                [0.86, 0.16, 0.04, 0.03], axisbg=self.axcolor)
            self.axbuttprevclosing = self.fig.add_axes(
                [0.81, 0.16, 0.04, 0.03], axisbg=self.axcolor)
            self.axbuttnextopening = self.fig.add_axes(
                [0.86, 0.12, 0.04, 0.03], axisbg=self.axcolor)
            self.axbuttprevopening = self.fig.add_axes(
                [0.81, 0.12, 0.04, 0.03], axisbg=self.axcolor)
            self.axbuttreset = self.fig.add_axes(
                [0.79, 0.08, 0.06, 0.03], axisbg=self.axcolor)
            self.axbuttcontinue = self.fig.add_axes(
                [0.86, 0.08, 0.06, 0.03], axisbg=self.axcolor)

            # Zalozeni tlacitek
            self.bnext1 = Button(self.axbuttnext1, '+1.0')
            self.bprev1 = Button(self.axbuttprev1, '-1.0')
            self.bnext2 = Button(self.axbuttnext2, '+1.0')
            self.bprev2 = Button(self.axbuttprev2, '-1.0')
            self.bnextclosing = Button(self.axbuttnextclosing, '+1.0')
            self.bprevclosing = Button(self.axbuttprevclosing, '-1.0')
            self.bnextopening = Button(self.axbuttnextopening, '+1.0')
            self.bprevopening = Button(self.axbuttprevopening, '-1.0')
            self.breset = Button(self.axbuttreset, 'Reset')
            self.bcontinue = Button(self.axbuttcontinue, 'Next UI')

            # Funkce tlacitek pri jejich aktivaci
            self.bnext1.on_clicked(self.buttonMinNext)
            self.bprev1.on_clicked(self.buttonMinPrev)
            self.bnext2.on_clicked(self.buttonMaxNext)
            self.bprev2.on_clicked(self.buttonMaxPrev)
            self.bnextclosing.on_clicked(self.buttonNextClosing)
            self.bprevclosing.on_clicked(self.buttonPrevClosing)
            self.bnextopening.on_clicked(self.buttonNextOpening)
            self.bprevopening.on_clicked(self.buttonPrevOpening)
            self.breset.on_clicked(self.buttonReset)
            self.bcontinue.on_clicked(self.buttonContinue)

            self.smin.valtext.set_text('{}'.format(self.smin.val))
            self.smax.valtext.set_text('{}'.format(self.smax.val))

    def run(self):
        """

            Spusteni UI.

        """

        if(self.errorsOccured == True):

            return self.data

        self.firstRun = True

        if self.interactivity == False:

            self.updateImage(-1)
            garbage.collect()

        else:

            self.updateImage(-1)
            garbage.collect()
            self.on_show_fcn()

        del(self.data)

        garbage.collect()

        return self.imgFiltering

    def returnLastThreshold(self):

        return self.threshold

    def updateImage(self, val):
        """

        Hlavni update metoda.
        Cinny kod pro gaussovske filtrovani, prahovani, binarni uzavreni a
        otevreni a vraceni nejvetsich nebo oznacenych objektu.

        """

        if (sys.version_info[0] < 3):

            import copy
            self.imgFiltering = copy.copy(self.data)

        else:

            self.imgFiltering = self.data.copy()

        # Filtrovani

        # Zjisteni jakou sigmu pouzit
        if(self.firstRun == True and self.inputSigma >= 0):
            sigma = numpy.round(self.inputSigma, 2)
        else:
            sigma = numpy.round(self.ssigma.val, 2)
        sigmaNew = thresholding_functions.calculateSigma(self.voxel, sigma)

        self.imgFiltering = thresholding_functions.gaussFilter(
            self.imgFiltering, sigmaNew)

        del(sigmaNew)

        # Prahovani (smin, smax)

        max_threshold = -1
        min_threshold = self.threshold

        if self.interactivity:

            self.smin.val = (numpy.round(self.smin.val, 2))
            self.smin.valtext.set_text('{}'.format(self.smin.val))
            self.smax.val = (numpy.round(self.smax.val, 2))
            self.smax.valtext.set_text('{}'.format(self.smax.val))

            min_threshold = self.smin.val
            max_threshold = self.smax.val

            self.threshold = min_threshold

        if (self.threshold == -1) and self.firstRun:

            min_threshold = thresholding_functions.calculateAutomaticThreshold(
                self.imgFiltering, self.arrSeed)

        self.imgFiltering = thresholding_functions.thresholding(
            self.imgFiltering, min_threshold, max_threshold, True,
            self.interactivity)

        # Operace binarni otevreni a uzavreni.

        # Nastaveni hodnot slideru.
        if (self.interactivity == True):

            closeNum = int(numpy.round(self.sclose.val, 0))
            openNum = int(numpy.round(self.sopen.val, 0))
            self.sclose.valtext.set_text('{}'.format(closeNum))
            self.sopen.valtext.set_text('{}'.format(openNum))

        else:

            closeNum = self.ICBinaryClosingIterations
            openNum = self.ICBinaryOpeningIterations

        self.imgFiltering = thresholding_functions.binaryClosingOpening(
            self.imgFiltering, closeNum, openNum, True)

# Fill holes
        if self.fillHoles:

            self.imgFiltering = thresholding_functions.fillHoles(
                self.imgFiltering)

        # Zjisteni nejvetsich objektu.
        self.getBiggestObjects()

        # Vykresleni dat
        if (self.interactivity == True):
            self.drawVisualization()

        # Nastaveni kontrolnich hodnot
        self.firstRun = False

        garbage.collect()

        self.debugInfo()

    def debugInfo(self):

        logger.debug('======')
        logger.debug('!Debug')
        logger.debug('\tUpdate cycle:')
        logger.debug('\t\tThreshold min: ' +
                     str(numpy.round(self.threshold, 2)))
        if (self.interactivity == True):
            logger.debug(
                '\t\tThreshold max: ' + str(numpy.round(self.smax.val, 2)))
            logger.debug(
                '\t\tBinary closing: ' + str(numpy.round(self.sclose.val, 0)))
            logger.debug(
                '\t\tBinary opening: ' + str(numpy.round(self.sopen.val, 0)))
            logger.debug(
                '\t\tSigma filter param: ' + str(numpy.round(self.ssigma.val,
                                                             2)))
        logger.debug('======')

    def getBiggestObjects(self):
        """

        Vraceni nejvetsich objektu (nebo objektu, ktere obsahuji prioritni
        seedy).

        """

        logger.debug('biggest objects ' + str(self.biggestObjects))
        logger.debug('self.seeds ' + str(self.seeds))
        if (self.biggestObjects == True or
                (self.seeds != None and self.useSeedsOfCompactObjects)):

            self.imgFiltering = thresholding_functions.getPriorityObjects(
                self.imgFiltering, self.nObj, self.seeds)

    def __drawSegmentedSlice(self, ax, contour, i):
        ax.cla()
        ax.imshow(self.data[i, :, :], cmap=self.cmap)
        if contour is not None:
            ax.contour(contour[i, :, :])
            # import sed3
            # sed3.sed3(contour, show=True)

    def drawVisualization(self):
        """

        Vykresleni dat.

        """

        # Predani dat k vykresleni
        if (self.imgFiltering == None):

            # print '(DEBUG) Typ dat: ' + str(type(self.data[0][0][0]))

            self.ax1.imshow(
                numpy.amax(numpy.zeros(self.data.shape), axis=0,
                           keepdims=self.numpyAMaxKeepDims), self.cmap)
            self.ax2.imshow(
                numpy.amax(numpy.zeros(self.data.shape), axis=1,
                           keepdims=self.numpyAMaxKeepDims), self.cmap)
            self.ax3.imshow(
                numpy.amax(numpy.zeros(self.data.shape), axis=2,
                           keepdims=self.numpyAMaxKeepDims), self.cmap)

        else:

            img0, img1, img2 = thresholding_functions.prepareVisualization(
                self.imgFiltering)

            # t2 = time.time()
            # print 't1 %f t2 %f ' % (t1 - t0, t2 - t1)

            self.ax1.imshow(img0, self.cmap)
            self.ax2.imshow(img1, self.cmap)
            self.ax3.imshow(img2, self.cmap)

            # del(img0)
            # del(img1)
            # del(img2)

        self.__drawSegmentedSlice(self.ax4, self.imgFiltering, self.imgFiltering.shape[0]/2)
        # Prekresleni
        self.fig.canvas.draw()

    def buttonReset(self, event):

        self.sclose.valtext.set_text(
            '{}'.format(self.ICBinaryClosingIterations))
        self.sopen.valtext.set_text(
            '{}'.format(self.ICBinaryOpeningIterations))
        self.ssigma.valtext.set_text('{}'.format(self.inputSigma))

        self.firstRun = True
        self.lastSigma = -1
        self.threshold = -1

        self.updateImage(0)

    def buttonContinue(self, event):

        matpyplot.clf()
        if self.on_close_fcn is not None:
            self.on_close_fcn(self)
        matpyplot.close()

    def buttonMinNext(self, event):

        if self.max0 < (self.smin.val + 1.0):

            self.smin.val = self.max0

        else:

            self.smin.val += 1.0

        self.smin.val = (numpy.round(self.smin.val, 2))
        self.smin.valtext.set_text('{}'.format(self.smin.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonMinPrev(self, event):

        if self.min0 > (self.smin.val - 1.0):

            self.smin.val = self.min0

        else:

            self.smin.val -= 1.0

        self.smin.val = (numpy.round(self.smin.val, 2))
        self.smin.valtext.set_text('{}'.format(self.smin.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonMaxNext(self, event):

        if self.max0 < (self.smax.val + 1.0):

            self.smax.val = self.max0

        else:

            self.smax.val += 1.0

        self.smax.val = (numpy.round(self.smax.val, 2))
        self.smax.valtext.set_text('{}'.format(self.smax.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonMaxPrev(self, event):

        if self.min0 > (self.smax.val - 1.0):

            self.smax.val = self.min0

        else:

            self.smax.val -= 1.0

        self.smax.val = (numpy.round(self.smax.val, 2))
        self.smax.valtext.set_text('{}'.format(self.smax.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonNextOpening(self, event):

        self.sopen.val += 1.0
        self.sopen.val = (numpy.round(self.sopen.val, 2))
        self.sopen.valtext.set_text('{}'.format(self.sopen.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonPrevOpening(self, event):

        if(self.sopen.val >= 1.0):
            self.sopen.val -= 1.0
            self.sopen.val = (numpy.round(self.sopen.val, 2))
            self.sopen.valtext.set_text('{}'.format(self.sopen.val))
            self.fig.canvas.draw()
            self.updateImage(0)

    def buttonNextClosing(self, event):

        self.sclose.val += 1.0
        self.sclose.val = (numpy.round(self.sclose.val, 2))
        self.sclose.valtext.set_text('{}'.format(self.sclose.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonPrevClosing(self, event):

        if(self.sclose.val >= 1.0):
            self.sclose.val -= 1.0
            self.sclose.val = (numpy.round(self.sclose.val, 2))
            self.sclose.valtext.set_text('{}'.format(self.sclose.val))
            self.fig.canvas.draw()
            self.updateImage(0)
