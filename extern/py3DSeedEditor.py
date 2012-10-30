#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import unittest
import sys
sys.path.append("./src/")
import pdb
#  pdb.set_trace();

import scipy.io

import logging
logger = logging.getLogger(__name__)


import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider, Button, RadioButtons


#Ahooooooj


class py3DSeedEditor:
    """ Viewer and seed editor for 2D and 3D data. 

    py3DSeedEditor(img, ...)

    img: 2D or 3D grayscale data
    voxelsizemm: size of voxel, default is [1, 1, 1]
    initslice: 0
    colorbar: True/False, default is True
    cmap: colormap


    ed = py3DSeedEditor(img)
    ed.show()
    selected_seeds = ed.seeds

    """
    #if data.shape != segmentation.shape:
    #    raise Exception('Input size error','Shape if input data and segmentation must be same')

    def __init__(self, img, voxelsizemm=[1,1,1], initslice = 0 , colorbar = True,
            cmap = matplotlib.cm.Greys_r):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        if len(img.shape) == 2:
            imgtmp = img
            img = np.zeros([imgtmp.shape[0], imgtmp.shape[1], 1])
            #self.imgshape.append(1)
            img[:,:,-1] = imgtmp
            #pdb.set_trace();
        self.imgshape = list(img.shape)
        self.img = img
        self.actual_slice = initslice
        self.colorbar = colorbar
        self.cmap = cmap 
        self.seeds = np.zeros(self.imgshape, np.int8)
        self.imgmax = np.max(img)
        self.imgmin = np.min(img)


        self.press = None
        self.press2 = None

        self.fig.subplots_adjust(left=0.25, bottom=0.25)


        self.show_slice()

        if self.colorbar:
            self.fig.colorbar(self.imsh)

        # user interface look

        axcolor = 'lightgoldenrodyellow'
        ax_actual_slice = self.fig.add_axes([0.2, 0.2, 0.5, 0.03], axisbg=axcolor)
        self.actual_slice_slider = Slider(ax_actual_slice, 'Slice', 0, 
                self.imgshape[2], valinit=initslice)
        
        # conenction to wheel events
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.actual_slice_slider.on_changed(self.sliceslider_update)
# draw
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.show_slice()


    def show_slice(self):
        sliceimg = self.img[:,:,self.actual_slice]
        self.imsh = self.ax.imshow(sliceimg, self.cmap, vmin = self.imgmin, vmax = self.imgmax)
        #plt.hold(True)
        #pdb.set_trace();
        self.ax.imshow(self.prepare_overlay(self.seeds[:,:,self.actual_slice]))
        self.fig.canvas.draw()
        #pdb.set_trace();
        #plt.hold(False)

    def next_slice(self):
        self.actual_slice = self.actual_slice + 1
        if self.actual_slice >= self.imgshape[2]:
            self.actual_slice = 0

    def prev_slice(self):
        self.actual_slice = self.actual_slice - 1
        if self.actual_slice < 0:
            self.actual_slice = self.imgshape[2] - 1

    def sliceslider_update(self, val):
# zaokrouhlení
        #self.actual_slice_slider.set_val(round(self.actual_slice_slider.val))
        self.actual_slice = round(val)
        self.show_slice()

    def prepare_overlay(self,seeds):
        sh = list(seeds.shape)
        if len(sh) == 2:
            sh.append(4)
        else:
            sh[2] = 4
        # assert sh[2] == 1, 'wrong overlay shape'
        # sh[2] = 4
        overlay = np.zeros(sh)

        overlay[:,:,0] = (seeds == 1)
        overlay[:,:,1] = (seeds == 2)
        overlay[:,:,2] = (seeds == 3)

        overlay[:,:,3] = (seeds > 0)

        return overlay



    def show(self):
        """ Function run viewer window.
        """
        plt.show()
        return self.seeds

    def on_scroll(self, event):
        ''' mouse wheel is used for setting slider value'''
        if event.button == 'up':
            self.next_slice()
        if event.button == 'down':
            self.prev_slice()
        self.actual_slice_slider.set_val (self.actual_slice)
        #tim, ze dojde ke zmene slideru je show_slce volan z nej
        #self.show_slice()
        #print self.actual_slice


## malování -------------------
    def on_press(self, event):
        'on but-ton press we will see if the mouse is over us and store some data'
        if event.inaxes != self.ax: return
        #contains, attrd = self.rect.contains(event)
        #if not contains: return
        #print 'event contains', self.rect.xy
        #x0, y0 = self.rect.xy
        self.press = [event.xdata], [event.ydata], event.button
        #self.press1 = True
    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return

        if event.inaxes != self.ax: return
        #print event.inaxes

        x0, y0, btn = self.press
        x0.append(event.xdata)
        y0.append(event.ydata)

    def on_release(self, event):
        'on release we reset the press data'
        if self.press is None: return
        #print self.press
        x0, y0, btn = self.press
        if btn == 1:
            color = 'r'
        elif btn == 2:
            color = 'b'

        #plt.axes(self.ax)
        #plt.plot(x0, y0)
        self.set_seeds(y0, x0, self.actual_slice, btn )
        #self.fig.canvas.draw()
        #pdb.set_trace();
        self.press = None
        self.show_slice()

    def set_seeds(self, px, py, pz, value = 1, voxelsizemm = [1,1,1], cursorsizemm = [1,1,1]):
        assert len(px) == len(py) , 'px and py describes a point, their size must be same'

        for i, item in enumerate(px):
            self.seeds[item, py[i], pz] = value


#self.rect.figure.canvas.draw()

    #return data 

# --------------------------tests-----------------------------
class Tests(unittest.TestCase):
    def test_t(self):
        pass
    def setUp(self):
        """ Nastavení společných proměnných pro testy  """
        datashape = [120,85,30]
        self.datashape = datashape
        self.rnddata = np.random.rand(datashape[0], datashape[1], datashape[2])
        self.segmcube = np.zeros(datashape)
        self.segmcube[30:70, 40:60,5:15] = 1

        self.ed = py3DSeedEditor(self.rnddata)
        #ed.show()
        #selected_seeds = ed.seeds

    def test_same_size_input_and_output(self):
        """Funkce testuje stejnost vstupních a výstupních dat"""
        #outputdata = vesselSegmentation(self.rnddata,self.segmcube)
        self.assertEqual(self.ed.seeds.shape, self.rnddata.shape)
    def test_set_seeds(self):
        ''' Testuje uložení do seedů '''
        val = 7
        self.ed.set_seeds([10,12,13],[13,13,15], 3, value=val)
        self.assertEqual(self.ed.seeds[10,13,3],val)

    def test_prepare_overlay(self):
        ''' Testuje vytvoření rgba obrázku z labelů'''
        overlay = self.ed.prepare_overlay(self.segmcube[:,:,6])
        onePixel = overlay[30,40]
        self.assertTrue(all(onePixel == [1,0,0,1]))



#
#    def test_different_data_and_segmentation_size(self):
#        """ Funkce ověřuje vyhození výjimky při různém velikosti vstpních
#        dat a segmentace """
#        pdb.set_trace();
#        self.assertRaises(Exception, vesselSegmentation, (self.rnddata, self.segmcube[2:,:,:]) )
#
        
        
# --------------------------main------------------------------
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
# při vývoji si necháme vypisovat všechny hlášky
    #logger.setLevel(logging.DEBUG)

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
    parser.add_argument('-f','--filename',  
            #default = '../jatra/main/step.mat',
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
        # hack for use argparse and unittest in one module
        sys.argv[1:]=[]
        unittest.main()

    if args.filename == 'lena':
        from scipy import misc
        data = misc.lena()
    else:
    #   load all 
        mat = scipy.io.loadmat(args.filename)
        logger.debug( mat.keys())

        # load specific variable
        dataraw = scipy.io.loadmat(args.filename, variable_names=['data'])
        data = dataraw['data']

        #logger.debug(matthreshold['threshold'][0][0])


        # zastavení chodu programu pro potřeby debugu, 
        # ovládá se klávesou's','c',... 
        # zakomentovat
        #pdb.set_trace();

        # zde by byl prostor pro ruční (interaktivní) zvolení prahu z klávesnice 
        #tě ebo jinak

    pyed = py3DSeedEditor(data)
    output = pyed.show()

    scipy.io.savemat(args.outputfile,{'data':output})

