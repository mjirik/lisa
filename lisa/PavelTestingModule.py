# -*- coding: utf-8 -*-
"""
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky
Email:       volkovinsky.pavel@gmail.com

Created:     2013/10/14
Copyright:   (c) Pavel Volkovinsky
"""

import segmentation as sg
import numpy as np

import misc

def main1():

    ## Vytvorim si vlastni testovaci matici.
    matrix = np.zeros((7, 7))
    matrix[0:2, 3:] = 0.7
    matrix[0:4, 0:2] = 0.8
    matrix[3, 2:4] = 1
    matrix[3:5, 4:6] = 1
    matrix[6, 3:5] = -0.7
    matrix[5, 4] = 1
    matrix[6, 6] = 1
    matrix[5:, 0:2] = 0
    matrix[3:5, 4] = 0
    print 'Matrix:'
    print matrix

    matrixX = matrix.copy()
    matrixX[matrixX != 0] = 1
    print 'Matrix binary:'
    print matrixX

    ## Vratit 4 nejvetsi objekty - v debug modu.
    obj = 2
    print '>>> Nejvetsi objekty v matici (' + str(obj) + '):'
    matrix2 = sg.getPriorityObjects(matrix, nObj = obj, seeds = None, debug = True)

    ## Vytvoreni seedu - uhlopricka "/".
    print '>>> Moje seeds:'
    mySeeds = np.zeros((7, 7))
    for index in range(0, 7):
        mySeeds[6 - index, index] = 1
    print 'mySeeds:'
    print mySeeds
    mySeeds = mySeeds.nonzero()
    print 'mySeeds (nonzero):'
    print mySeeds

    ## Uplatneni seedu na celou matici
    print '>>> Seeds na celou matici:'
    matrix3 = sg.getPriorityObjects(matrix, nObj = obj, seeds = mySeeds, debug = True)
    print '>>> Seeds na matici po vraceni nejvetsich objektu:'
    ## Uplatneni seedu na matici po ziskani nejvetsich objektu
    matrix4 = sg.getPriorityObjects(matrix2, nObj = obj, seeds = mySeeds, debug = True)

def main2():

    slab = {'none':0, 'liver':1, 'porta':2}
    voxelsize_mm = np.array([1.0, 1.0, 1.2])

    segm = np.zeros([256, 256, 80], dtype = np.uint8)

    # liver
    segm[70:180, 40:190, 30:60] = slab['liver']
    # port
    segm[120:130, 70:190, 40:45] = slab['porta']
    segm[80:130, 100:110, 40:45] = slab['porta']
    segm[120:170, 130:135, 40:44] = slab['porta']

    data3d = np.zeros(segm.shape)
    data3d[segm == slab['liver']] = 156
    data3d[segm == slab['porta']] = 206
    #noise = (np.random.rand(segm.shape[0], segm.shape[1], segm.shape[2])*30).astype(np.int16)
    noise = (np.random.normal(0, 30, segm.shape))#.astype(np.int16)
    data3d = (data3d + noise).astype(np.uint8)

    # @TODO je tam bug, prohl??e? neum? korektn? pracovat s doubly
    #        app = QApplication(sys.argv)
    #        #pyed = QTSeedEditor(noise )
    #        pyed = QTSeedEditor(data3d)
    #        pyed.exec_()
    #        #img3d = np.zeros([256,256,80], dtype=np.int16)

    # pyed = sed3.sed3(data3d)
    # pyed.show()

    outputTmp = sg.vesselSegmentation(
        data3d,
        segmentation = segm == slab['liver'],
        voxelsize_mm = voxelsize_mm,
        threshold = 180,
        inputSigma = 0.15,
        dilationIterations = 2,
        nObj = 1,
        interactivity = True,
        biggestObjects = False,
        binaryClosingIterations = 5,
        binaryOpeningIterations = 1)

    if outputTmp == None:
       print 'Final output is None'

def main3():

    import matplotlib
    matplotlib.use('TkAgg')

    from numpy import arange, sin, pi
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
    from matplotlib.figure import Figure

    import sys
    if sys.version_info[0] < 3:
        import Tkinter as Tk
    else:
        import tkinter as Tk

    def destroy(e): 
        
        sys.exit()

    root = Tk.Tk()
    root.wm_title("Embedding in TK")
    #root.bind("<Destroy>", destroy)


    f = Figure(figsize=(5,4), dpi=100)
    a = f.add_subplot(111)
    t = arange(0.0,3.0,0.01)
    s = sin(2*pi*t)

    a.plot(t,s)
    a.set_title('Tk embedding')
    a.set_xlabel('X axis label')
    a.set_ylabel('Y label')


    # a tk.DrawingArea
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    #toolbar = NavigationToolbar2TkAgg( canvas, root )
    #toolbar.update()
    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    button = Tk.Button(master=root, text='Quit', command=sys.exit)
    button.pack(side=Tk.BOTTOM)

    Tk.mainloop()

def main4():

    data = misc.obj_from_file('c:\_bp_data\d5\org-liver-orig004.mhd-3mm_alpha45.pklz', filetype = 'pickle')

    outputTmp = sg.vesselSegmentation(
        data['data3d'],
        segmentation = data['segmentation'],
        threshold = -1,
        inputSigma = 0.15,
        dilationIterations = 2,
        interactivity = True,
        biggestObjects = False,
        binaryClosingIterations = 2,
        binaryOpeningIterations = 0)


if __name__ == '__main__':

    main4()
