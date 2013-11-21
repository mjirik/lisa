#-------------------------------------------------------------------------------
# Name:        PavelTestingModule
# Purpose:     ZCU - FAV
#
# Author:      Pavel Volkovinsky
# Email:       volkovinsky.pavel@gmail.com
#
# Created:     14/10/2013
# Copyright:   (c) Pavel Volkovinsky 2013
#-------------------------------------------------------------------------------

import segmentation as sg
import numpy as np

def main1():

    ## Vytvorim si vlastni testovaci matici.
    matrix = np.zeros((7, 7))
    matrix[0:2, 3:] = 1
    matrix[0:4, 0:2] = 1
    matrix[3, 2:4] = 1
    matrix[3:5, 4:6] = 1
    matrix[6, 3:5] = 1
    matrix[5, 4] = 1
    matrix[6, 6] = 1
    matrix[5:, 0:2] = 1
    matrix[3:5, 4] = 0
    print 'Matrix:'
    print matrix

    ##matrixX = matrix / matrix
    ##print 'Matrix binary:'
    ##print matrixX
    ##matrixX[np.isnan(matrixX)] = 0
    ##print 'Matrix binary:'
    ##print matrixX

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

    segm = np.zeros([256, 256, 80], dtype = np.int16)

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
    data3d = (data3d + noise).astype(np.int16)

    # @TODO je tam bug, prohl??e? neum? korektn? pracovat s doubly
    #        app = QApplication(sys.argv)
    #        #pyed = QTSeedEditor(noise )
    #        pyed = QTSeedEditor(data3d)
    #        pyed.exec_()
    #        #img3d = np.zeros([256,256,80], dtype=np.int16)

    # pyed = py3DSeedEditor.py3DSeedEditor(data3d)
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
        biggestObjects = True,
        binaryClosingIterations = 5,
        binaryOpeningIterations = 1)

    if outputTmp == None:
       print 'Final output is None'

if __name__ == '__main__':

    main2()
