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

def main():

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

    ## Vratit 4 nejvetsi objekty - v debug modu.
    obj = 4
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

if __name__ == '__main__':

    main()
