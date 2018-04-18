#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy as np

import logging
logger = logging.getLogger(__name__)
import argparse

from scipy import ndimage
from . import qmisc

class ShapeModel():
    """
    Cílem je dát dohromady vstupní data s různou velikostí a různou polohou
    objektu. Výstup je pak zapotřebí opět přizpůsobit libovolné velikosti a
    poloze objektu v obraze.

    Model je tvořen polem s velikostí definovanou v konstruktoru (self.shape).
    U modelu je potřeba brát v potaz polohu objektu. Ta je udávána pomocí
    crinfo. To je skupina polí s minimální a maximální hodnotou pro každou osu.

    Trénování je prováděno opakovaným voláním funkce train_one().

    :param model_margin: stanovuje velikost okraje v modelu. Objekt bude ve
    výchozím nastavení vzdálen 0 px od každého okraje.

    """

    def __init__(self, shape=[5, 5, 5]):
        """TODO: to be defined1. """
        self.model = np.ones(shape)
        self.data_number = 0
        self.model_margin = [0, 0, 0]
        pass

    def get_model(self, crinfo, image_shape):
        """
        :param image_shape: Size of output image
        :param crinfo: Array with min and max index of object for each axis.
        [[minx, maxx], [miny, maxy], [minz, maxz]]

        """
        # Průměrování
        mdl = self.model / self.data_number
        print(mdl.shape)
        print(crinfo)
        # mdl_res = io3d.misc.resize_to_shape(mdl, crinfo[0][]
        uncr = qmisc.uncrop(mdl, crinfo, image_shape, resize=True)
        return uncr

    def train_one(self, data,voxelSize_mm):
        """
        Trenovani shape modelu
        data se vezmou a oriznou (jen jatra)
        na oriznuta data je aplikovo binarni otevreni - rychlejsi nez morphsnakes
        co vznikne je uhlazena cast ktera se odecte od puvodniho obrazu
        cimz vzniknou spicky
        orezany obraz se nasledne rozparceluje podle velikosti (shape) modelu
        pokud pocet voxelu v danem useku prekroci danou mez, je modelu 
        prirazena nejaka hodnota. Meze jsou nasledujici:
        0%-50% => 1
        50%-75% => 2
        75%-100% => 3
        """
        
        crinfo = qmisc.crinfo_from_specific_data(data, margin=self.model_margin)
        datacr = qmisc.crop(data, crinfo=crinfo)
        dataShape = self.model.shape
        datacrres = self.trainThresholdMap(datacr, voxelSize_mm, dataShape)

        self.model += datacrres

        self.data_number += 1

# Tady bude super kód pro trénování

    def train(self, data_arr):
        for data in data_arr:
            self.train_one(data)
            
    def objectThreshold(self,objekt,thresholds,values):
        '''
        Objekt - 3d T/F pole 
        thresholds = [0,0.5,0.75] zacina nulou
        values = [3,2,1]
        vrati hodnotu z values odpovidajici thresholds
        podle podilu True voxelu obsazenych v 3d poli 
        zde napriklad 60% =>2, 80% => 1.
        '''
        bile = np.sum(objekt)
        velikost = objekt.shape
        velikostCelkem = 1.0
        for x in velikost:
            velikostCelkem = velikostCelkem*x
            
        podil = bile/velikostCelkem #podil True voxelu    
        #print podil    
        #vybrani threshold
        final = 0 #vracena hodnota
        pomocny = 0 #pomocna promenna
        for threshold in thresholds:
            if(podil >= threshold ):
                final = values[pomocny]
            pomocny = pomocny+1
        return final
        
    
    def rozdelData(self,crData,dataShape, nasobitel1=1,nasobitel2 = 2):
        '''
        crData - vstupni data
        dataShape - velikost vraceneho pole
        volte 0<nasobitel1 < nasobitel2, vysvetleni nasleduje:
        rozdeli pole crData na casti vrati pole rozmeru dataShape
        vysledne hodnoty pole jsou urceny funkci objectThreshold(object,thresholds,values)
        intervaly prirazeni values [1-3] jsou nasledujici: 
        [0-prumer*nasobitel1],[prumer*nasobitel1-prumer*nasobitel2],[prumer*nasobitel2 a vice]
        '''
            
        'vypocet prumerneho podilu bilych voxelu'
        bile = np.sum(crData)
        velikost = crData.shape
        velikostCelkem = 1.0
        for x in velikost:
            velikostCelkem = velikostCelkem*x
            
        podil = bile/velikostCelkem #prumerny podil True voxelu
        
        thresholds = [0,nasobitel1*podil,nasobitel2*podil]
        values = [3,2,1]  
        
        'vybrani voxelu a vytvoreni objektu'    
        velikostDat = crData.shape
        voxelySmer = [0,0,0]
        vysledek = np.zeros(dataShape)
        for poradi in range(3):
            voxelySmer[poradi] = velikostDat[poradi]/dataShape[poradi]   
        
        for x in range(dataShape[0]):
            for y in range(dataShape[1]):
                for z in range(dataShape[2]):
                    xStart = x * voxelySmer[0]
                    xKonec = xStart + voxelySmer[0]
                    
                    yStart = y * voxelySmer[1]
                    yKonec = yStart + voxelySmer[1]
                    
                    zStart = z * voxelySmer[2]
                    zKonec = zStart + voxelySmer[2]
                    objekt = crData[
                             int(xStart):int(xKonec),
                             int(yStart):int(yKonec),
                             int(zStart):int(zKonec)
                             ]
                    vysledek[x,y,z] = self.objectThreshold(objekt,thresholds,values)
                    
                    
        return vysledek
    
    def vytvorKouli3D(self,voxelSize_mm,polomer_mm):
        '''voxelSize:mm = [x,y,z], polomer_mm = r
        Vytvari kouli v 3d prostoru postupnym vytvarenim
        kruznic podel X (prvni) osy. Predpokladem spravnosti
        funkce je ze Y a Z osy maji stejne rozliseni
        funkce vyuziva pythagorovu vetu'''
    
        print('zahajeno vytvareni 3D objektu')
    
        x = voxelSize_mm[0]
        y = voxelSize_mm[1]
        z = voxelSize_mm[2]
        xVoxely = int(np.ceil(polomer_mm/x))
        yVoxely = int(np.ceil(polomer_mm/y))
        zVoxely = int( np.ceil(polomer_mm/z))
    
        rozmery = [xVoxely*2+1,yVoxely*2+1,yVoxely*2+1]
        xStred  = xVoxely
        konec = yVoxely*2+1
        koule = np.zeros(rozmery) #pole kam bude ulozen vysledek
    
        for xR in range(xVoxely*2+1):
    
            if(xR == xStred):
                print('3D objekt z 50% vytvoren')
    
            c = polomer_mm #nejdelsi strana
            a = (xStred-xR )*x
            vnitrek = (c**2-a**2)
            b = 0.0
            if(vnitrek > 0):
                b = np.sqrt((c**2-a**2))#pythagorova veta   b je v mm
            rKruznice = float(b)/float(y)
            if(rKruznice == np.NAN):
                continue
            #print rKruznice #osetreni NAN
            kruznice = self.vytvoritTFKruznici(yVoxely,rKruznice)
            koule[xR,0:konec,0:konec] = kruznice[0:konec,0:konec]
    
        print('3D objekt uspesne vytvoren')
        return koule

    def vytvoritTFKruznici(self,polomerPole,polomerKruznice):
        '''vytvori 2d pole velikosti 2xpolomerPole+1
         s kruznici o polomeru polomerKruznice uprostred '''
        radius = polomerPole
        r2 = np.arange(-radius, radius+1)**2
        dist2 = r2[:, None] + r2
        vratit =  (dist2 <= polomerKruznice**2).astype(np.int)
        return vratit
    
    def trainThresholdMap(self,data3d,voxelSize,dataShape):
        structure = self.vytvorKouli3D(voxelSize, 5)
        smoothed = ndimage.binary_opening(data3d, structure, 3)    
        spicky = smoothed != data3d
        vysledek = self.rozdelData(spicky,dataShape)
        return vysledek

def main():
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    # fh = logging.FileHandler('log.txt')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.debug('start')

    # input parser
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        required=True,
        help='input file'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)


if __name__ == "__main__":
    main()
