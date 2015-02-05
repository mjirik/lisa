#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2014 mjirik <mjirik@mjirik-HP-Compaq-Elite-8300-MT>
#
# Distributed under terms of the MIT license.

"""
Modul slouží k segmentaci jater.
Třída musí obsahovat funkce run(), interactivity_loop() a proměnné seeds,
voxelsize, segmentation a interactivity_counter.
Spoluautor: Martin Červený

"""

import logging
logger = logging.getLogger(__name__)
import argparse
import numpy as np
import io3d
import os
import pickle



def vyhledejSoubory(cesta):
    ''' vrátí pole názvů všech souborů končících  .mhd v daném adresáři
    předpoklad je že jsou seřazeny nejprve originály, pak trénovací
    kousky. Pokud s tímto máte problémy pojmenujte je následovně:
    liver-orig001.mhd atd... liver-seg001.mhd atd a seřaďte abecedně'''
    
    konec = '.mhd'
    novy = []
    seznam = os.listdir(cesta)
    for polozka in seznam:
        if (polozka.endswith(konec)):
            novy.append(polozka)
    return novy
    
def nactiSoubor(cesta,seznamSouboru,polozka,reader):
    ''' rozebere nacteny soubor na jednotlive promenne jako je
    velikost voxelu apod. ze slovniku do jednoho pole ,tabulka 
    je použitelná v sed3 editoru, první dimenze = Z (hlava-nohy)'''
    cesta =  cesta+"/" +seznamSouboru[polozka] 
    datap = reader.Get3DData(cesta, dataplus_format=False)
    tabulka = datap[0]
    slovnik = datap[1]
    velikostVoxelu = slovnik['voxelsize_mm']
    vektor = [tabulka,velikostVoxelu]
    '''
    ed = sed3.sed3(tabulka)
    ed.show()
    '''
    return vektor

def metoda1(cesta,seznamSouboru):
    '''METODA 1 - PRIMITIVNI
    predpoklady: sudy pocet trenovacich dat, 
    originalni data jsou prvni polovina, pak 
    segmentovana. Kde je segmentace True je 0.
    vypocte prumer a varianci ze segmentovanych voxelu-
    vysledou hodnotu je pak mozno pouzit pro prahovani
    hodnota zapsana do souboru "Metoda1.p" '''
    
    def vypoctiPrumer(pocetVzorku,cislo,prumer):
        '''funkce na vypocet rekurzivniho prumeru'''
        prumerNovy = ((pocetVzorku-1)/float(pocetVzorku)*prumer) + (cislo/float(pocetVzorku))
        return prumerNovy

    def vypoctiVarianci(pocetVzorku,cislo,var,prumer):
        '''funkce na vypocet rekurzivni variance VYZADUJE I VYPOCET PRUMERU'''
        if(pocetVzorku ==1):
            return 0
        varNova = ((pocetVzorku-1)/float(pocetVzorku)*var) + (1/float(pocetVzorku-1))*((cislo-prumer)**2)
        #print (1/float(pocetVzorku-1))*((cislo-prumer)**2)
        return varNova
    
    def zapisPrumVar(prumer,variance):
        '''zapise pole [prumer,variance] pomoci pickle do souboru'''
        radek = [prumer,variance]
        soubor = open('Metoda1.p','wb')
        pickle.dump(radek,soubor)
        soubor.close()
    
    def nactiPrumVar(prumer,variance):
        '''vrati pole [prumer,variance] nactene pomoci pickle ze souboru'''
        soubor = open('Metoda1.p','rb')
        [prumer,variance]=pickle.load(soubor)
        return [prumer,variance]
    
    print "zahajeno trenovani metodou c.1"
    pocetSouboru = len(seznamSouboru)
    pocetOrig = pocetSouboru/2    
    ctenar =  io3d.DataReader()
    
    rekPrumer = 0
    rekOdchylka = 0
    
    pomocny = 0
    for soubor in seznamSouboru:
        ukazatel = str(pomocny+1) + "/" + str(pocetOrig)
        print ukazatel
        originalni = nactiSoubor(cesta,seznamSouboru,pomocny,ctenar) #originalni pole
        segmentovany = nactiSoubor(cesta,seznamSouboru,pomocny+pocetOrig,ctenar) #segmentovane pole(0)

        #print originalni[1]
        #print segmentovany[1]
        pole1 = np.asarray(originalni[1])
        pole2 = np.asarray(segmentovany[1])
        if (not(np.linalg.norm(pole1-pole2) <= 10**(-5))):
            raise NameError('Chyba ve vstupnich datech original c.' + str(pomocny+1) + 'se neshoduje se segmentaci')
        '''ZDE PRACOVAT S originalni A segmentovany'''
       
        poleSeg = segmentovany[0]
        poleOri = originalni[0]
        zeli1 = 0 #radek
        zeli2 = 0 #sloupec
        zeli3 = 0 #rez
        pocetVzorku = 0
        prumer = 0
        var = 0
        
        for rez in poleSeg:
            print ukazatel +  " " + str(float(float(zeli3)/len(poleSeg)))
            for radek in rez:
                for cislo in radek:
                    if(cislo ==0):
                        pocetVzorku = pocetVzorku+1
                        #souradnice = [zeli1,zeli2,zeli3]
                        #print souradnice
                        rezOrig = poleOri[zeli3]
                        #print len(rezOrig)
                        radekOrig = rezOrig[zeli2]
                        #print len(radekOrig)
                        cislo = radekOrig[zeli1]


                        prumer = vypoctiPrumer(pocetVzorku,cislo,prumer)
                        var = vypoctiVarianci(pocetVzorku,cislo,var,prumer)
                    zeli1 = zeli1+1
                zeli1 = 0 #novy radek
                zeli2 = zeli2+1
            zeli1 = 0
            zeli2=0 #novy rez
            zeli3= zeli3+1
            print "prumer: "+ str(prumer) + " variance: " + str(var)
        #print "prumer: "+ str(prumer) + "variance: " + str(var)
        pomocny = pomocny +1
        '''NASLEDUJICI RADEK LZE OMEZIT CISLEM PRO NETRENOVANI CELE MNOZINY'''
        if(pomocny >= pocetOrig): #if(pomocny >= pocetOrig): 
            break    
    print "vysledny prumer a variance:"
    print prumer
    print var
    print "vysledky ukladany do souboru 'Metoda1.p'"
    zapisPrumVar(prumer,var)
    return [prumer,var]    

class LiverSegmentation:
    """
    """
    def __init__(
        self,
        data3d,
        segparams={'some_parameter': 22},
        voxelsize=[1, 1, 1]
    ):
        """TODO: Docstring for __init__.

        :data3d: 3D array with data
        :segparams: parameters of segmentation
        :returns: TODO

        """
        # used for user interactivity evaluation
        self.interactivity_counter = 0
        # 3D array with object and background selections by user
        self.seeds = None
        self.voxelsize = voxelsize
        self.segmentation = np.zeros(data3d.shape, dtype=np.int8)
        pass

    def run(self):
        # @TODO dodělat
        self.segmentation[3:5, 13:17, :8] = 1
        pass
    
    
    def nacistTrenovaciData(self,path):
        pass

    def interactivity_loop(self, pyed):
        """
        Function called by seed editor in GUI

        :pyed: link to seed editor
        """
        # self.seeds = pyed.getSeeds()
        # self.voxels1 = pyed.getSeedsVal(1)
        # self.voxels2 = pyed.getSeedsVal(2)
        self.run()
        pass
    
    

def main():
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)


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
    
    pass



if __name__ == "__main__":
    main()
