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
from scipy import ndimage

def segmentace0(tabulka,velikostVoxelu):
    '''RYCHLA TESTOVACI METODA - PRO TESTOVANI
    Vybere pouze prvky blizke nule a to je cele
    vraci segmentaci ve formatu numpy Matrix'''
    #print np.shape(tabulka)
    segmentaceVysledek = []
    odchylka = 5
    prumer = 0

    zeli3=0
    for rez in tabulka:
        print str(zeli3+1) + '/' + str(len(tabulka))
        #print np.matrix(rez)
        rezNovy1 = ( (np.matrix(rez)>=prumer -2*odchylka))
        rezNovy2 = (np.matrix(rez)<=prumer +2*odchylka)
        rezNovy =np.multiply( rezNovy1, rezNovy2)
        rezNovy = rezNovy.astype(int)
        #seznam = rezNovy.tolist()
        seznam = rezNovy
        #print rezNovy
        segmentaceVysledek.append(seznam)       
        zeli3 = zeli3+1 #prochazeni rezu
    
     
    #print segmentaceVysledek  
    #print np.shape(tabulka)
    #print np.shape(segmentaceVysledek)
    return segmentaceVysledek

def segmentace1(tabulka,velikostVoxelu):
    '''PRIMITIVNI METODA - PRAHOVANI
    Nacte parametry prumer a odchylka ze souboru Metoda1.p
    pak pomoci prahovani vybere z kazdeho rezu cast z intervalu
    prumer +-2 sigma, nasledne provede binarni operace
    otevreni (1x) a uzavreni (3x)  tak aby byly odstraneny drobne pixely'''
    
    def nactiPrumVar():
        '''vrati pole [prumer,variance] nactene pomoci pickle ze souboru'''
        soubor = open('Metoda1.p','rb')
        vektor=pickle.load(soubor)
        prumer = vektor[0]
        variance = vektor[1]
        return [prumer,variance]
    
    [prumer,var] = nactiPrumVar() 
    odchylka = np.sqrt(var) 
    #print np.shape(tabulka)
    segmentaceVysledek = []
    zeli3=0
    for rez in tabulka:
        print str(zeli3+1) + '/' + str(len(tabulka))
        #print np.matrix(rez)
        rezNovy1 = ( (np.matrix(rez)>=prumer -2*odchylka))
        rezNovy2 = (np.matrix(rez)<=prumer +2*odchylka)
        rezNovy =np.multiply( rezNovy1, rezNovy2)
        rezNovy = rezNovy.astype(int)
        
        original = rezNovy
        struktura = [[0,1,1,1,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,1,1,1,0]]
        vylepseny = ndimage.binary_opening(original, struktura, 1)
        rezNovy = ndimage.binary_closing(vylepseny, struktura,3)
        
        #seznam = rezNovy.tolist()
        seznam = rezNovy
        #print rezNovy
        segmentaceVysledek.append(seznam)       
        zeli3 = zeli3+1 #prochazeni rezu
    
     
    #print segmentaceVysledek  
    #print np.shape(tabulka)
    #print np.shape(segmentaceVysledek)
    return segmentaceVysledek

def trenovaniCele(metoda):
    '''Metoda je cislo INT, dane poradim metody pri implementaci prace
    nacte cestu ze souboru Cesta.p, vsechny soubory v adresari
     natrenuje podle zvolene metody a zapise vysledek do TrenC.p. 
    ''', 
    soubor = open('Cesta.p','r')
    cesta = pickle.load(soubor)
    #print cesta
    seznamSouboru = vyhledejSoubory(cesta)
    
    vybrano = False
    
    if(metoda ==0):
        def metoda(cesta,seznamSouboru):
            vysledek =nahrazka(cesta,seznamSouboru)
            return vysledek
        vybrano = True
    
    if(metoda ==1):
        def metoda(cesta,seznamSouboru):
            vysledek =metoda1(cesta,seznamSouboru)
            return vysledek
        vybrano = True
    
    if(not vybrano):
        print "spatne zvolena metoda trenovani"
        return
        
    print "Probiha trenovani"
    vysledek1= metoda(cesta,seznamSouboru)
    soubor = open("TrenC.p","wb")
    pickle.dump(vysledek1,soubor)
    soubor.close()
    print "trenovani  dokonceno"

def nahrazka(cesta,seznamSouboru):
    '''METODA 0 - nahrada
    nahrazka trenovaci metody pro rychly beh a testovani'''
    return [25,3]

def trenovaniTri(metoda):
    '''Metoda je cislo INT, dane poradim metody pri implementaci prace
    nacte cestu ze souboru Cesta.p, vsechny soubory v adresari rozdeli na tri casti
    pro casti 1+2,2+3 a 1+3 natrenuje podle zvolene metody. 
    ulozene soubory: 1) seznam trenovanych souboru 2)seznam na kterych ma probehnout segmentace
    3) vysledek trenovani (napr. prumer a odchylka u metody 1)
    ''', 
    soubor = open('Cesta.p','r')
    cesta = pickle.load(soubor)
    #print cesta
    
    def rozdelTrenovaciNaTri(cesta):
        '''Rozdeli trenovaci mnozinu na tri dily'''
        vektorSouboru = vyhledejSoubory(cesta)
        delkaTrenovacich = len(vektorSouboru)/2
        delka1= round(float(len(vektorSouboru))/6)
        dily = [delka1,delka1,delkaTrenovacich-2*delka1]
        Cast1 = vektorSouboru[0:int(dily[0])] + vektorSouboru[delkaTrenovacich:int(dily[0])+delkaTrenovacich]
        Cast2 = vektorSouboru[int(dily[0]):int(dily[0])+int(dily[1])] + vektorSouboru[delkaTrenovacich+int(dily[0]):int(dily[0])+int(dily[1])+delkaTrenovacich]
        Cast3 = vektorSouboru[int(dily[0])+int(dily[1]):int(dily[0])+int(dily[1])+int(dily[2])]
        Cast3 = Cast3 + vektorSouboru[delkaTrenovacich+int(dily[0])+int(dily[1]):delkaTrenovacich+int(dily[0])+int(dily[1])+int(dily[2])]
        return[Cast1,Cast2,Cast3]
    
    [cast1,cast2,cast3] = rozdelTrenovaciNaTri(cesta)
    delka = len(cast1)/2
    delka3 = len(cast3)/2
    #print cast2
    tren12 = cast1[0:delka]+cast2[0:delka]+cast1[delka:delka*2]+cast2[delka:delka*2]
    #print tren12
    tren23 = cast2[0:delka]+cast3[0:delka3]+cast2[delka:delka*2]+cast3[delka3:delka3*2]
    tren13 = cast1[0:delka]+cast3[0:delka3]+cast1[delka:delka*2]+cast3[delka3:delka3*2]
    
    vybrano = False
    
    if(metoda ==0):
        def metoda(cesta,seznamSouboru):
            vysledek =nahrazka(cesta,seznamSouboru)
            return vysledek
        vybrano = True
    
    if(metoda ==1):
        def metoda(cesta,seznamSouboru):
            vysledek =metoda1(cesta,seznamSouboru)
            return vysledek
        vybrano = True
    if(not vybrano):
        print "spatne zvolena metoda trenovani"
        return
        
    print "Probiha trenovani Prvni Casti"
    vysledek1= metoda(cesta,tren12)
    soubor = open("Tren1+2.p","wb")
    pickle.dump(tren12,soubor)
    pickle.dump(cast3,soubor)    
    pickle.dump(vysledek1,soubor)
    soubor.close()
    print "Probiha trenovani druhe casti"
    vysledek2= metoda(cesta,tren23)
    soubor = open("Tren2+3.p","wb")
    pickle.dump(tren23,soubor)
    pickle.dump(cast1,soubor)    
    pickle.dump(vysledek2,soubor)
    soubor.close()
    print "Probiha trenovani treti casti"  
    vysledek3= metoda(cesta,tren13) 
    soubor = open("Tren1+3.p","wb")
    pickle.dump(tren13,soubor)
    pickle.dump(cast2,soubor)    
    pickle.dump(vysledek3,soubor)
    soubor.close()
    print "trenovani  dokonceno"
    
def zapisCestu():
    cesta = 'C:/Users/asus/workspace/training'
    print cesta
    soubor = open('Cesta.p','w')
    pickle.dump(cesta,soubor)
    soubor.close()
    print "cesta uspesne zapsana"
    
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
    
    def vypoctiPrumer(poctyVzorku,prumery):
        'vypocte prumer z prumeru a poctu vzorku vektoru ruzne delky'
        sumaPrumeru = 0
        sumaVzorku = 0
        pomocny = 0
        for pocet in poctyVzorku:
            sumaVzorku = sumaVzorku+pocet
            sumaPrumeru = sumaPrumeru+prumery[pomocny]*pocet
            pomocny = pomocny+1
        
        prumerCelkem = float(sumaPrumeru)/float(sumaVzorku)
        return prumerCelkem

    def vypoctiVar(poctyVzorku,prumery,variance,prumerCelkem):
        'vypocte varianci z prumeru varianci a poctu vzorku vektoru ruzne delky'
        sumaVar = 0
        sumaVzorku = 0
        pomocny = 0
        for pocet in poctyVzorku:
            sumaVzorku = sumaVzorku+pocet
            pomocny = pomocny+1
        #mam sumuVzorku
        pomocny = 0
        for minivar in variance:
            scitanec = float( minivar*poctyVzorku[pomocny])/sumaVzorku
            nasobitel = poctyVzorku[pomocny]*((prumery[pomocny]-prumerCelkem)**2)/sumaVzorku
            sumaVar = sumaVar+nasobitel+scitanec
            pomocny = pomocny+1        
        return sumaVar
    
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
    
    def zpracuj(cesta,seznamSouboru,pomocny,ctenar,pocetOrig):
        originalni = nactiSoubor(cesta,seznamSouboru,pomocny,ctenar) #originalni pole
        segmentovany = nactiSoubor(cesta,seznamSouboru,pomocny+pocetOrig,ctenar) #segmentovane pole(0)

        #print originalni[1]
        #print segmentovany[1]
        pole1 = np.asarray(originalni[1])
        pole2 = np.asarray(segmentovany[1])
        #print np.linalg.norm(pole1-pole2)

        if (not(np.linalg.norm(pole1-pole2) <= 10**(-2))):
            raise NameError('Chyba ve vstupnich datech original c.' + str(pomocny+1) + ' se neshoduje se segmentaci')
        
        '''ZDE PRACOVAT S originalni A segmentovany'''       
       
        poleSeg = segmentovany[0]#nuly jsou kde neni segmentace jednicky kde je
        poleOri = originalni[0]
        
        kombinace = np.multiply(poleSeg,poleOri)#skalarni soucin
        X = np.ma.masked_equal(kombinace,0)
        bezNul = X.compressed()
        return bezNul
    
    print "zahajeno trenovani metodou c.1"
    pocetSouboru = len(seznamSouboru)
    pocetOrig = pocetSouboru/2    
    ctenar =  io3d.DataReader()
    
    prumery = []
    variance = []    
    poctyVzorku=[]  
    
    
    pomocny = 0
    for soubor in seznamSouboru:
        ukazatel = str(pomocny+1) + "/" + str(pocetOrig)
        print ukazatel
        bezNul = zpracuj(cesta,seznamSouboru,pomocny,ctenar,pocetOrig)
        
        prumery.append(np.mean(bezNul))
        variance.append(np.var(bezNul))
        poctyVzorku.append(len(bezNul))
        originalni = 0
        segmentovany = 0
        
        pomocny = pomocny +1
        '''NASLEDUJICI RADEK LZE OMEZIT CISLEM PRO NETRENOVANI CELE MNOZINY'''
        #print (pomocny+1 >= pocetOrig)
        if(pomocny+1 >= pocetOrig): #if(pomocny >= pocetOrig): 
            print "trenovani ukonceno"
            break    
    prumer = vypoctiPrumer(poctyVzorku,prumery)
    var = vypoctiVar(poctyVzorku,prumery,variance,prumer)

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
        voxelsize=[1, 1, 1],segparams={'cisloMetody':0,'some_parameter': 22}
    ):
        """TODO: Docstring for __init__.

        :data3d: 3D array with data
        :segparams: parameters of segmentation
        :returns: TODO

        """
        # used for user interactivity evaluation
        self.data3d = data3d
        self.interactivity_counter = 0
        # 3D array with object and background selections by user
        self.seeds = None
        self.voxelSize = voxelsize
        self.segParams = {'cisloMetody':0}
        self.segmentation = np.zeros(data3d.shape, dtype=np.int8)
        pass
    
    def setCisloMetody(self,cislo):
        self.segParams['cisloMetody'] = cislo

    def run(self):
        # @TODO dodělat
        self.segmentation[3:5, 13:17, :8] = 1
        pass
    
    def runVolby(self):
        '''metoda s vice moznostmi vyberu metody-vybrana v segParams'''
        numero = self.segParams['cisloMetody']
        spatne = True
        
        if(numero == 0):
            print('testovaci metoda')            
            self.segmentation = segmentace0(self.data3d,self.voxelSize)
            spatne = False
        if(numero == 1):            
            self.segmentation = segmentace1(self.data3d,self.voxelSize)
            spatne = False
        
        if(spatne):
            print('Zvolena metoda nenalezena')
        
    
    
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
