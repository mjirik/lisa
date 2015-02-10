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
import yaml
from scipy import ndimage
import volumetry_evaluation as ve

def vyhodnoceniMetodyTri(metoda):
    '''metoda- int cislo metody (poradi pri vyvoji) 
    nacte cestu ze souboru path.yml, dale nacte soubory v adresari kde je situovana a sice 
    Tren1+2.yml, Tren1+3.yml a Tren2+3.yml. Pri nacteni souboru vznikne pole:
    [seznamSouboruTrenovaciMnoziny(nepodstatny),seznamSouboruTESTOVACImnoziny,vysledkyMETODY]
    na souborech ze seznamuTestovacimnoziny provede segmentaci metodou s cislem METODA
    a zapise do souboru vysledky.yml pole 
    [vsechnyVysledky,prumerScore]
    se vsemi vysledky a take vypise prumer na konzoli
    '''
    ctenar = io3d.DataReader()
    cesta = nactiYamlSoubor('path.yml')
    
    def segmentujSubor(nazevSouboru):
        [seznamTM,seznamTestovaci,vysledky]= nactiYamlSoubor(nazevSouboru)#'Tren1+2.yml'
        seznamVsechVysledku = []
        for x in seznamTM: #male overeni spravnosti testovacich a trenovacich dat
            if(seznamTestovaci.count(x)>0):
                print('TM obsahuje testovaci data!!!')
                
        def iterace(cesta,seznamTestovaci,ctenar,counter):
            #print polozka
            '''ZDE UPRAVIT'''
            nacti = nactiSoubor(cesta,seznamTestovaci,counter,ctenar)#
            poleTest = nacti[0]
            voxelSizeTest = nacti[1]
            slovnik = {'cisloMetody':metoda,'vysledkyDostupne':vysledky}
            #print slovnik['vysledkyDostupne']
            vytvoreny = LiverSegmentation(poleTest,voxelSizeTest,slovnik)
            #vysledky ==> nenacitaji se data z trenovani cele
            vytvoreny.setCisloMetody(metoda)#prirazeniMetody
            #print vytvoreny.segParams
            vytvoreny.runVolby()
            segmentovany = vytvoreny.segmentation
            segmentovanyVoxelSize = vytvoreny.voxelSize
            nactiNovy = nactiSoubor(cesta,seznamTestovaci,counter+maximum,ctenar)
            poleRucni = nactiNovy[0]
            voxelSizeRucni = nactiNovy[1]
            poleRucni = np.array(poleRucni)
            print segmentovany 
            print poleRucni
            return[segmentovany,segmentovanyVoxelSize,poleRucni,voxelSizeRucni]
            
        counter = 0
        maximum = len(seznamTestovaci)/2
        for polozka in seznamTestovaci:
            #[segmentovany,segmentovanyVoxelSize,poleRucni,voxelSizeRucni] = iterace(cesta,seznamTestovaci,ctenar,counter)
            #skoreData = vyhodnoceniSnimku(segmentovany,segmentovanyVoxelSize,poleRucni,voxelSizeRucni)
            '''PLACEHOLDER, NAHRADIT '''
            skoreData = [polozka,0] #
            seznamVsechVysledku.append(skoreData)
            
            counter = counter+1
            if(counter >= maximum): #preruseni pro vycerpani originalu
                break
                #pass
        return seznamVsechVysledku
    print 'ANALYZA PRVNI TRETINY'
    seznam1 = segmentujSubor('Tren1+2.yml')
    print 'ANALYZA DRUHE TRETINY'
    seznam2 = segmentujSubor('Tren1+3.yml')
    print 'ANALYZA TRETI TRETINY'
    seznam3 = segmentujSubor('Tren2+3.yml')
    seznamVsech = seznam1+seznam2+seznam3
    prumerSeznam = np.zeros(len(seznamVsech))
    pomocnik = 0
    for polozka in seznamVsech:
        prumerSeznam[pomocnik] = polozka[1]
        pomocnik = pomocnik+1
    celkovyPrumer = np.mean(prumerSeznam)
    zapsat = [seznamVsech,celkovyPrumer]

    print 'celkovy prumer je: ' + str(celkovyPrumer)
    zapisYamlSoubor('vysledky.yml',zapsat)
    print 'soubory zapsany do vysledky.yml'
    
    return

def vyhodnoceniSnimku(snimek1,voxelsize1,snimek2,voxelsize2):
    '''Provede vyhodnoceni snimku pomoci metod z volumetry_evaluation,
    slucuje dve metody a vraci pole [evalData (slovnik),score(%)],
    dale protoze velikosti voxelu se mirne lisi u rucni segmentace
    a originalniho obrazku (10^-2 a mene) udela z nich prumer
    '''
    print 'probiha vyhodnoceni snimku pockejte prosim'
    voxelsize_mm = [((voxelsize1[0]+voxelsize2[0])/2.0),((voxelsize1[1]+voxelsize2[1])/2.0),((voxelsize1[2]+voxelsize2[2])/2.0)]#prumer z obou
    snimek1 = np.array(snimek1)
    snimek2 = np.array(snimek2)
    evaluace = ve.compare_volumes(snimek1, snimek2, voxelsize_mm)
    #score = ve.sliver_score_one_couple(evaluace)
    score = 0
    vysledky = [evaluace,score]
    
    return vysledky

def zapisYamlSoubor(nazevSouboru,Data):
    '''DATA NUTNO ZAPSAT V 1 BEHU, nejlepe 1 pole
    Zapise Data do souboru (.yml) nazevSouboru '''
    with open(nazevSouboru, 'w') as outfile:
        outfile.write( yaml.dump(Data, default_flow_style=True) )
    return

def nactiYamlSoubor(nazevSouboru):
    '''nacte data z (.yml) souboru nazevSouboru'''
    soubor = open(nazevSouboru,'r')
    dataNova = yaml.load(soubor)
    return dataNova

def segmentace0(tabulka,velikostVoxelu,vysledky = False):
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

def segmentace1(tabulka,velikostVoxelu,source='Metoda1.yml',vysledky = False):
    '''PRIMITIVNI METODA - PRAHOVANI
    Nacte parametry prumer a odchylka ze souboru Metoda1.yml
    (lze zmenit pomoci volitelneho argumentu source)
    pak pomoci prahovani vybere z kazdeho rezu cast z intervalu
    prumer +-2 sigma, nasledne provede binarni operace
    otevreni (1x) a uzavreni (3x)  tak aby byly odstraneny drobne pixely
    metode lze take zadat vysledky
    '''
    
    def nactiPrumVar():
        '''vrati pole [prumer,variance] nactene pomoci yaml ze souboru'''
        source = 'Metoda1.yml'
        vektor=nactiYamlSoubor(source)
        prumer = vektor[0]
        variance = vektor[1]
        return [prumer,variance]
    
    if(vysledky == False): #v pripade nezadani vysledku
        [prumer,var] = nactiPrumVar()
    else:
        prumer = vysledky[0]  
        var = vysledky[1]
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
    nacte cestu ze souboru path.yml, vsechny soubory v adresari
     natrenuje podle zvolene metody a zapise vysledek do TrenC.yml. 
    '''
    cesta = nactiYamlSoubor('path.yml')
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
    soubor = open("TrenC.yml","wb")
    zapisYamlSoubor("TrenC.yml",vysledek1)
    print "trenovani  dokonceno"

def nahrazka(cesta,seznamSouboru):
    '''METODA 0 - nahrada
    nahrazka trenovaci metody pro rychly beh a testovani'''
    return [25,3]

def trenovaniTri(metoda):
    '''Metoda je cislo INT, dane poradim metody pri implementaci prace
    nacte cestu ze souboru path.yml, vsechny soubory v adresari rozdeli na tri casti
    pro casti 1+2,2+3 a 1+3 natrenuje podle zvolene metody. 
    ulozene soubory: 1) seznam trenovanych souboru 2)seznam na kterych ma probehnout segmentace
    3) vysledek trenovani (napr. prumer a odchylka u metody 1)
    '''
    cesta = nactiYamlSoubor('path.yml')
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
    poleMega = [tren12,cast3,vysledek1]
    zapisYamlSoubor("Tren1+2.yml",poleMega)
    
    print "Probiha trenovani druhe casti"
    vysledek2= metoda(cesta,tren23)
    poleMega = [tren23,cast1,vysledek2]
    zapisYamlSoubor("Tren2+3.yml",poleMega)
    
    print "Probiha trenovani treti casti"  
    vysledek3= metoda(cesta,tren13) 
    poleMega = [tren13,cast2,vysledek3]
    zapisYamlSoubor("Tren1+3.yml",poleMega)
    print "trenovani  dokonceno"
    
def zapisCestu():
    cesta = 'C:/Users/asus/workspace/training'
    print cesta
    zapisYamlSoubor('path.yml',cesta)
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
    hodnota zapsana do souboru "Metoda1.yml" '''
    
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
        zapisYamlSoubor('Metoda1.yml',radek)
    
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
    print "vysledky ukladany do souboru 'Metoda1.yml'"
    zapisPrumVar(prumer,var)
    return [prumer,var]  

class LiverSegmentation:
    """
    """
    def __init__(
        self,
        data3d,
        voxelsize=[1, 1, 1],segparams={'cisloMetody':0,'vysledkyDostupne':False,'some_parameter': 22}
    ):
        """TODO: Docstring for __init__.

        :data3d: 3D array with data
        :segparams: parameters of segmentation
        cisloMetody = INT, cislo pouzite metody (0-testovaci)
        vysledkyDostupne = F/vysledek, F =>vysledek se nacte, jinak se vezme
        :returns: TODO

        """
        # used for user interactivity evaluation
        self.data3d = data3d
        self.interactivity_counter = 0
        # 3D array with object and background selections by user
        self.seeds = None
        self.voxelSize = voxelsize
        self.segParams = segparams
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
        print self.segParams
        vysledek = self.segParams['vysledkyDostupne']
        spatne = True
        
        if(numero == 0):
            print('testovaci metoda')            
            self.segmentation = segmentace0(self.data3d,self.voxelSize,vysledek)
            spatne = False
        if(numero == 1):            
            self.segmentation = segmentace1(self.data3d,self.voxelSize,vysledek)
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
