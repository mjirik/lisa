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
import sed3
import qmisc
import nearpy


def otestujVzdalenost(vol1, vol2, voxelsize_mm):
    engine = nearpy.Engine()
    [avgd, rmsd, maxd] = vzdalenosti(vol1, vol2, voxelsize_mm,engine)
    return [avgd, rmsd, maxd] 

def compare_volumesUPRAVA(vol1, vol2, voxelsize_mm,engine):
    """
    vol1: reference
    vol2: segmentation
    """
    volume1 = np.sum(vol1 > 0)
    volume2 = np.sum(vol2 > 0)
    volume1_mm3 = volume1 * np.prod(voxelsize_mm)
    volume2_mm3 = volume2 * np.prod(voxelsize_mm)
    logger.debug('vol1 [mm3]: ' + str(volume1_mm3))
    logger.debug('vol2 [mm3]: ' + str(volume2_mm3))

    df = vol1 - vol2
    df1 = np.sum(df == 1) * np.prod(voxelsize_mm)
    df2 = np.sum(df == -1) * np.prod(voxelsize_mm)

    logger.debug('err- [mm3]: ' + str(df1) + ' err- [%]: '
                 + str(df1 / volume1_mm3 * 100))
    logger.debug('err+ [mm3]: ' + str(df2) + ' err+ [%]: '
                 + str(df2 / volume1_mm3 * 100))

    # VOE[%]
    intersection = np.sum(df != 0).astype(float)
    union = (np.sum(vol1 > 0) + np.sum(vol2 > 0)).astype(float)
    voe = 100 * ((intersection / union))
    logger.debug('VOE [%]' + str(voe))

    # VD[%]
    vd = 100 * (volume2 - volume1).astype(float) / volume1.astype(float)
    logger.debug('VD [%]' + str(vd))
    # import pdb; pdb.set_trace()

    # pyed = sed3.sed3(vol1, contour=vol2)
    # pyed.show()

    # get_border(vol1)
    [avgd, rmsd, maxd] = vzdalenosti(vol1, vol2, voxelsize_mm,engine)
    logger.debug('AvgD [mm]' + str(avgd))
    logger.debug('RMSD [mm]' + str(rmsd))
    logger.debug('MaxD [mm]' + str(maxd))
    evaluation = {
        'volume1_mm3': volume1_mm3,
        'volume2_mm3': volume2_mm3,
        'err1_mm3': df1,
        'err2_mm3': df2,
        'err1_percent': df1 / volume1_mm3 * 100,
        'err2_percent': df2 / volume1_mm3 * 100,
        'voe': voe,
        'vd': vd,
        'avgd': avgd,
        'rmsd': rmsd,
        'maxd': maxd
    }
    return evaluation

def vzdalenosti(pole1,pole2,voxelSize_mm,engine):
    '''sample_MAX = maximalni pocet vzorku se kterym se pocita
    vstup: pole1/2 = 3d T/F pole objektu1/2, voxelsize = pole velikosti voxelu
    vystup = [asd,rmsd,maxd] prumerne a maximalni vzdalenosti mezi objekty
    vypocteni vzdalenosti mezi povrchy poli presne podle PDF
    pouziva metodu approximate nearest neighbout (ANN) z knihovny nearpy'''
    
    sample_MAX = 8000.0 #maximalni pocet vzorku objektu OPTIMALMI SE ZDA 8000
    
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
    
    def pridejVektor(a,b,c,voxelSize,engine):
        vektor = np.array([a*voxelSize_mm[0],b*voxelSize_mm[1],c*voxelSize_mm[1]])
        #print vektor
        engine.store_vector(vektor)
    
    def nejblizsi(a,b,c,voxelSize,engine):
        vzdalenost = 0
        vektor = np.array([a*voxelSize_mm[0],b*voxelSize_mm[1],c*voxelSize_mm[1]])
        #print vektor
        vysledek = engine.neighbours(vektor)
        if(vysledek != []):
            vzdalenost = vysledek[0][2]
        return vzdalenost
    
    
    
    print 'zahajuje se vypocet vzdalenosti mezi dvema objekty'
    print 'maximalni pocet vzorku: '+str(sample_MAX)
    
    border1 = ve.get_border(pole1)
    border2 = ve.get_border(pole2)
    
    vektory1 = np.where(border1 == 1) #pole s informaci o vektorech (jejich pozice v souradnicich)
    vektory2= np.where(border2 == 1)
    
    
    downsampling = float(len(vektory1[0]))/sample_MAX #'Downsampling pro urychleni vypoctu na unosnou mez'
    downsampling = np.round(downsampling,2)
    #print np.round(downsampling) 
    print 'ukladaji se data o objektu 1 '    
    counter = 0
    for x in range(len(vektory1[0])):
        if((x %100000) == 0):
            print str(x) + '/' + str(len(vektory1[0]))
        if((x % downsampling) == 0):
            a = vektory1[0][x]
            b = vektory1[1][x]
            c = vektory1[2][x]  
            pridejVektor(a,b,c,voxelSize_mm,engine)
            counter = counter+1
    vzdalenosti2k1 = []  #pole kde budou ulozeny vzdalenosti od objektu 1 k objektu 2
    'v engine jsou ulozeny vektory z border1'
    print 'probiha vypocet vzdalenosti mezi 2 a 1'
    for x in range(len(vektory2[0])):
        if((x %100000) == 0):
            print str(x) + '/' + str(len(vektory1[0]))
        if((x % downsampling) == 0):
            #print str(x) + '/' + str(len(vektory1[0]))
            a = vektory2[0][x]
            b = vektory2[1][x]
            c = vektory2[2][x] 
            vysledek =nejblizsi(a,b,c+5,voxelSize_mm,engine)
            vzdalenosti2k1.append( vysledek)
    vzdalenosti2k1 = np.array( vzdalenosti2k1)
            
    print 'vzdalenosti 2-1 vypocteny, vyprazdnovani enginu'
    #print vzdalenosti2k1.shape  
    engine.clean_all_buckets()      
    
    print 'ukladaji se data o objektu 2 '    
    counter = 0
    for x in range(len(vektory2[0])):
        if((x %100000) == 0):
            print str(x) + '/' + str(len(vektory2[0]))
        if((x % downsampling) == 0):
            a = vektory2[0][x]
            b = vektory2[1][x]
            c = vektory2[2][x]  
            pridejVektor(a,b,c,voxelSize_mm,engine)
            counter = counter+1
    vzdalenosti1k2 = []
    'v engine jsou ulozeny vektory z border2'
    print 'probiha vypocet vzdalenosti mezi 1 a 2'
    for x in range(len(vektory1[0])): 
        if((x %100000) == 0):
            print str(x) + '/' + str(len(vektory1[0]))       
        if((x % downsampling) == 0):
            #print str(x) + '/' + str(len(vektory1[0]))
            a = vektory1[0][x]
            b = vektory1[1][x]
            c = vektory1[2][x] 
            vysledek =nejblizsi(a,b,c+5,voxelSize_mm,engine)
            vzdalenosti1k2.append(vysledek)
    engine.clean_all_buckets() 
    vzdalenosti1k2 = np.array( vzdalenosti1k2)        
    print 'vzdalenosti 1-2 vypocteny'    
    #print vzdalenosti2k1  
    #print vzdalenosti1k2  
    print 'vypocet asd,rmsd,msd'
    'vypocet prumeru ze VSECH vzdalenosti'  
    prumery = [np.mean(vzdalenosti2k1),np.mean(vzdalenosti1k2)]
    poctyVzorku = [len(vzdalenosti2k1),len(vzdalenosti1k2)]
    celkovyPrumer = vypoctiPrumer(poctyVzorku,prumery)
    asd = celkovyPrumer
    rmsd =np.sqrt(1/float(len(vzdalenosti2k1)*len(vzdalenosti1k2)))#* np.sqrt(np.sum(vzdalenosti2k1**2)+np.sum(vzdalenosti1k2**2))
    maxd = max(np.max(vzdalenosti2k1), np.max(vzdalenosti1k2))
    print [asd,rmsd,maxd]
    return [asd,rmsd,maxd]

def souborAsegmentace(cisloSouboru,cisloMetody,cesta):
    '''nacte soubor a vytvori jeho segmentaci, pomoci zvolene metody
    vrati parametry potrebne pro evaluaci'''
    ctenar = io3d.DataReader()
    seznamSouboru = vyhledejSoubory(cesta)
    print 'probiha nacitani souboru'
    vektor = nactiSoubor(cesta,seznamSouboru,(cisloSouboru+len(seznamSouboru)/2),ctenar)
    rucniPole = vektor[0]
    rucniVelikost = vektor[1]
    vektor2 = nactiSoubor(cesta,seznamSouboru,cisloSouboru,ctenar)
    originalPole = vektor2[0]
    originalVelikost = vektor2[1]
    vytvoreny = LiverSegmentation(originalPole,originalVelikost)
    vytvoreny.setCisloMetody(cisloMetody) #ZVOLENI METODY
    vytvoreny.runVolby()
    segmentovany = vytvoreny.segmentation
    segmentovanyVelikost = vytvoreny.voxelSize
    rucniPole = np.array(rucniPole)
    segmentovany = np.array(segmentovany)
    return [rucniPole,rucniVelikost,segmentovany,segmentovanyVelikost,originalPole]


def zobrazUtil(cmetody,cisloObrazu = 1):
    'utilita pro rychle zobrazeni metody s cislem cmetody'
    cesta = nactiYamlSoubor('path.yml')
    
    [rucni,rucniVelikost,strojova,segmentovanyVelikost,original] = souborAsegmentace(cisloObrazu,cmetody,cesta)
    #segmentace = np.zeros(rucni.shape, dtype=np.int8)
    #segmentace[0:-1,100:400,100:400] = 1    
    zobrazit(original,rucni,strojova)

def zobrazit(original,rucni,strojova):
    '''Metoda pro srovnani rucni a 
    automaticke segmentace z lidskeho pohledu
    cerna oblast - shoda strojoveho a rucniho
    bila oblast - neshoda strojoveho a rucniho'''
    
    '''
    cesta2 = 'C:/Users/asus/workspace/training'
    ctenar = io3d.DataReader()
    datap = ctenar.Get3DData(cesta, dataplus_format=False)
    seznamSouboru = main.vyhledejSoubory(cesta2)
    vektor = main.nactiSoubor(cesta2,seznamSouboru,21,ctenar)
    rucniPole = vektor[0]
    rucniVelikost = vektor[1]
    vektor2 = main.nactiSoubor(cesta2,seznamSouboru,1,ctenar)
    originalPole = vektor2[0]
    originalVelikost = vektor2[1]
    segmentace = np.zeros(rucniPole.shape, dtype=np.int8)
    segmentace[0:-1,100:400,100:400] = 1
    #rez sloupec radek
    zobrazit(originalPole,rucniPole,segmentace)
    '''
    souhlas = 1300
    nesouhlas = -1000
    ukazatel1 = -9000
    ukazatel2 = 9000
    ctenar = io3d.DataReader()
    '''
    opak = rucni*(-1)+1 #kde neni rucni segmentace   
    opakStrojova = strojova*(-1)+1 
    kombinaceSouhlas = np.multiply(rucni,strojova)#skalarni soucin
    kombinaceSouhlas = kombinaceSouhlas * ukazatel1
    kombinaceNesouhlas = np.multiply(opak,strojova)
    kombinaceNesouhlas = kombinaceNesouhlas * ukazatel2
    kombinace = original+kombinaceNesouhlas+kombinaceSouhlas
  
    '''
    blbe = np.abs(-strojova+rucni)*9000
    dobre = np.multiply(rucni,strojova)*(-9000)
    kombinace = dobre+blbe+ original#obmena
    
    kombinace[kombinace > 4000] = souhlas
    kombinace[kombinace < -4000] = nesouhlas
    #poleVysledek = kombinace   
    ed = sed3.sed3(kombinace)
    #print kombinaceNesouhlas
    ed.show()
    return

def vyhodnoceniMetodyTri(metoda):
    '''metoda- int cislo metody (poradi pri vyvoji) 
    nacte cestu ze souboru path.yml, dale nacte soubory v adresari kde je situovana a sice 
    Tren1+2.yml, Tren1+3.yml a Tren2+3.yml. Pri nacteni souboru vznikne pole:
    [seznamSouboruTrenovaciMnoziny(nepodstatny),seznamSouboruTESTOVACImnoziny,vysledkyMETODY]
    na souborech ze seznamuTestovacimnoziny provede segmentaci metodou s cislem METODA
    a zapise do souboru vysledky.yml pole 
    [vsechnyVysledky,prumerScore]
    se vsemi vysledky a take vypise prumer na konzoli'''
    
    def nacteniMnoziny(nazevSouboru,cesta,metoda):
        [seznamTM,seznamTestovaci,vysledky]= nactiYamlSoubor(nazevSouboru)#'Tren1+2.yml'
        #print seznamTM
        ctenar = io3d.DataReader()
        seznamVsechVysledku = []
        for x in range(len(seznamTestovaci)): #male overeni spravnosti testovacich a trenovacich dat
            if(x >= len(seznamTestovaci)/2):
                break
            vysledek = vyhodnotSoubor(cesta,x,seznamTestovaci,ctenar,vysledky,metoda)
            seznamVsechVysledku.append(vysledek)
        return seznamVsechVysledku
                
    def vyhodnotSoubor(cesta,x,seznamTestovaci,ctenar,vysledky,metoda):
        originalNazev =  [seznamTestovaci[x]]
        rucniNazev = [seznamTestovaci[x+len(seznamTestovaci)/2]]        
        vektor = nactiSoubor(cesta,rucniNazev,0,ctenar)
        rucniPole = vektor[0]
        rucniVelikost = vektor[1]
        vektor2 = nactiSoubor(cesta,originalNazev,0,ctenar)
        originalPole = vektor2[0]
        originalVelikost = vektor2[1]
        slovnik = {'cisloMetody':metoda,'vysledkyDostupne':vysledky}
        vytvoreny = LiverSegmentation(originalPole,originalVelikost,slovnik)
        vytvoreny.setCisloMetody(metoda) #ZVOLENI METODY
        vytvoreny.runVolby()
        segmentovany = vytvoreny.segmentation
        segmentovanyVelikost = vytvoreny.voxelSize
        engine = nearpy.Engine(dim = 3)
        vysledky = vyhodnoceniSnimku(rucniPole,rucniVelikost,segmentovany,segmentovanyVelikost,engine)   
        #vysledky =[1,2]    
        
        return vysledky
    
    cesta = nactiYamlSoubor('path.yml')
    print 'ANALYZA PRVNI TRETINY'
    seznam1 =nacteniMnoziny('Tren1+2.yml',cesta,metoda)
    print 'ANALYZA DRUHE TRETINY'
    seznam2 = nacteniMnoziny('Tren1+3.yml',cesta,metoda)
    print 'ANALYZA TRETI TRETINY'
    seznam3 = nacteniMnoziny('Tren2+3.yml',cesta,metoda)
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



def vyhodnoceniMetodyTriX(metoda):
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
    engine = nearpy.Engine(dim = 3) #vytvoreni enginu
    
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
            #print segmentovany 
            #print poleRucni
            return[segmentovany,segmentovanyVoxelSize,poleRucni,voxelSizeRucni]
        def pomocna(cesta,seznamTestovaci,ctenar,counter):
            [segmentovany,segmentovanyVoxelSize,poleRucni,voxelSizeRucni] = iterace(cesta,seznamTestovaci,ctenar,counter)
            skoreData = vyhodnoceniSnimku(segmentovany,segmentovanyVoxelSize,poleRucni,voxelSizeRucni,engine)
            return skoreData
        counter = 0
        maximum = len(seznamTestovaci)/2
        for polozka in seznamTestovaci:
            skoreData = pomocna(cesta,seznamTestovaci,ctenar,counter)
            '''PLACEHOLDER, NAHRADIT '''
            #skoreData = [polozka,0] #
            seznamVsechVysledku.append(skoreData)
            
            counter = counter+1
            if(counter >= maximum): #preruseni pro vycerpani originalu
                break
                #pass
        return seznamVsechVysledku
    
    
    
    print 'ANALYZA PRVNI TRETINY'
    seznam1 =segmentujSubor('Tren1+2.yml')
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

def vyhodnoceniSnimku(snimek1,voxelsize1,snimek2,voxelsize2,engine):
    '''Provede vyhodnoceni snimku pomoci metod z volumetry_evaluation,
    slucuje dve metody a vraci pole [evalData (slovnik),score(%)],
    dale protoze velikosti voxelu se mirne lisi u rucni segmentace
    a originalniho obrazku (10^-2 a mene) udela z nich prumer
    '''
    print 'probiha vyhodnoceni snimku pockejte prosim'
    voxelsize_mm = [((voxelsize1[0]+voxelsize2[0])/2.0),((voxelsize1[1]+voxelsize2[1])/2.0),((voxelsize1[2]+voxelsize2[2])/2.0)]#prumer z obou
    snimek1 = np.array(snimek1)
    snimek2 = np.array(snimek2)
    evaluace = compare_volumesUPRAVA(snimek1, snimek2, voxelsize_mm,engine)
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
    velikost = np.shape(tabulka)
    a = velikost[0]
    b = velikost[1]
    c = velikost[2]
    segmentaceVysledek = np.zeros(tabulka.shape)
    segmentaceVysledek[a/4:3*a/4,b/4:3*b/4,c/4:3*c/4] = 1

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
    print 'pouzita metoda 1'
    konstanta = 0.5 #EXPERIMENTALNE NALEZENA KONSTANTA
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
    mezHorni = prumer +konstanta*odchylka
    mezDolni = prumer -konstanta*odchylka
    
    for rez in tabulka:
        print str(zeli3+1) + '/' + str(len(tabulka))
        rezNovy1 = ( (np.array(rez)>=mezDolni))
        rezNovy2 = (np.array(rez)<=prumer +mezHorni)
        rezNovy =np.multiply( rezNovy1, rezNovy2)
        rezNovy = rezNovy.astype(int)
        
        seznam = rezNovy
        segmentaceVysledek.append(seznam)       
        zeli3 = zeli3+1 #prochazeni rezu
    
    
    #ed = sed3.sed3(np.array(segmentaceVysledek))
    #print kombinaceNesouhlas
    #ed.show()
     
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
        #print self.segParams
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
