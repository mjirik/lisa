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
import SimpleITK as sitk
import os.path as op
import volumetry_evaluation as ve
import sed3

def segmentace4(tabulka,velikostVoxelu,source='Metoda1.yml',vysledky = False):
    '''binarni operace nasledovane region growingem
    vzorkovaciKonstanta === pocitac urci sam
    hrannicni konstanta = ohraniceni region growingu kolem seedu v mm
    maxSeedKonstanta = pocet vybranych seedu (rovnomerne)'''

    print 'pouzita metoda 4 pouzivajici metodu 3'
    vzorkovaciKonstanta = np.shape(tabulka)[0]*0.33+33 #*0.33+22 #KONSTANTA VYPOCITANA Z DELKY 3D SNIMKU (ROZLISENI)
    hranicniKonstanta = 45.0 #50 a 10 vypada dobre 35 a 50seed =malo/ moc ///45 20 nej
    maxSeedKonstanta = 30
    #print vzorkovaciKonstanta
    #vzorkovaciKonstanta = 80
    #return
    binarniOperace = segmentace3(tabulka,velikostVoxelu,source='Metoda1.yml',vysledky = False)
    segmentaceVysledek = binarniOperace
    'konstanta urcuje rozmezi region growingu'
    segmentaceVysledek = regionGrowingCTIF(tabulka,binarniOperace,velikostVoxelu,konstanta = vzorkovaciKonstanta,
                                           konstantaHranice = hranicniKonstanta,maxSeeds = maxSeedKonstanta)


    return segmentaceVysledek

def regionGrowingCTIF(ctImage,array,velikostVoxelu,konstanta = 100,konstantaHranice = 5.0,maxSeeds = None):
    ''' ctImage = ct snimek, array = numpy aray po binarnich operacich (lokalizace vnitrku jater
    konstanta = rozmezi v kterem jsou pridavany pixely k seedu je urcovana z poctu rezu
    konstantaHranice = delka kterou je RG ohranicen v mm
    vysledek = pole array s vetsi segmentaci'''

    'Krok 1 vybrani seedu z array'


    segmentationFilter = sitk.ConnectedThresholdImageFilter()
    segmentationFilter.SetConnectivity(segmentationFilter.FaceConnectivity) #FaceConnectivity , FullConectivity
    segmentationFilter.SetReplaceValue( 1 )

    original = sitk.GetImageFromArray(ctImage)
    #segmentaceImg = sitk.GetImageFromArray(array) #nepotrebne, zustane ve formatu numpy

    border = ve.get_border(array)

    vektory = np.where(array == 1) #pole s informaci o vektorech (jejich pozice v souradnicich)
    #print vektory

    #print 'ukladaji se data o objektu 1 '

    def iterace(x,vektory,segmentationFilter,konstanta,array,koule):
        '''Vyrazne setreni pameti '''
        a = int(vektory[0][x])
        b = int(vektory[1][x])
        c = int(vektory[2][x])

        seed = [c,b,a] #SITK ma souradnice obracene
        #print seed
        bod = [a,b,c]
        segmentationFilter.AddSeed(seed)
        hodnota = original.GetPixel(*seed )
        segmentationFilter.SetLower( hodnota-konstanta )
        segmentationFilter.SetUpper( hodnota+konstanta )
        vysledek = segmentationFilter.Execute( original )
        vysledek[seed] = 1




        ukazat = sitk.GetArrayFromImage(vysledek)
        ukazat.astype(np.int8)

        'Omezeni rozsahu'
        okoli = vytvor3DobjektPole(koule,ukazat,bod)
        ukazat = np.multiply(okoli,ukazat)
        'konec omezeni rozsahu'
        array = np.add(array,ukazat)
        array = array > 0 #array = array > 0
        #print ukazat #kontrola
        return array

    'Downsampling pro udrzitelnost vypoctu'
    if(maxSeeds == None):
        sample_MAX = 30.0 #pocet seedu
    else:
        sample_MAX = maxSeeds
    downsampling = float(len(vektory[0]))/sample_MAX #'Downsampling pro urychleni vypoctu na unosnou mez'
    downsampling = np.round(downsampling,0)
    #print 'downsampling'
    #print downsampling

    koule = vytvorKouli3D(velikostVoxelu,konstantaHranice)
    #koule = 0

    for x in range(len(vektory[0])):
        if((x % (len(vektory[0])/5)) == 0):
            print str(x) +'/' + str(len(vektory[0]))
        if((x % downsampling) == 0):
            array = iterace(x,vektory,segmentationFilter,konstanta,array,koule)





    return array

def vytvoritTFKruznici(polomerPole,polomerKruznice):
    '''vytvori 2d pole velikosti 2xpolomerPole+1
     s kruznici o polomeru polomerKruznice uprostred '''
    radius = polomerPole
    r2 = np.arange(-radius, radius+1)**2
    dist2 = r2[:, None] + r2
    vratit =  (dist2 <= polomerKruznice**2).astype(np.int)
    return vratit

def vytvorKouli3D(voxelSize_mm,polomer_mm):
    '''voxelSize:mm = [x,y,z], polomer_mm = r
    Vytvari kouli v 3d prostoru postupnym vytvarenim
    kruznic podel X (prvni) osy. Predpokladem spravnosti
    funkce je ze Y a Z osy maji stejne rozliseni
    funkce vyuziva pythagorovu vetu'''

    print 'zahajeno vytvareni 3D objektu'

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
            print '3D objekt z 50% vytvoren'

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
        kruznice = vytvoritTFKruznici(yVoxely,rKruznice)
        koule[xR,0:konec,0:konec] = kruznice[0:konec,0:konec]

    print '3D objekt uspesne vytvoren'
    return koule

def miniSouradnice(delkaPole,delkaKoule,bod):
    '''Pomocna funkce pro nalezeni souradnic pro vlozeni objektu do pole
    jedna se o dukladne osetreni okraju '''
    delka = delkaPole
    polomer = (delkaKoule-1)/2
    uprava =  (polomer -bod)
    if(uprava <0):
        uprava = 0
    #print bod
    hraniceDolniKoule = uprava
    #print hraniceDolniKoule
    uprava2 = delka-bod
    uprava2 = polomer-uprava2
    uprava2 = 2*polomer-uprava2-1
    if(uprava2 > delkaKoule-1):
        uprava2 = delkaKoule-1
    hraniceHorniKoule = uprava2+1
    #print hraniceHorniKoule
    #print koule[hraniceDolniKoule:hraniceHorniKoule]

    rozdil = hraniceHorniKoule-hraniceDolniKoule
    start = bod -polomer
    if(start <0):
        start = 0
    konec = start + rozdil

    startP = start
    konecP = konec
    startK = hraniceDolniKoule
    konecK = hraniceHorniKoule
    return[startP,konecP,startK,konecK]

def vytvor3DobjektPole(koule,pole,bod):
    ''' pole: 3d pole T/F vstupnich dat
    koule = 3d objekt mensi nez pole
    bod = souradnice [x,y,z] bodu
    Vytvori objekt koule v nulovem poli velikosti pole v bod.'''

    #nalezeni pocatku a konce v kouli
    kouleRozmer =  np.shape(koule)
    poleRozmer = np.shape(pole)
    #print np.shape(koule)
    #print np.shape(pole)
    #print bod


    delkaKouleX = kouleRozmer[0]
    delkaPoleX = poleRozmer[0]
    [startPX,konecPX,startKX,konecKX] = miniSouradnice(delkaPoleX,delkaKouleX,bod[0])

    delkaKouleY = kouleRozmer[1]
    delkaPoleY = poleRozmer[1]
    [startPY,konecPY,startKY,konecKY] = miniSouradnice(delkaPoleY,delkaKouleY,bod[1])

    delkaKouleZ = kouleRozmer[2]
    delkaPoleZ = poleRozmer[2]
    [startPZ,konecPZ,startKZ,konecKZ] = miniSouradnice(delkaPoleZ,delkaKouleZ,bod[2])

    pomocny = np.zeros(pole.shape,dtype = np.int8)

    #print [startPX,konecPX,startKX,konecKX]
    #print [startPY,konecPY,startKY,konecKY]
    #print [startPZ,konecPZ,startKZ,konecKZ]
    pomocny[startPX:konecPX,startPY:konecPY,startPZ:konecPZ] = koule[startKX:konecKX,startKY:konecKY,startKZ:konecKZ]
    return pomocny

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
    #zobrazit2(original,rucni,strojova) #pouziva contour
    return

def zobrazitOriginal(original):
    '''Pomocna metoda pro zobrazeni
    libovolneho snimku '''
    ed = sed3.sed3(original)
    #print kombinaceNesouhlas
    ed.show()
    return

def zobrazit(original,rucni,strojova):
    '''Metoda pro srovnani rucni a
    automaticke segmentace z lidskeho pohledu
    cerna oblast - NESHODA strojoveho a rucniho
    bila oblast - SHODA strojoveho a rucniho
    seda oblast - NULY u strojoveho i rucniho'''

    prunik = np.multiply(rucni,strojova)
    opak = (strojova-1)*(-1)
    prunikOpak =  np.multiply(opak,rucni)
    vysledek = -strojova-rucni  +3*prunik
    #poleVysledek = kombinace
    ed = sed3.sed3(vysledek)
    #print kombinaceNesouhlas
    ed.show()
    return

def zobrazit2(original,rucni,strojova):
    '''Metoda pro srovnani rucni a
    automaticke segmentace z lidskeho pohledu
    cerna oblast - NESHODA strojoveho a rucniho
    bila oblast - SHODA strojoveho a rucniho
    seda oblast - NULY u strojoveho i rucniho'''


    #poleVysledek = kombinace
    ed = sed3.sed3(original,contour=strojova)
    #print kombinaceNesouhlas
    ed.show()
    return

def vyhodnoceniMetodyTri(metoda,path = None):
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
        vysledky = vyhodnoceniSnimku(rucniPole,rucniVelikost,segmentovany,segmentovanyVelikost)
        #vysledky =[1,2]

        return vysledky

    cesta = ''
    if(path == None):
        cesta = nactiYamlSoubor('path.yml')
    else:
        cesta = path


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






def vyhodnoceniSnimku(snimek1,voxelsize1,snimek2,voxelsize2):
    '''Provede vyhodnoceni snimku pomoci metod z volumetry_evaluation,
    slucuje dve metody a vraci pole [evalData (slovnik),score(%)],
    dale protoze velikosti voxelu se mirne lisi u rucni segmentace
    a originalniho obrazku (10^-2 a mene) udela z nich prumer
    '''
    print 'probiha vyhodnoceni snimku pockejte prosim'
    voxelsize_mm = [((voxelsize1[0]+voxelsize2[0])/2.0),((voxelsize1[1]+voxelsize2[1])/2.0),((voxelsize1[2]+voxelsize2[2])/2.0)]#prumer z obou
    snimek1 = np.array(snimek1,dtype = np.int8)
    snimek2 = np.array(snimek2,dtype = np.int8)
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
    konstanta = 0.7 #EXPERIMENTALNE NALEZENA KONSTANTA
    def nactiPrumVar(source):
        '''vrati pole [prumer,variance] nactene pomoci yaml ze souboru'''
        # source = 'Metoda1.yml'
        vektor=nactiYamlSoubor(source)
        prumer = vektor[0]
        variance = vektor[1]
        return [prumer,variance]

    if(vysledky == False): #v pripade nezadani vysledku
        [prumer,var] = nactiPrumVar(source)
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
        rezNovy1 = ( (np.array(rez)>=prumer+mezDolni))
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

def segmentace2(tabulka,velikostVoxelu,source='Metoda1.yml',vysledky = False):
    '''PRIMITIVNI METODA - PRAHOVANI Z PDF -50 az 250
    optimalni se zda 0 az 200
    pro metodu 0 az 180'''
    print 'pouzita metoda 2'
    segmentaceVysledek = []
    zeli3 = 0
    mezDolni = 0
    mezHorni = 250
    for rez in tabulka:
        print str(zeli3+1) + '/' + str(len(tabulka))
        rezNovy1 = ( (np.array(rez)>=mezDolni))
        rezNovy2 = (np.array(rez)<=mezHorni)
        rezNovy =np.multiply( rezNovy1, rezNovy2)
        rezNovy2 = rezNovy.astype(int)

        'BINARNI OPERACE'
        struktura1 = [[0,1,0],[1,1,1],[0,1,0]]
        struktura4 = np.ones([7,1])
        rezNovy = ndimage.binary_erosion(rezNovy2,struktura1, 5)
        rezNovy2 = ndimage.binary_fill_holes(rezNovy)
        rezNovy = ndimage.binary_erosion(rezNovy2,struktura1, 18)
        rezNovy2 = ndimage.binary_erosion(rezNovy,struktura4, 8)
        rezNOvy = rezNovy2



        segmentaceVysledek.append(rezNovy)
        zeli3 = zeli3+1 #prochazeni rezu


    #ed = sed3.sed3(np.array(segmentaceVysledek))
    #ed.show()

    #print segmentaceVysledek
    #print np.shape(tabulka)
    #print np.shape(segmentaceVysledek)
    return segmentaceVysledek


def segmentace3(tabulka,velikostVoxelu,source='Metoda1.yml',vysledky = False):
    '''Morfologie z 2D na 3D '''
    print 'pouzita metoda 3'
    segmentaceVysledek = []
    #print tabulka
    #zeli3 = 0
    pokus = nactiYamlSoubor(source)
    prumer = pokus[0]
    odchylka = np.sqrt(pokus[1])
    konstanta = 2.5 #STARA DOBRA = 2
    mezHorni=prumer + konstanta*odchylka
    mezDolni=prumer - konstanta*odchylka
    print 'probiha prahovani'

    rezNovy1 = ( tabulka>=mezDolni)
    rezNovy2 = (tabulka<=mezHorni)
    rezNovy =np.multiply( rezNovy1, rezNovy2)
    rezNovy2 = rezNovy.astype(np.int8)



    'BINARNI OPERACE 3D'
    print 'probihaji binarni operace'
    struktura1 = np.array([[[0,1,0],[1,1,1],[0,1,0]],[[1,1,1],[1,1,1],[1,1,1]],[[0,1,0],[1,1,1],[0,1,0]]])

    rezNovy = ndimage.binary_dilation(rezNovy2,struktura1, 1)
    print '1/2'
    rezNovy2 = ndimage.binary_opening(rezNovy,struktura1, 15) #CELKEM OK
    print '2/2'
    rezNovy = ndimage.binary_erosion(rezNovy2,struktura1, 10)
    rezNovy2 = rezNovy

    print 'probiha vybrani nejvetsiho objektu'
    [labelImage, labels] = ndimage.label(rezNovy2)
    #print nb_labels
    vytvoreny = np.zeros(labelImage.shape,dtype = np.int8)
    nejvetsi = 0 #index nejvetsiho objektu
    maximum = 0
    for x in range(labels):
        print str(x+1) + '/' + str(labels)
        vytvoreny = (labelImage == x+1)
        suma = np.sum(vytvoreny)
        #print suma
        if(suma > maximum):
            nejvetsi = x+1
            maximum = suma

    print x+1
    print maximum
    rezNovy2 = labelImage == nejvetsi




    rezNovy = rezNovy2.astype(np.int8)
    #print rezNovy



    'REGION GROWING'
    print 'Probiha region growing'



    #ed = sed3.sed3(np.array(rezNovy))
    #ed.show()

    return rezNovy

def trenovaniCele(metoda,path = None):
    '''Metoda je cislo INT, dane poradim metody pri implementaci prace
    nacte cestu ze souboru path.yml, vsechny soubory v adresari
     natrenuje podle zvolene metody a zapise vysledek do TrenC.yml.
    '''
    cesta = ''
    if(path == None):
        cesta = nactiYamlSoubor('path.yml')
    else:
        cesta = path
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

def trenovaniTri(metoda,path = None):
    '''Metoda je cislo INT, dane poradim metody pri implementaci prace
    nacte cestu ze souboru path.yml, vsechny soubory v adresari rozdeli na tri casti
    pro casti 1+2,2+3 a 1+3 natrenuje podle zvolene metody.
    ulozene soubory: 1) seznam trenovanych souboru 2)seznam na kterych ma probehnout segmentace
    3) vysledek trenovani (napr. prumer a odchylka u metody 1)
    '''
    cesta = ''
    if(path == None):
        cesta = nactiYamlSoubor('path.yml')
    else:
        cesta = path
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

    #ed = sed3.sed3(tabulka)
    #ed.show()

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
        voxelsize=[1, 1, 1],segparams={}
    ):
        """
        :data3d: 3D array with data
        :segparams: parameters of segmentation
        cisloMetody = INT, cislo pouzite metody (0-testovaci)
        vysledkyDostupne = F/vysledek, F =>vysledek se nacte, jinak se vezme
        path = path s trenovacimi/testovacimi daty
        :returns: TODO

        """
        # used for user interactivity evaluation
        self.data3d = data3d
        self.interactivity_counter = 0
        # 3D array with object and background selections by user
        self.seeds = None
        self.voxelSize = voxelsize
        self.segParams = {'cisloMetody':4,'vysledkyDostupne':False,'some_parameter': 22,'path':None}
        self.segParams.update(segparams)
        self.segmentation = np.zeros(data3d.shape, dtype=np.int8)
        pass

    def setCisloMetody(self,cislo):
        self.segParams['cisloMetody'] = cislo

    def getCisloMetody(self):
        return self.segParams['cisloMetody']


    def setPath(self, string):
        self.segParams['path'] = string

    def getPath(self):
        return self.segParams['path']

    def run(self):
        numero = self.segParams['cisloMetody']
        #print self.segParams
        vysledek = self.segParams['vysledkyDostupne']
        spatne = True

        if(numero == 0):
            print('testovaci metoda')
            self.segmentation = segmentace0(self.data3d,self.voxelSize,vysledek)
            spatne = False
        if(numero == 1):
            self.segmentation = segmentace1(self.data3d,self.voxelSize,
                                            vysledky=vysledek)
            spatne = False
        if(numero == 2):
            self.segmentation = segmentace2(self.data3d,self.voxelSize,
                                            vysledky=vysledek)
            spatne = False
        if(numero == 3):
            self.segmentation = segmentace3(self.data3d,self.voxelSize,
                                            vysledky=vysledek)
            spatne = False
        if(numero == 4):
            self.segmentation = segmentace4(self.data3d,self.voxelSize,
                                            vysledky=vysledek)
            spatne = False

        if(spatne):
            print('Zvolena metoda nenalezena')

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

        if(numero == 2):
            self.segmentation = segmentace2(self.data3d,self.voxelSize,vysledek)
            spatne = False
        if(numero == 3):
            self.segmentation = segmentace3(self.data3d,self.voxelSize,vysledek)
            spatne = False
        if(numero == 4):
            self.segmentation = segmentace4(self.data3d,self.voxelSize,vysledek)
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

    def vyhodnoceniMetodyTri(self):
        '''Funkce je ulozena zvlast aby bylo mozne menit pocet parametru
        a jednoduse volat defaultni metodu '''
        metoda = self.getCisloMetody()
        path = self.getPath()
        vyhodnoceniMetodyTri(metoda,path)


    def  trenovaniCele(self):
        '''Funkce je ulozena zvlast aby bylo mozne menit pocet parametru
        a jednoduse volat defaultni metodu '''
        metoda = self.getCisloMetody()
        path = self.getPath()
        trenovaniCele(metoda,path)

    def  trenovaniTri(self):
        '''Funkce je ulozena zvlast aby bylo mozne menit pocet parametru
        a jednoduse volat defaultni metodu '''
        metoda = self.getCisloMetody()
        path = self.getPath()
        trenovaniTri(metoda,path)


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
