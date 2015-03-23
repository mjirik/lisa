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
from sklearn import mixture
import morphsnakes
import sys

def vytvor3DMrizku(vzorkovani_mm,rozmer_mm):
    '''
    vytvori 3d mrizku se zvolenym vzorkovanim a rozmerem
     a vrati objekt:    [mrizka,vzorkovani_mm,stred]
    stred = int souradnice stredu ve vsech smerech
    '''
    pocet = np.round(rozmer_mm/vzorkovani_mm)
    if(pocet% 2 == 0):
        pocet = pocet+1
    mrizka = np.zeros([pocet,pocet,pocet])
    stred = pocet/2+1
    objekt = [mrizka,vzorkovani_mm,stred]
    return objekt

def vypoctiMrizku(data3d,voxelSize,mrizkavzk_mm = 20,mrizka_mm=250):
    ''' vypocte mrizku umistenou ve stredu objektu v data3d
    a umisti do ni data podle tvaru objektu 
    ZATIM SUMA VOXELU
    data3d - vstupni data (T/F)
    voxelSize - velikost voxelu [x,y,z]
    mrizkavzk_mm - velikost vzorkovani mrizky v milimetrech
    mrizka_mm - velikost mrizky v milimetrech (x,y i z)'''
    [mrizka,vzorkovaniMrizka,stredMrizka] = vytvor3DMrizku(50,250)
    krychlePocet = mrizka.shape
    
    krychlePocet2 = list(range(0,int(krychlePocet[0])))
    krychlePocet3 = list(range(0,krychlePocet[0]))
    krychlePocet = list(range(0,int(krychlePocet[0])))
    #print krychlePocet
    stredKrychle_mm =vzorkovaniMrizka*stredMrizka

    data = data3d
    stredPresne = ndimage.center_of_mass(data3d)
    stredData = (np.round(stredPresne))
    stredData_mm =  np.multiply(stredData,voxelSize)
    dataVelikost = data.shape
    #print stredData
    #print mrizka.shape    
    xPridat = np.round(vzorkovaniMrizka/voxelSize[0])#voxely krychlicky v rozmeru x
    yPridat = np.round(vzorkovaniMrizka/voxelSize[1])
    zPridat = np.round(vzorkovaniMrizka/voxelSize[2])
    
    for xMrizka in krychlePocet:        
        for yMrizka in krychlePocet2:
            for zMrizka in krychlePocet3:
                #print [xMrizka,yMrizka,zMrizka]
                souradniceAbsolut = np.array([xMrizka,yMrizka,zMrizka])*vzorkovaniMrizka #stred = [000]
                souradniceRelativ = souradniceAbsolut - stredKrychle_mm #vzdalenosti od stredu v mm
                voxelyX = np.round(souradniceRelativ[0]/voxelSize[0])
                voxelyY = np.round(souradniceRelativ[1]/voxelSize[1])
                voxelyZ = np.round(souradniceRelativ[2]/voxelSize[2])
                poziceOdStredu = [voxelyX,voxelyY,voxelyZ] # ve voxelech
                poziceVPoli = stredData + poziceOdStredu #zacatek
                
                #print poziceVPoli
                
                xStart = poziceVPoli[0]#osetreni okraje - nizke cislo
                if(xStart <0):
                    continue
                xKonec = xStart + xPridat
                if(xKonec > dataVelikost[0]):#osetreni okraje - vysoke cislo
                    continue
                
                yStart = poziceVPoli[1]#osetreni okraje - nizke cislo
                if(yStart <0):
                    continue
                yKonec = yStart + yPridat
                if(yKonec > dataVelikost[1]):#osetreni okraje - vysoke cislo
                    continue
                
                zStart = poziceVPoli[2]#osetreni okraje - nizke cislo
                if(zStart <0):
                    continue
                zKonec = zStart + zPridat
                if(zKonec > dataVelikost[2]):#osetreni okraje - vysoke cislo
                    continue
                
                objekt = data[xStart:xKonec,yStart:yKonec,zStart:zKonec]   
                
                'algoritmus naplneni mrizky'
                #print np.sum(objekt)
                mrizka[xMrizka,yMrizka,zMrizka] = np.sum(objekt)
    return mrizka

def simpleSnake(data3d,segmentace,iterace,vyhlazovani=1,l1=1,l2=2):
    '''Metoda pouzivajici knihovnu morphSnakes (morphological Chan-Vese evolution)
    data3d - 3d CT snimek
    segmentace - vnitrek jater pro zahajeni algoritmu
    iterace - zvoleny pocet iteraci
    vyhlazovani - pocet operaci vyhlazovani po dokonceni algoritmu
    l1,l2 =  relativni dulezitost vnitrnich a vnejsich pixelu
    (treba experimentalne nastavit)'''
    print 'vytvareni instance morphsnakes'
    macwe = morphsnakes.MorphACWE(data3d, smoothing=vyhlazovani, lambda1=l1, lambda2=l2)
    macwe.set_levelset(segmentace)
    print 'probiha beh morphsnakes'
    macwe.run(iterace)
    vysledek = macwe.levelset
    vysledek.astype(np.int8)
    return vysledek

def binarniOperace3D2D(pole3d,voxelSize,rKoule = 2.5,rKruznice = 2):
    '''Kombinace 3D a 2D binarnich operaci, vraci 3d pole true/false rozmeru pole3d
    pouzite po variabilnim prahovani
    struktury 2D: kruznice o polomeru rKruznice (2), 3D: koule o polomeru rKoule (2.5)
    operace:
    1) 3D dilatace 1x (zaplneni struktury jater)
    2) 2D otevreni 10x, zaplneni der
    3) 2D konvoluce vybrana >=5 20x po sobe
    4) 3D otevreni 5x 
    Vysledkem v kombinaci s variabilnim prahovanim (prahovaniProcenta) je relativne 
    dobre urcena oblast jater, ale je vzdy MENSI, testovano na 20ti snimcich sliver07    
    '''

    poleNew = [voxelSize[0]/1.0,voxelSize[1],voxelSize[2]]
    struktura1 = vytvorKouli3D(poleNew, rKoule)    
    rezNovy = ndimage.binary_dilation(pole3d,struktura1, 1)
    voxelyKruznice = rKruznice/voxelSize[1]
    voxelyPole = np.ceil(voxelyKruznice)
    utvar1 = vytvoritTFKruznici(voxelyPole ,voxelyKruznice)
    sumaObjektu = np.sum(utvar1)
    pomocny = 0    
    
    for rez in rezNovy:
        rez2 = ndimage.binary_opening(rez, utvar1, 10)
        konvoluce = ndimage.binary_fill_holes(rez2)
        for x in range(20):
            konvoluce = np.array(konvoluce,dtype = np.int8)
            konvoluce = ndimage.convolve(konvoluce, utvar1)
            konvoluce = (konvoluce >=sumaObjektu)
                 
        #bonus = ndimage.binary_dilation(konvoluce, utvar1, 10)
        vysledek = konvoluce
        rezNovy[pomocny,:,:] = vysledek
        pomocny = pomocny+1   
    velikost = np.shape(pole3d)
    velikost = velikost[0]
    hodnota = 5
    
    rezNovy2 = ndimage.binary_opening(rezNovy,struktura1,hodnota)
    
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

    data3d = labelImage == nejvetsi
    return data3d

def binarniOperaceNove(pole3d,voxelsize):
    '''Kombinace 3D a 2D binarnich operaci, vraci 3d pole true/false rozmeru pole3d
    pouzite po variabilnim prahovani
    struktury 2D: diamond 3x3, 3D: koule o polomeru 3 'krychlove' voxely
    operace:
    1) odstraneni okraje (sobel, jeho odecteni od puvodniho)
    2) konvoluce s 3d kouli, vybrana kde > 100 (koule = 123)
    3) 2D otevreni diamondem 5x
    4) vybrani nejvetsiho objektu
    Cilem techto operaci je odstranit sum a prevytekle casti po Region Growingu
    '''
    def odstranOkraj(objekt):
        dataNew = objekt
        dataNew.astype(np.int8)
        okraj = ndimage.sobel(dataNew)
        silny = okraj.astype(np.bool)
        silny = silny*(-1) +1
        novy = np.multiply(silny,pole3d)
        return novy   
    
    print 'probihaji 3D binarni operace'
    koule = vytvorKouli3D( [1,1,1] ,3)
    #print np.sum(koule)
    
    
    okraj = pole3d
    for x in range(1): #4
        okraj = odstranOkraj(okraj)
    
    konvoluce = ndimage.convolve(okraj,koule)
    okraj = konvoluce > 100
    
    utvar1 = np.array([[0,1,0],[1,1,1],[0,1,0]]) 
    rezNovy = okraj
    pomocny = 0
    for rez in rezNovy:
        konvoluce = ndimage.binary_fill_holes(rez)                 
        bonus = ndimage.binary_opening(konvoluce, utvar1, 5)
        vysledek = bonus
        rezNovy[pomocny,:,:] = vysledek
        pomocny = pomocny+1
    okraj = rezNovy
    
    print 'probiha vybrani nejvetsiho objektu'
    [labelImage, labels] = ndimage.label(okraj)
    #print nb_labels
    vytvoreny = np.zeros(labelImage.shape,dtype = np.int8)
    nejvetsi = 0 #index nejvetsiho objektu
    maximum = 0
    for x in range(labels):
        print str(x+1) + '/' + str(labels)
        vytvoreny = (labelImage == x+1)
        suma = np.sum(vytvoreny)
        if(suma > maximum):
            nejvetsi = x+1
            maximum = suma
    okraj = labelImage == nejvetsi
    
    
    
    return okraj


def prahovaniProcenta(data3d,procentaHranice = 0.32,procentaJatra = 0.18):
    '''Procentualni metoda prahovani
    data3d - prahovany obraz CT,
    procentaHranice - procenta ohraniceni dat pro model gaussovek
    procentaJatra - procenta podilu finalniho vysledku
    vraci prahovany obraz
    vysledek obsahuje jatra, pricemz obsahuji cerne casti - sum
    (jatra nejsou vzdy vyplnena)
    POPIS METODY
    Nejprve se vypocita histogram. Ten se pak zprava ohranici mnozstvim
    procentaHranice (35%)
    nasledne se data z ohraniceneho histogramu (po pretvoreni) vlozi do
    modelu gaussovskych funkci (dvou). Vybrana je funkce s vyssi vahou (obvykle 
    z dvojice 0.9 a 0.5). Stredni hodnota teto funkce je vybrana jako dolni hranice.
    Horni hranice je pak urcena tak aby vysledne prahovani pokrylo procenta obrazku:
    procentaJatra (15%)
    Zduvodnen - jatra jsou "hrbolek" na pravem "ramenu" nejpravejsi gaussovky.
    
    '''
    procenta = procentaHranice
    procentaJater = procentaJatra
    
    'vypocet histogramu'
    maxx = np.max(data3d)
    minx = np.min(data3d)
    bins=np.arange(minx, maxx+1)
    histogram = np.histogram(data3d, bins)
    cetnosti = histogram[0]
    celkem = np.sum(cetnosti)
    
    'ohraniceni histogramu'
    delka = len(cetnosti)
    suma = 0
    for x in range(delka):
        y = delka-x-1
        suma = suma+cetnosti[y]
        podil = float(suma)/float(celkem)
        #print podil
        if(podil >= procenta):
            break
    
    #OHRANICENY HISTOGRAM
    ukazat1 = bins[y:len(bins)-2] #hodnoty bins[0:len(bins)-1]
    ukazat2 = cetnosti[y:len(cetnosti)-1] #cetnosti    
    
    #vytvoreni dat pro gaussovsky model (cetnosti-> vic cisel)
    data = np.zeros(np.sum(ukazat2))
    prochazet = 0
    for x in range(len(ukazat2)-1):
        mnozstvi = ukazat2[x]
        konec = prochazet + mnozstvi
        tamto = ukazat1[x]
        data[prochazet:konec] = tamto
        prochazet = konec
        #print x
    sample_MAX = 8000 #omezeni na 8000 vzorku pro gaussovky
    n =  (float(len(data))/float(sample_MAX))
    data = data[0::n] 
    
    'gaussovsky model'
    clf = mixture.GMM(n_components=2, covariance_type='full')
    print 'probiha modelovani normalnimi funkcemi'
    clf.fit(data)
    print 'modelovani dokonceno'
    stredniHodnoty = clf.means_ #[m1,m2]
    vahy = clf.weights_ #[w1,w2]
    #c1, c2 = clf.covars_ #kovariance, netreba
    
    vybrat = np.argmax(vahy)
    hraniceDolni = stredniHodnoty[vybrat][0]
    
    #plt.plot(ukazat1,ukazat2)
    #plt.show()   
    
    'ohraniceni po nalezeni dolni hranice'   
    kopie = np.zeros(len(ukazat1)) 
    kopie[:] = hraniceDolni
    vzdalenost = np.abs(kopie - ukazat1) #nalezeni pozice nejblizsi hodnoty
    poziceDolni = np.argmin(vzdalenost) 
    #print ukazat2[poziceDolni] #cetnosti
    suma = 0
    delka = len(ukazat1)
    
    prochazet = np.arange(poziceDolni,delka)

    for x in prochazet:
        suma = suma+ukazat2[x]
        podil = float(suma)/float(celkem)
        if(podil >= procentaJater):
            break
    hraniceHorni = ukazat1[x]
    
    'vysledne prahovani'
    #print hraniceDolni
    #print hraniceHorni    
    prumer = np.mean(data3d)
    vetsi = data3d> hraniceDolni
    mensi = data3d < hraniceHorni
    segmentaceVysledek = np.multiply(vetsi,mensi)
    #zobrazitOriginal(segmentovany)    
    
    return segmentaceVysledek

def updatujSegparams(seznamNazvuPolozek):
    '''nacte stary slovnik z segparams1.yml, a prida, pripadne prepise
    stare nazvy a polzky novymi ze seznamu seznamNazvuPolozek:
    seznamNazvuPolozek = [['hranicniKonstanta',50.0],
    ['maxSeedKonstanta',30],
    ['prumer', 130.79240506015375],
    ['variance',  2114.51201903427]]
    '''
    path_to_script = op.dirname(os.path.abspath(__file__))
    paramfile = op.join(path_to_script, 'data/segparams1.yml')
    #print paramfile
    puvodni = nactiYamlSoubor(paramfile)
    if(isinstance(puvodni,dict)):
        updatovany = puvodni
    else:
        updatovany = {} #osetreni pripadu spatnych dat v souboru
    #print updatovany
    
    for dvojice in seznamNazvuPolozek:
        nazev = dvojice[0]
        polozka = dvojice[1]
        updatovany[nazev] = polozka
    
    zapisYamlSoubor(paramfile, updatovany)

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

    vektory = np.where(border == 1) #pole s informaci o vektorech (jejich pozice v souradnicich) DRIVE array
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

def souborAsegmentace(cisloSouboru,metoda,cesta):
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
    vytvoreny.setMethodNumber(metoda)#ZVOLENI METODY
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
    
    #zobrazitOriginal(rucni)#kontrola
    zobrazit(rucni,strojova)
    
    #skore = vyhodnoceniSnimku(rucni, rucniVelikost, strojova, segmentovanyVelikost)
    #print skore
    
    #zobrazit2(original,rucni,strojova) #pouziva contour
    return

def zobrazitOriginal(original):
    '''Pomocna metoda pro zobrazeni
    libovolneho snimku '''
    ed = sed3.sed3(original)
    #print kombinaceNesouhlas
    ed.show()
    return

def zobrazit(rucni,strojova):
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
            #print seznamTestovaci
            vysledek = vyhodnotSoubor(cesta,x,seznamTestovaci,ctenar,vysledky,metoda)
            vysledek = 0
            seznamVsechVysledku.append(vysledek)
        return seznamVsechVysledku

    def vyhodnotSoubor(cesta,x,seznamTestovaci,ctenar,vysledkySeg,metoda):
        originalNazev =  [seznamTestovaci[x]]
        rucniNazev = [seznamTestovaci[x+len(seznamTestovaci)/2]]
        print 'probiha nacitani souboru'
        vektor = nactiSoubor(cesta,rucniNazev,0,ctenar)
        rucniPole = vektor[0]
        rucniVelikost = vektor[1]
        vektor2 = nactiSoubor(cesta,originalNazev,0,ctenar)
        originalPole = vektor2[0]
        originalVelikost = vektor2[1]
        print'probiha nastavovani parametru'
        vytvoreny = LiverSegmentation(originalPole,originalVelikost)
        vytvoreny.setVysledky(vysledkySeg)
        vytvoreny.setMethodNumber(metoda) #ZVOLENI METODY
        print 'probiha segmentace'
        vytvoreny.runVolby()
        segmentovany = vytvoreny.segmentation
        segmentovanyVelikost = vytvoreny.voxelSize
        print 'zahajeno vyhodnoceni'
        rucniPole.astype(np.int8)
        segmentovany.astype(np.int8)
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

def segPlaceholder(data3d,velikostVoxelu,source,vysledky = False):
    '''RYCHLA TESTOVACI METODA - PRO TESTOVANI
    Pouzivejte v pripade testovani programu'''
    velikost = np.shape(data3d)
    a = velikost[0]
    b = velikost[1]
    c = velikost[2]
    segmentaceVysledek = np.zeros(data3d.shape,dtype = np.int8)
    segmentaceVysledek[a/4:3*a/4,b/4:3*b/4,c/4:3*c/4] = 1

    return segmentaceVysledek

def segFind(data3d,velikostVoxelu,source,vysledky = False):
    '''Metoda ktera nalzene vnitrek jater - odhadne zakladni obrysy tvaru
    testovano uspesne na 20ti snimcich sliver07
    data3d - vstupni pole CT snimku
    velikostVoxelu - [x,y,z] rozměry voxelu
    source - soubor obsahujici data segmentaci (segparams1.yml)
     '''
    prahovany = prahovaniProcenta(data3d,procentaHranice = 0.32,procentaJatra = 0.18) #0.35 0.18
    operovany = binarniOperace3D2D(prahovany,velikostVoxelu)
    #zobrazitOriginal(operovany)
    #sys.exit()   
    
    segmentaceVysledek = operovany
    return segmentaceVysledek

def segFindImproved(data3d,velikostVoxelu,source,vysledky = False):
    prahovany = prahovaniProcenta(data3d,procentaHranice = 0.3,procentaJatra = 0.1) #0.32 0.18
    utvar = vytvorKouli3D(velikostVoxelu, 2)
    utvar2 = vytvorKouli3D(velikostVoxelu, 3)
    utvar = utvar.astype(np.int8)
    prahovany = prahovany.astype(np.int8)
    soucet = np.sum(utvar)
    konvoluce = ndimage.convolve(prahovany, utvar)
    zobrazeny = konvoluce >= soucet*0.4
    vyriznuty = ndimage.binary_opening(zobrazeny, utvar, 1)
    zesileny = ndimage.binary_dilation(vyriznuty, utvar, 3)
    otevreny = ndimage.binary_opening(zesileny, utvar2, 10)
    
    vysledek = otevreny
    #main.zobrazitOriginal(vysledek)
    return vysledek

def segRGrow(data3d,velikostVoxelu,source,vysledky = False):
    '''
    Metoda pouzivajici segFind nasledovany region growingem.
    data3d - vstupni data (CT snimek)
    velikostVoxelu - vektor urcujici velikostVoxelu
    source - cesta k souboru opsahujici segmentacni parametry
    vysledky - existence trenovacich dat, jejich predani
    binarni operace nasledovane region growingem
    vzorkovaciKonstanta === pocitac urci sam
    hrannicni konstanta = ohraniceni region growingu kolem seedu v mm
    maxSeedKonstanta = pocet vybranych seedu (rovnomerne)'''

    #print 'pouzita metoda 4'
    vzorkovaciKonstanta = np.shape(data3d)[0]*0.33+40 #*0.33+33 #KONSTANTA VYPOCITANA Z DELKY 3D SNIMKU (ROZLISENI)
    
    slovnik = nactiYamlSoubor(source)
    hranicniKonstanta = slovnik['hranicniKonstanta']
    maxSeedKonstanta  = slovnik['maxSeedKonstanta']
    #hranicniKonstanta = 50
    #maxSeedKonstanta  = 30
    
    #hranicniKonstanta = 50.0 #50 a 10 vypada dobre 35 a 50seed =malo/ moc ///45 20 nej
    #maxSeedKonstanta = 30
    #print vzorkovaciKonstanta
    #vzorkovaciKonstanta = 80
    #return
    binarniOperace = segFind(data3d,velikostVoxelu,source=source,vysledky = False)
    #segmentaceVysledek = binarniOperace
    'konstanta urcuje rozmezi region growingu'
    segmentaceVysledek = regionGrowingCTIF(data3d,binarniOperace,velikostVoxelu,konstanta = vzorkovaciKonstanta,
                                           konstantaHranice = hranicniKonstanta,maxSeeds = maxSeedKonstanta)
    np.save('jatraPred.npy',segmentaceVysledek)
    ##zobrazitOriginal(segmentaceVysledek)
    #sys.exit()
    segmentaceVysledek = binarniOperaceNove(segmentaceVysledek,velikostVoxelu)
    



    return segmentaceVysledek



def segSimpleSnake(data3d,velikostVoxelu,source,vysledky = False):
    '''Metoda pouzivajici morphSNakes bez mapy 
    pouziva take segfind
     '''
    slovnik = nactiYamlSoubor(source)
    lambda2 = slovnik['snakeLambda1']
    lambda1  = slovnik['snakeLambda2']
    iterace = slovnik['snakeIterace']
    segmentace = segFind(data3d,velikostVoxelu,source)
    #zobrazitOriginal(segmentace)
    #print [iterace,lambda1,lambda2]
    #sys.exit()
    segmentaceVysledek = simpleSnake(data3d, segmentace, iterace,l1 = lambda1,l2=lambda2)
    return segmentaceVysledek

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
    #soubor = open("TrenC.yml","wb")
    zapisYamlSoubor("TrenC.yml",vysledek1)
    print "trenovani  dokonceno"

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
        voxelsize_mm=[1, 1, 1],segparams={}
    ):
        """
        :data3d: 3D array with data
        :segparams: parameters of segmentation
        method = nazev pouzite metody, seznam = getMethodList
        vysledkyDostupne = F/vysledek, F =>vysledek se nacte, jinak se vezme
        path = path s trenovacimi/testovacimi daty
        :returns: TODO

        """
        # used for user interactivity evaluation
        self.data3d = data3d
        self.interactivity_counter = 0
        # 3D array with object and background selections by user
        self.seeds = None
        self.voxelSize = voxelsize_mm
        self.segParams = {
            'method': 'find',
            'vysledkyDostupne': False,
            'paramfile': self._get_default_paramfile_path(),
            'path': None
        }
        self.segParams.update(segparams)
        self.segmentation = np.zeros(data3d.shape, dtype=np.int8)

    def _get_default_paramfile_path(self):
        path_to_script = op.dirname(os.path.abspath(__file__))
        paramfile = op.join(path_to_script, 'data/segparams1.yml')
        return paramfile

    def setMethod(self, nazev):
        self.segParams['method'] = nazev
        
    def setMethodNumber(self, n):
        '''Vybere n-tou polozku ze seznamu getMethodList a nastavi metodu na ni.'''
        seznam = self.getMethodList()
        self.segParams['method'] = seznam[n]

    def getMethod(self):
        return self.segParams['method']
    
    def getMethodList(self):
        '''Vraci seznam vsech platnych nazvu metod ktere lze pouzit'''
        return ['placeholder','find','regionGrowing','snakeSimple']
    
    def setVysledky(self,vysledky):
        self.segParams['vysledky'] = vysledky

    def set_seeds(self, seeds):
        self.seeds = seeds
        pass

    def setPath(self, string):
        self.segParams['path'] = string

    def getPath(self):
        return self.segParams['path']

    def run(self):
        '''metoda s vice moznostmi vyberu metody-vybrana v segParams'''
        nazev = self.segParams['method']
        #print self.segParams
        vysledek = self.segParams['vysledkyDostupne']
        spatne = True

        if(nazev  == 'placeholder'):
            metoda = segPlaceholder
            spatne = False
        if(nazev  == 'find'):
            metoda = segFindImproved
            spatne = False
        if(nazev  == 'regionGrowing'):
            metoda = segRGrow
            spatne = False     
        if(nazev  == 'snakeSimple'):
            metoda = segSimpleSnake
            spatne = False       

        if(spatne):
            print('Zvolena metoda nenalezena')
        else:
            self.segmentation = metoda(self.data3d,self.voxelSize,source=self.segParams['paramfile'],vysledky=vysledek)

    def runVolby(self):
        '''metoda s vice moznostmi vyberu metody-vybrana v segParams'''
        nazev = self.segParams['method']
        #print self.segParams
        vysledek = self.segParams['vysledkyDostupne']
        spatne = True

        if(nazev  == 'placeholder'):
            metoda = segPlaceholder
            spatne = False
        if(nazev  == 'find'):
            metoda = segFindImproved
            spatne = False
        if(nazev  == 'regionGrowing'):
            metoda = segRGrow
            spatne = False
        if(nazev  == 'snakeSimple'):
            metoda = segSimpleSnake
            spatne = False           

        if(spatne):
            print('Zvolena metoda nenalezena')
        else:
            self.segmentation = metoda(self.data3d,self.voxelSize,source=self.segParams['paramfile'],vysledky=vysledek)



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
        metoda = self.getMethod()
        path = self.getPath()
        vyhodnoceniMetodyTri(metoda,path)


    def  trenovaniCele(self):
        '''Funkce je ulozena zvlast aby bylo mozne menit pocet parametru
        a jednoduse volat defaultni metodu '''
        metoda = self.getMethod()
        path = self.getPath()
        trenovaniCele(metoda,path)

    def  trenovaniTri(self):
        '''Funkce je ulozena zvlast aby bylo mozne menit pocet parametru
        a jednoduse volat defaultni metodu '''
        metoda = self.getMethod()
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
