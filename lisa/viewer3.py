#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
VTK Viewer pro 3D zobrazeni

program mozne spustit ve dvou rezimech View a Cut. Pokud neni zvolen zadny mod, tak
prohlizec pracuje v rezimu View

Mod Cut :
Priklady spusteni:
viewer3.py -vtk mesh_new.vtk -mode 'Cut' -slab 'liver'
viewer3.py -pkl out -mode 'Cut'
viewer3.py -pkl out -mode 'Cut' -slab 'porta'

Spusti se editor volby resekcni linie. V nem je mozno volit rez pomoci roviny a to
stisknutim tlacitka Plane. Rez se provede stisknutim tlacitka Cut. Tlacitko
Point zatim nefunkcni

Mod View :
Priklady spusteni:
viewer3.py -pkl file.pkl -mode 'View'
viewer3.py -vtk mesh_new.vtk -mode 'View' -slab 'liver'
viewer3.py -pkl vessels002.pkl -mode 'View' -deg 5


Spusti prohlizec slouzici pouze pro vizualizaci jater

'''

'''
Importování potřebných knihoven a skriptů
'''
import sys
import virtual_resection
import numpy as np
import numpy as nm
import scipy.ndimage
import argparse

from PyQt4 import QtCore, QtGui
# from PyQt4.QtCore import pyqtSignal, QObject, QRunnable, QThreadPool, Qt
from PyQt4.QtGui import QMainWindow, QGridLayout, QApplication
# from PyQt4 import *
# from PyQt4.QtGui import *
# from PyQt4.QtCore import *

import vtk
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from Tkinter import *
try:
    from dicom2fem import seg2fem
except:
    print("deprecated import of seg2fem")

    import seg2fem

import misc

'''
Používané globální proměnné
'''
plane = vtk.vtkPlane()
normal = None
coordinates = None
iren = vtk.vtkRenderWindowInteractor()
surface = vtk.vtkDataSetSurfaceFilter()
app = QApplication(sys.argv)
label = QtGui.QLabel()
myLayout = QGridLayout()
widget = vtk.vtkSphereSource()
planeWidget = vtk.vtkImplicitPlaneWidget()


'''
Kód grafického editoru resekční linie. Tento kód byl automaticky vygenerován pomocí programu Qt Designer
'''
##------------------------------------------------------------------------------------------
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        '''
        Zde se vytvoří hlavní okno editoru. Nastaví se jeho velikost a název
        '''
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1000, 800)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.widget = QtGui.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(370, 49, 401, 411))
        self.widget.setObjectName(_fromUtf8("widget"))
        '''
        Vytvoření tlačítka CUT, které při stisku spouští metodu liver_cut (zvolený resekční algoritmus)
        '''
        self.toolButton = QtGui.QPushButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(140, 140, 71, 41))
        self.toolButton.setObjectName(_fromUtf8("toolButton"))
        QtCore.QObject.connect(self.toolButton, QtCore.SIGNAL("clicked()"), MainWindow.liver_cut )
        '''
        Vytvoření tlačítka PLANE, které při stisku volá metodu Plane
        '''
        self.toolButton_2 = QtGui.QPushButton(self.centralwidget)
        self.toolButton_2.setGeometry(QtCore.QRect(140, 280, 71, 41))
        self.toolButton_2.setObjectName(_fromUtf8("toolButton_2"))
        QtCore.QObject.connect(self.toolButton_2, QtCore.SIGNAL("clicked()"), MainWindow.Plane )
        '''
        Vytvoření tlačítka POINT, které při stisku volá metodu Point
        '''
        self.toolButton_3 = QtGui.QPushButton(self.centralwidget)
        self.toolButton_3.setGeometry(QtCore.QRect(140, 210, 71, 41))
        self.toolButton_3.setObjectName(_fromUtf8("toolButton_3"))
        QtCore.QObject.connect(self.toolButton_3, QtCore.SIGNAL("clicked()"), MainWindow.Point )
        '''
        Vytvoření textového pole pro uživatelské výpisy
        '''
        self.info_text = QtGui.QPlainTextEdit(self.centralwidget)
        self.info_text.setGeometry(QtCore.QRect(20, 350, 280, 100))
        self.info_text.setObjectName(_fromUtf8("lineEdit"))
        self.info_text.setReadOnly(True)
        '''
        Vytvoření textového pole pro výpisy informací o velikosti odstraněné části jater
        '''
        self.liver_text = QtGui.QPlainTextEdit(self.centralwidget)
        self.liver_text.setGeometry(QtCore.QRect(380, 490, 380, 50))
        self.liver_text.setObjectName(_fromUtf8("lineEdit"))
        self.liver_text.setReadOnly(True)

        '''
        Vytvoření vizualizačního okna
        '''
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.gridlayout = QtGui.QGridLayout(self.widget)
        self.vtkWidget = QVTKRenderWindowInteractor(self.widget)

        self.gridlayout.addWidget(self.vtkWidget, 0, 0, 1, 1)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Nástroj pro volbu resekční linie", None))
        self.toolButton.setText(_translate("MainWindow", "CUT", None))
        self.toolButton_2.setText(_translate("MainWindow", "PLANE", None))
        self.toolButton_3.setText(_translate("MainWindow", "POINT", None))
##------------------------------------------------------------------------------------------
'''
Hlavní třída pro vytvoření prohlížeče
'''
class Viewer(QMainWindow):

    '''
    Konstruktor pro vytvoření objektu prohlížeče
    Pokud je zvolen resekční režim, nejdříve se vytvoří editor resekční linie a poté se nastaví interaktor
    '''
    def __init__(self, inputfile,mode,parent = None):
        self.ren = vtk.vtkRenderer()
        if mode == 'Cut':
            QtGui.QMainWindow.__init__(self, parent)
            self.vtk_filename = inputfile
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)
            self.info_text = self.ui.info_text
            self.liver_text = self.ui.liver_text
            self.planew = None
            self.cut_point = None
            self.iren = self.ui.vtkWidget.GetRenderWindow().GetInteractor()
            self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        '''
        Pokud je zvolen prohlížecí režim, musíme vytvořit jak renderovací okna tak samotný interaktor
        '''
        if mode == 'View':
            QMainWindow.__init__(self,parent)
            self.renWin = vtk.vtkRenderWindow()
            self.renWin.AddRenderer(self.ren)
            self.iren = vtk.vtkRenderWindowInteractor()
            self.iren.SetRenderWindow(self.renWin)

##------------------------------------------------------------------------------------------
    '''
    Tato metoda slouží pro vygenerování vtk souboru ze segmentovaných dat. Této metodě jsou tedy
    předána segmentovaná data společně s velikostí voxelu.
    '''
    def generate_mesh(self,segmentation,voxelsize_mm):
        self.segment = segmentation
        print("Generuji data...")
        self.voxelsize_mm = voxelsize_mm
        '''
        Nahrazuje původní velikost voxelu, velikostí vexoelů zvětšenou o hodnotu degradace
        '''
        self.new_vox = voxelsize_mm * self.degrad
        '''
        Zde je použita metoda ze skriptu seg2fem, která ze zadaných dat a velikosti voxelu vytvoří vtk soubor
        Data se před vytvořením vtk souboru ještě vyhlazují,aby měla hladší povrch
        Výsledný soubor je uložen pod názvem mesh_new.vtk
        '''
        mesh_data = seg2fem.gen_mesh_from_voxels_mc(self.segment, self.new_vox)
        if True:
            for x in xrange (50):
                mesh_data.coors = seg2fem.smooth_mesh(mesh_data)

        print("Done")
        vtk_file = "mesh_new.vtk"
        mesh_data.write(vtk_file)
        return vtk_file

##------------------------------------------------------------------------------------------
    '''
    Tato metoda slouží pro vytvoření virtuální roviny. K tomu je využita třída z VTK vtkImlicitPlaneWidget()
    '''
    def Plane(self):
        if(self.planew != None):
            self.info_text.appendPlainText (_fromUtf8("Nelze použít více rovin najednou. Nejdříve proveďte řez"))
        else:
            planeWidget = vtk.vtkImplicitPlaneWidget()
            # předaní interaktoru objektu roviny
            planeWidget.SetInteractor(self.iren)
            # nastavení velikosti prostoru ve kterém se může rovina pohybovat
            planeWidget.SetPlaceFactor(1.5)
            # nastavení vstupních dat
            planeWidget.SetInput(surface.GetOutput())
            planeWidget.PlaceWidget()
            planeWidget.TubingOff()
            planeWidget.OutsideBoundsOff()
            planeWidget.ScaleEnabledOff()
            planeWidget.OutlineTranslationOff()
            planeWidget.AddObserver("InteractionEvent", self.Cutter)

            planeWidget.On()
            self.planew = planeWidget
            self.planew.SetNormal(2.0,0.0,0.0)

##------------------------------------------------------------------------------------------
    '''
    Tato metoda slouží pro vytvoření virtuálního bodu. K tomu je využita třída z VTK vtkPointWidget()
    '''
    def Point(self):
            print('Point')
            self.cut_point = vtk.vtkPointWidget()
            self.cut_point.SetInput(surface.GetOutput())
            self.cut_point.AllOff()
            self.cut_point.PlaceWidget()
            # nastavení interaktoru
            self.cut_point.SetInteractor(self.iren)
            self.cut_point.On()
            point = vtk.vtkPolyData()
            self.cut_point.GetPolyData(point);

##------------------------------------------------------------------------------------------

    def Cutter(self,obj, event):
        global plane, selectActor
        obj.GetPlane(plane)

    '''
    Tato metoda obsluhuje spouštění resekčních algoritmů.
    '''
    def liver_cut(self):
        '''
        Pokud není zvolené kriterium pro provádění resekční linie, program zahlásí chybu
        '''
        if (self.planew == None) & (self.cut_point == None):
            self.info_text.appendPlainText (_fromUtf8("Neexistuje rovina řezu"))
            self.info_text.appendPlainText (_fromUtf8('Nejdříve vytvořte rovinu(Plane), nebo bod(Point)'))
        '''
        Pokud je zvoleno jako resekční kritérium rovina spustí se metoda Rez_podle_roviny ze skriptu virtual_resection
        '''
        if self.planew != None:
            self.info_text.appendPlainText (_fromUtf8("Provádění řezu. Prosím čekejte"))
            data_z_resekce,odstraneni_procenta = virtual_resection.Rez_podle_roviny(self.planew,self.segment,self.new_vox)
            '''
            Zde se provádí výpis velikosti (v procentech) odříznuté části jater do editoru pro uživatele
            Pokud je z jater odříznuto příliš mnoho, algoritmus někdy přepočítá jejich hodnotu nad hranici sta procent, proto
            je to zde ošetřeno podmínkou
            '''
            if (odstraneni_procenta > 100):
                self.liver_text.appendPlainText(_fromUtf8("Odstraněno příliš mnoho. Nelze spočítat"))
            else:
                self.liver_text.appendPlainText(_fromUtf8("Bylo ostraněno cca "+str(odstraneni_procenta)+" % jater"))
            '''
            Zde je vypnuta vytvořená rovina, což je provedeno z toho důvodu, aby mohlo být prováděno více řezů za sebou
            '''
            self.planew.Off()
        '''
        Pokud je zvoleno jako resekční kritérium bod spustí metoda. Tato metoda nejdříve vytvoří metici nul o stejné velikosti jako je matice
        původních dat. V této matici je na pozici bodu, který zvolí uživatel nula naahrazena jedničkou. Celá tato matice je společně s maticí
        původních dat předána resekčnímu algoritmu podle bodu ze skriptu virtual_resection
        '''
        if self.cut_point != None:
            self.info_text.appendPlainText (_fromUtf8("Provádění řezu. Prosím čekejte"))
            pozice = self.cut_point.GetPosition()
            self.data['data3d'] = self.data['data3d'][::self.degrad,::self.degrad,::self.degrad]
            seeds = np.zeros((self.data['segmentation'].shape[0],(self.data['segmentation'].shape[1]),(self.data['segmentation'].shape[2])))
            seeds[pozice[0]/self.new_vox[0]][pozice[1]/self.new_vox[1]][pozice[2]/self.new_vox[2]] = 1
            self.data = virtual_resection.Resekce_podle_bodu(self.data,seeds)
            data_z_resekce = self.data['segmentation'] == self.data['slab']['liver']
            self.cut_point.Off()
        '''
        Následně je vytvořen soubor VTK obsahující část jater, bez uříznuté části.
        '''
        mesh_data = seg2fem.gen_mesh_from_voxels_mc(data_z_resekce, self.new_vox)
        if True:
            for x in xrange (15):
                mesh_data.coors = seg2fem.smooth_mesh(mesh_data)
        print("Done")
        self.planew = None
        self.cut_point = None
        vtk_file = "mesh_new.vtk"
        mesh_data.write(vtk_file)
        self.View(vtk_file)


##------------------------------------------------------------------------------------------
    '''
    Tato metoda slouží pro práci s editorem a prohlížečem v jiném programu bez použítí příkazové řádky (Zatím ve zkušebním provozu, ještě není plně funkční!)
    '''
    def prohlizej(self,data, mode, slab=None):

        if slab == 'liver':
            self.degrad = 5
        else:
            self.degrad = 5
        mesh = self.generate_mesh(data['segmentation'] == data['slab'][slab],data['voxelsize_mm'])
        #if mode == 'View' or mode == None:
        #if mode == 'Cut':
        self.View(mesh)
        self.iren.Initialize()
        app.exec_()
        sys.exit(app.exec_())


        return self


##-----------------------------------------------------------------------------------------
    '''
    Tato metoda slouží pro vizualizaci dat pomocí knihoven a tříd z VTK
    '''

    def View(self,filename):

        '''
        Nastavení interaktoru pro pohyb s objektem. Třída vtkInteractorStyleTrackballCamera(), nám umožní nastavit na levé tlačítko
        myši funkce pohybu s vizualizovaným objektem a na pravé možnost zoomu
        '''
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        '''
        Nastavení readeru pro čtení vtk souboru v podobě nestrukturované mřížky
        '''
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()

        '''
        Jako filtr je použit objekt třídy vtkDataSetSurfaceFilter(), který nám z dat vyextrahuje vnější povrch
        '''
        surface.SetInput(reader.GetOutput())
        surface.Update()

        '''
        Dále použijeme třídu vtkClipPolyData(), ta nám při pohybu roviny způsobí, že za ní bude nechávat pouze obrysy spojení
        buněk bez vyplnění povrchu, tím snadno poznáme kde jsme s rovinou po objektu již přejeli a kde ne
        '''
        clipper = vtk.vtkClipPolyData()
        clipper.SetInput(surface.GetOutput())
        clipper.SetClipFunction(plane)
        clipper.GenerateClippedOutputOn()

        clipMapper = vtk.vtkPolyDataMapper()
        clipMapper.SetInput(clipper.GetOutput())

        clipActor = vtk.vtkActor()
        clipActor.SetMapper(clipMapper)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInput(surface.GetOutput())



        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOn()
        # nastavuje šířku linek ohraničující buňky
        actor.GetProperty().SetLineWidth(0.1)
        actor.GetProperty().SetRepresentationToWireframe()
        self.ren.AddActor(clipActor)
        self.ren.AddActor(actor)

        self.iren.Initialize()
        self.iren.Start()
        '''
        Třídu vtkWindowToImageFilter použijeme pro uložení vizualizace z prohlížecího režimu ve formátu tif
        '''
        try:
            w2i = vtk.vtkWindowToImageFilter()
            writer = vtk.vtkTIFFWriter()
            w2i.SetInput(self.renWin)
            w2i.Update()
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.SetFileName("image.tif")
            self.renWin.Render()
            writer.Write()
            self.renWin.Render()
            self.renWin.Finalize()
        except(AttributeError):
            print()


##------------------------------------------------------------------------------------------
def main():
    '''
    Parser slouží pro zadávání vstupních parametrů přes příkazovou řádku. Je možné zadat celkem 6 parametrů.
    K čemu jednotlivé parametry slouží, se můžeme dočíst při zadání příkazu (viewer3.py --help) do příkazové řádky
    '''

    parser = argparse.ArgumentParser(description='Simple VTK Viewer')

    parser.add_argument('-pkl','--picklefile', default=None,
                      help='Zadání vstupního zdroje dat. Soubor .pkl jsou zpravidla segmentovaná data')
    parser.add_argument('-vtk','--vtkfile', default=None,
                      help='Zadání vstupního zdroje dat. Soubor .vtk je soubor vygenerovaný programem VTK')
    parser.add_argument('-mode','--mode', default='View',
                      help='Zadání resekčního, nebo zobrazovacího režimu')
    parser.add_argument('-slab','--slab', default = 'liver',
                      help='Zde zadáváme zda chceme zobrazit játra, nebo portální žílu')
    parser.add_argument('-vs','--voxelsize_mm', default = [1,1,1],
                      type=eval,
                      help='Viewer_size')
    parser.add_argument('-deg','--degradace', default = None, type=int,
                      help='Hodnota degradace (snizeni poctu dat)')

    args = parser.parse_args()

    '''
    Pokud programu nezadáme data vyhlásí chybu, že nemá data
    '''
    if (args.picklefile or args.vtkfile) is None:
       raise IOError('No input data!')

    '''
    Zde se program větví na dvě možnosti podle toho, jaký druh zdroje dat jsme zvolili
    Pokud jsme zvolili jako zdroj dat pickle soubor (segmentovaná data) vytvoří se objekt
    prohlížeče (Viewer) a předá se mu tento zdro, společně s informací zda chceme program
    spustit v resekčním, nebo prohlížecím režimu.
    '''
    if args.picklefile:
        viewer = Viewer(args.picklefile,args.mode)
        data = misc.obj_from_file(args.picklefile, filetype = 'pickle')

        '''
        Zde programu zadáme hodnotu degradace (pokud není změněna parametrem deg)
        '''
        if (args.slab == 'porta') &  (args.degradace is None):
            viewer.degrad = 2
        if (args.slab == 'liver') &  (args.degradace is None):
            viewer.degrad = 4
        if args.degradace != None:
            viewer.degrad = args.degradace

        '''
        Data jsou zmenšována degradací v každém rozměru
        '''
        viewer.data = data
        viewer.data['segmentation'] = viewer.data['segmentation'][::viewer.degrad,::viewer.degrad,::viewer.degrad]
        viewer.data['segmentation'] = viewer.data['segmentation'][:,::-1,:]
        '''
        Pokud data neobsahují portální žílu je zahlášena chyba a program dále pracuje s celými játry
        '''
        try:
            mesh = viewer.generate_mesh(viewer.data['segmentation'] == viewer.data['slab'][args.slab],viewer.data['voxelsize_mm'])
        except KeyError:
            try:
                print('Data bohuzel neobsahuji zadany slab:', args.slab)
                print('Zobrazena budou pouze dostupna data')
                viewer.info_text.appendPlainText (_fromUtf8('Data bohužel neobsahují zadanou část jater'))
                viewer.info_text.appendPlainText (_fromUtf8('Zobrazena budou pouze dostupná data'))
                mesh = viewer.generate_mesh(viewer.data['segmentation'] == viewer.data['slab']['liver'],viewer.data['voxelsize_mm'])
                '''
                Pokud data navíc neobsahují parametr rozměr voxelu (velmi vyjímečná záležitost) jsou programu předány jednotkové
                rozměry voxelu (1,1,1)
                '''
            except KeyError:
                data['voxelsize_mm'] = np.squeeze([1,1,1])
                mesh = viewer.generate_mesh(viewer.data['segmentation'] == viewer.data['slab'][args.slab],viewer.data['voxelsize_mm'])

        '''
        Pokud je zadán mód Cut(prohlížecí režim) nejdříve zobrazíme editor, abychom mohli začít vypisovat
        informace pro uživatele
        '''
        if args.mode == 'Cut':
            viewer.show()
        viewer.View(mesh)

    '''
    Pokud je zadán jako vstupní zdroj dat soubor vtk. Je vytvořen objekt prohlížeče(Viewer) a je mu předán vstupní soubor
    společně se zvoleným módem
    '''
    if args.vtkfile:
        viewer = Viewer(args.vtkfile,args.mode)
        if args.mode == 'Cut':
            viewer.show()
        viewer.View(args.vtkfile)

    viewer.iren.Initialize()
    app.exec_()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

##------------------------------------------------------------------------------------------

