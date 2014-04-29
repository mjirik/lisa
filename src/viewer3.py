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
viewer3.py -pkl vessels002.pkl -mode 'View' 


Spusti prohlizec slouzici pouze pro vizualizaci jater

'''

import sys
import virtual_resection
import numpy as np
import numpy as nm
import scipy.ndimage
import argparse

from PyQt4 import QtCore, QtGui
from PyQt4 import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from Tkinter import *
import seg2fem

import misc

# pouzivane promenne
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
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1000, 800)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.widget = QtGui.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(370, 49, 401, 411))
        self.widget.setObjectName(_fromUtf8("widget"))
        
        self.toolButton = QtGui.QPushButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(140, 140, 71, 41))
        self.toolButton.setObjectName(_fromUtf8("toolButton"))
        QtCore.QObject.connect(self.toolButton, QtCore.SIGNAL("clicked()"), MainWindow.liver_cut )
        
        self.toolButton_2 = QtGui.QPushButton(self.centralwidget)
        self.toolButton_2.setGeometry(QtCore.QRect(140, 280, 71, 41))
        self.toolButton_2.setObjectName(_fromUtf8("toolButton_2"))
        QtCore.QObject.connect(self.toolButton_2, QtCore.SIGNAL("clicked()"), MainWindow.Plane )

        self.toolButton_3 = QtGui.QPushButton(self.centralwidget)
        self.toolButton_3.setGeometry(QtCore.QRect(140, 210, 71, 41))
        self.toolButton_3.setObjectName(_fromUtf8("toolButton_3"))
        QtCore.QObject.connect(self.toolButton_3, QtCore.SIGNAL("clicked()"), MainWindow.Point )

        self.info_text = QtGui.QPlainTextEdit(self.centralwidget)
        self.info_text.setGeometry(QtCore.QRect(20, 350, 280, 100))
        self.info_text.setObjectName(_fromUtf8("lineEdit"))
        self.info_text.setReadOnly(True)

        self.liver_text = QtGui.QPlainTextEdit(self.centralwidget)
        self.liver_text.setGeometry(QtCore.QRect(380, 490, 380, 50))
        self.liver_text.setObjectName(_fromUtf8("lineEdit"))
        self.liver_text.setReadOnly(True)
        
        
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


class Viewer(QMainWindow):


    '''
    QVTKViewer(segmentation)
    QVTKViewer(segmentation, voxelsize_mm) # zobrazí vše, co je větší než nula
    QVTKViewer(segmentation, voxelsize_mm, slab) # umožňuje přepínat mezi více rovinami

    qv = QVTKViewer(segmentation, voxelsize_mm, slab, mode='select_plane')
    point = qv.getPlane()

    #def __init__(self, inputdata, voxelsize_mm=None, slab=None, mode='view', callbackfcn=None):
        self.inputdata = inputdata
        self.voxelsize_mm = voxelsize_mm
        self.slab = slab
        self.mode = mode
        self.callbackfcn = callbackfcn

    #def __init__(self,segmentation,voxelsize_mm):
        self.segmentation = segmentation
        self.voxelsize_mm = voxelsize_mm
    '''
##------------------------------------------------------------------------------------------

        
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
            self.oriznuti_jater = 0
            self.iren = self.ui.vtkWidget.GetRenderWindow().GetInteractor()
            self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
            
        if mode == 'View':
            QMainWindow.__init__(self,parent)
            self.renWin = vtk.vtkRenderWindow()
            self.renWin.AddRenderer(self.ren)
            self.iren = vtk.vtkRenderWindowInteractor()
            self.iren.SetRenderWindow(self.renWin)
            '''
            self.setGeometry(QtCore.QRect(500, 500, 500, 500))
            
            self.vtkWidget = QVTKRenderWindowInteractor(self)
            self.gridlayout = QtGui.QGridLayout(self.vtkWidget)
            self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
            self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
            self.gridlayout.addWidget(self, 0, 0, 1, 1)
            '''
        '''
        # Create source
        source = vtk.vtkSphereSource()
        widget = source
        source.SetCenter(0, 0, 0)
        source.SetRadius(5.0)
 
        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
 
        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
 
        self.ren.AddActor(actor)
        '''

##------------------------------------------------------------------------------------------
    def set_normal(self,normal):
        self.normal = normal

    def set_coordinates(self,coordinates):
        self.coordinates = coordinates
##------------------------------------------------------------------------------------------
    def generate_mesh(self,segmentation,voxelsize_mm,degrad):
        segmentation = segmentation[::degrad,::degrad,::degrad]
        segmentation = segmentation[:,::-1,:]
        self.segment = segmentation
        print self.segment.shape
        print 'Voxelsize_mm'
        print voxelsize_mm
        print("Generuji data...")
        self.voxelsize_mm = voxelsize_mm
        self.new_vox = voxelsize_mm * degrad
        print 'self_new'
        print self.new_vox
        mesh_data = seg2fem.gen_mesh_from_voxels_mc(self.segment, self.new_vox)
        print 'Voxer'
        print voxelsize_mm

        if True:
            for x in xrange (50):
                mesh_data.coors = seg2fem.smooth_mesh(mesh_data)

        print("Done")
        vtk_file = "mesh_new.vtk"
        mesh_data.write(vtk_file)
        return vtk_file
##------------------------------------------------------------------------------------------
    '''
    Args:
        inputdata: 3D numpy array
        voxelsize_mm: Array with voxel dimensions (default=None)
        slab: Dictionary with description of labels used in inputdata
        mode: 'view' or 'select_plane'
        callbackfcn: function which may affect segmentation

    '''
##------------------------------------------------------------------------------------------
    def Plane(self):
        if(self.planew != None):
            self.info_text.appendPlainText (_fromUtf8("Nelze použít více rovin najednou. Nejdříve proveďte řez"))
        else:
            planeWidget = vtk.vtkImplicitPlaneWidget()
            planeWidget.SetInteractor(self.iren)
            planeWidget.SetPlaceFactor(1.5)
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
            print self.planew.GetNormal()

        #window.show()
        #self.iren.Initialize()
        #renWin.Render()
        #iren.Start()
##------------------------------------------------------------------------------------------

    def Point(self):
            print 'Point'
            self.cut_point = vtk.vtkPointWidget()
            self.cut_point.SetInput(surface.GetOutput())
            self.cut_point.AllOff()
            self.cut_point.PlaceWidget()
            self.cut_point.SetInteractor(self.iren)
            self.cut_point.On()
            print (self.cut_point.GetPosition())
            point = vtk.vtkPolyData()
            self.cut_point.GetPolyData(point);
        #window.show()
            self.iren.Initialize()
        #renWin.Render()
            self.iren.Start()
##------------------------------------------------------------------------------------------

    def callback(self,button):
        print button

    def Cutter(self,obj, event):
        global plane, selectActor
        obj.GetPlane(plane)

    def liver_view(self):
        print('Zobrazuji liver')
        vessel_cut.View('liver')

    def vein_view(self):
        print('Zobrazuji vein')
        vessel_cut.View('porta')

    def liver_cut(self):
        global normal
        global coordinates
        self.info_text.appendPlainText (_fromUtf8("Provádění řezu. Prosím čekejte"))
        if self.planew != None:
            try:
                self.set_normal(self.planew.GetNormal())
                self.set_coordinates(self.planew.GetOrigin())
                self.cutter()
                self.Rezani()
                #print(self.normal)
                #print(self.coordinates)
            except AttributeError:
                self.info_text.appendPlainText (_fromUtf8("Neexistuje rovina řezu"))
                self.info_text.appendPlainText (_fromUtf8('Nejdříve vytvořte rovinu stisknutím tlačítka Plane'))
                print('Neexistuje rovina rezu')
                print('Nejdrive vytvorte rovinu stisknutim tlacitka Plane')
        if self.cut_point != None:
            print('Souradnice bodu')
            print (self.cut_point.GetPosition())
            pozice = self.cut_point.GetPosition()
            self.data['segmentation'] = self.data['segmentation'][::self.degrad,::self.degrad,::self.degrad]
            self.data['data3d'] = self.data['data3d'][::self.degrad,::self.degrad,::self.degrad]
            #self.data['voxelsize_mm'] = self.voxelsize_mm
            seeds = np.zeros((self.data['segmentation'].shape[0],(self.data['segmentation'].shape[1]),(self.data['segmentation'].shape[2])))
            seeds[pozice[0]/self.new_vox[0]][pozice[1]/self.new_vox[1]][pozice[2]/self.new_vox[2]] = 1
            print 'Seedu'
            print seeds.shape
            
            self.data = virtual_resection.cut_for_3D_Viewer(self.data,seeds)
            self.data['segmentation'] = self.data['segmentation'][::self.degrad,::self.degrad,::self.degrad]
            print (self.data['segmentation'] == self.data['slab']['liver']).shape
            mesh_data = seg2fem.gen_mesh_from_voxels_mc(self.data['segmentation'] == self.data['slab']['liver'], self.new_vox)
            if True:
                for x in xrange (15):
                    mesh_data.coors = seg2fem.smooth_mesh(mesh_data)
            print("Done")
            vtk_file = "mesh_new.vtk"
            mesh_data.write(vtk_file)
            self.cut_point.Off()
            #self.cut_point = None
            self.View(vtk_file)

##------------------------------------------------------------------------------------------
    def Rez(self,a,b,c,d):
        mensi = 0
        vetsi = 0
        mensi_objekt = 0
        vetsi_objekt = 0
        print 'x: ',a,' y: ',b,' z: ',c
        print('Pocitani rezu...')
        data = self.segment
        prava_strana = np.ones((data.shape[0],data.shape[1],data.shape[2]))
        leva_strana = np.ones((data.shape[0],data.shape[1],data.shape[2]))
        dimension = data.shape
        for x in range(dimension[0]):
            for y in range(dimension[1]):
                for z in range(dimension[2]):
                    rovnice = a*x + b*y + c*z + d
                    #print self.data['segmentation'][x][y][z]
                    #pdb.set_trace()
                    if((rovnice) <= 0):
                        mensi = mensi+1
                        if(data[x][y][z] == 1):
                            mensi_objekt = mensi_objekt+1
                        prava_strana[x][y][z] = 0
                    else:
                        vetsi = vetsi+1
                        if(data[x][y][z] == 1):
                            vetsi_objekt = vetsi_objekt+1
                        leva_strana[x][y][z] = 0
                        #self.data['segmentation'][x][y][z] = False
        prava_strana = prava_strana * data
        objekt = mensi_objekt + vetsi_objekt
        procenta = ((100*mensi_objekt)/objekt)
        self.oriznuti_jater += procenta
        if (self.oriznuti_jater > 100):
            self.liver_text.appendPlainText(_fromUtf8("Odstraněno příliš mnoho. Nelze spočítat"))
        else:
            self.liver_text.appendPlainText(_fromUtf8("Bylo ostraněno cca "+str(self.oriznuti_jater)+" % jater"))
            
        print 'Mensi: ',mensi
        print 'Vetsi: ',vetsi
        print 'Mensi_objekt: ',mensi_objekt
        print 'Vetsi_objekt: ',vetsi_objekt
        print("Generuji data...")
        mesh_data = seg2fem.gen_mesh_from_voxels_mc(prava_strana, self.new_vox)
        
        if True:
            for x in xrange (15):
                mesh_data.coors = seg2fem.smooth_mesh(mesh_data)
        print("Done")
        vtk_file = "mesh_new.vtk"
        mesh_data.write(vtk_file)
        self.planew.Off()
        self.planew = None
        self.View(vtk_file)
                
##------------------------------------------------------------------------------------------
    def Rezani(self):
        # vzorec roviny ax + by + cz + d = 0;

        a = self.normal[0]*self.new_vox[0]
        b = self.normal[1]*self.new_vox[1]
        c = self.normal[2]*self.new_vox[2]
        xx = self.coordinates[0]/self.new_vox[0]
        yy = self.coordinates[1]/self.new_vox[1]
        zz = self.coordinates[2]/self.new_vox[2]

        '''
        a = self.normal[0]
        b = self.normal[1]
        c = self.normal[2]
        xx = self.coordinates[0]
        yy = self.coordinates[1]
        zz = self.coordinates[2]

        '''

        
        d = -(a*xx)-(b*yy)-(c*zz)
        print d
        self.Rez(a,b,c,d)

        '''
        print('Generuji rez')
        
        with open('mesh_new.vtk') as f:
            for line in f:
                if(pocet >=2):
                    break
                try:
                    bod = map(float, line.split())
                    if(bod) == []:
                        pocet = pocet+1
                        continue
                    self.Rovina(bod[0],bod[1],bod[2],d)
                except(ValueError):
                    continue
        '''

        
##------------------------------------------------------------------------------------------
    def prohlizej(self,data, mode, slab=None):
        self.iren.Initialize()
        self.show()
        
        if slab == 'liver':
            degrad = 4
        else:
            degrad = 2
        mesh = self.generate_mesh(data['segmentation'] == data['slab'][slab],data['voxelsize_mm'],degrad)
        #if mode == 'View' or mode == None:
        #if mode == 'Cut':
        self.View(mesh)
        self.iren.Initialize()
        app.exec_()
        sys.exit(app.exec_())


        return self

##------------------------------------------------------------------------------------------
    def cutter(self):
        print 'Normal: '
        print self.normal
        print 'Coordinates: '
        print self.coordinates


##-----------------------------------------------------------------------------------------
    def View(self,filename):

        # Nastaveni interaktoru pro pohyb s objektem
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # VTK file
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()

        # VTK surface
        
        surface.SetInput(reader.GetOutput())
        surface.Update()

        # Cutter
        clipper = vtk.vtkClipPolyData()
        clipper.SetInput(surface.GetOutput())
        clipper.SetClipFunction(plane)
        #clipper.GenerateClipScalarsOn()
        clipper.GenerateClippedOutputOn()

        clipMapper = vtk.vtkPolyDataMapper()
        clipMapper.SetInput(clipper.GetOutput())

        clipActor = vtk.vtkActor()
        clipActor.SetMapper(clipMapper)

        '''
    
        cutEdges = vtk.vtkCutter()
        cutEdges.SetInput(surface.GetOutput())
        cutEdges.SetCutFunction(plane)
        cutEdges.GenerateCutScalarsOn()

        cutStrips = vtk.vtkStripper()
        cutStrips.SetInput(cutEdges.GetOutput())
        cutStrips.Update()

        cutPoly = vtk.vtkPolyData()
        cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
        cutPoly.SetPolys(cutStrips.GetOutput().GetLines())

        cutTriangles = vtk.vtkTriangleFilter()
        cutTriangles.SetInput(cutPoly)

        cutMapper = vtk.vtkPolyDataMapper()
        cutMapper.SetInput(cutTriangles.GetOutput())

        cutActor = vtk.vtkActor()
        cutActor.SetMapper(cutMapper)
        cutActor.GetProperty().SetColor(1.0,0.0,0.0)

        '''
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInput(surface.GetOutput())
        mapper.ScalarVisibilityOff()
        


        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOn()
        # nastavi barvu linek UnstructuredGrid
        actor.GetProperty().SetColor(0.0,0.0,1.0)
        #sirka linek u objektu
        actor.GetProperty().SetLineWidth(0.1)
        actor.GetProperty().SetRepresentationToWireframe()
        self.ren.AddActor(clipActor)
        #ren.AddActor(cutActor)
        self.ren.AddActor(actor)

        self.iren.Initialize()
        self.iren.Start()
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

    parser = argparse.ArgumentParser(description='Simple VTK Viewer')

    parser.add_argument('-pkl','--picklefile', default=None,
                      help='File as .pkl')
    parser.add_argument('-vtk','--vtkfile', default=None,
                      help='File as .vtk')
    parser.add_argument('-mode','--mode', default='View',
                      help='Mode for construction plane of resection')
    parser.add_argument('-slab','--slab', default = 'liver',
                      help='liver or porta - view')
    parser.add_argument('-vs','--voxelsize_mm', default = [1,1,1],
                      type=eval,
                      help='Viewer_size')
    args = parser.parse_args()

    if (args.picklefile or args.vtkfile) is None:
       raise IOError('No input data!')


    # vytvoreni okna

    viewer = Viewer(args.picklefile,args.mode)
    if args.picklefile:
        if args.slab == 'porta':
            viewer.degrad = 2
        else:
            viewer.degrad = 5
        data = misc.obj_from_file(args.picklefile, filetype = 'pickle')
        viewer.data = data
        #np.squeeze([1,1,1])

        try:
            mesh = viewer.generate_mesh(data['segmentation'] == data['slab'][args.slab],data['voxelsize_mm'],viewer.degrad)        
        except KeyError:
            try:
                print 'Data bohuzel neobsahuji zadany slab:', args.slab
                print 'Zobrazena budou pouze dostupna data'
                #degrad = 5
                mesh = viewer.generate_mesh(data['segmentation'] == data['slab']['liver'],data['voxelsize_mm'],viewer.degrad)
                viewer.info_text.appendPlainText (_fromUtf8('Data bohužel neobsahují zadanou část jater'))
                viewer.info_text.appendPlainText (_fromUtf8('Zobrazena budou pouze dostupná data'))
                
            except KeyError:
                data['voxelsize_mm'] = np.squeeze([1,1,1])
                mesh = viewer.generate_mesh(data['segmentation'] == data['slab'][args.slab],data['voxelsize_mm'],viewer.degrad)

        if args.mode == 'View' or args.mode == None:
            viewer.View(mesh)
        if args.mode == 'Cut':
            viewer.show()
            viewer.View(mesh)

    if args.vtkfile:
        if args.mode == 'View' or args.mode == None:
            viewer.View(args.vtkfile);
        if args.mode == 'Cut':
            viewer.show()
            viewer.View(args.vtkfile)


    '''
    window = QtGui.QWidget()
    grid = QtGui.QGridLayout()
    window.setWindowTitle("3D liver")
    window.setLayout(grid)
    
    window.setWindowTitle("3D liver")
    window.setLayout(grid)
    
    viewer = Viewer(args.picklefile)
    accept = False
    #print args.voxelsize_mm
    if args.picklefile:
        data = misc.obj_from_file(args.picklefile, filetype = 'pickle')
        print np.where(data['segmentation'] == 1)
        #print "unique ", np.unique(data['segmentation'])
        #print 'voxel' , data['voxelsize_mm']
        #data['voxelsize_mm'] = [1,1,1]
        viewer.data = data
        #np.squeeze([1,1,1])
        print data['segmentation']
        print (data['segmentation'] == data['slab'][args.slab]).shape
        try:
            mesh = viewer.generate_mesh(data['segmentation'] == data['slab'][args.slab],data['voxelsize_mm'])
        except KeyError:
            print 'Data bohuzel neobsahuji zadany slab:', args.slab
            print 'Zobrazena budou pouze dostupna data'
            mesh = viewer.generate_mesh(data['segmentation'] == data['slab']['liver'],np.squeeze([1,1,1]))
        print data['slab']
        if args.mode == 'View' or args.mode == None:
            accept = True
            viewer.View(mesh,accept)
        if args.mode == 'Cut':
            #viewer = QVTKViewer(data['segmentation'], data['voxelsize_mm'], data['slab'])
            accept = False
            viewer.Set_voxel_size()
            viewer.View(mesh,accept)
            viewer.buttons(window,grid)
            #viewer.Rez()

    if args.vtkfile:
        if args.mode == 'View' or args.mode == None:
            accept = True
            viewer.View(args.vtkfile,accept);
        if args.mode == 'Cut':
            accept = False
            #viewer = QVTKViewer(data['segmentation'], data['voxelsize_mm'], data['slab'])
            viewer.Set_voxel_size()
            viewer.View(args.vtkfile,accept)
            viewer.buttons(window,grid)
            #viewer.Rezani();
'''
    viewer.iren.Initialize()
    app.exec_()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

##------------------------------------------------------------------------------------------

