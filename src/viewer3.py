#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
VTK Viewer pro 3D zobrazeni

program mozne spustit ve dvou rezimech View a Cut
Vstupní soubor může být soubor pickle nebo už vygenerovaný vtk
Je mozne zobrazovat cela jatra, nebo hlavni portalni zilu prikazy : liver, porta
Priklady :

viewer3.py -pkl file.pkl

viewer3.py -vtk mesh_new.vtk -mode 'View' -slab 'liver'


viewer3.py -pkl vessels002.pkl -mode 'Cut'  -slab 'porta'
'''

from optparse import OptionParser
import sys
import vessel_cut
import numpy as np
import numpy as nm
import scipy.ndimage
import argparse

from PyQt4 import QtCore, QtGui
from PyQt4 import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk import *
from Tkinter import *
import seg2fem

import misc
nimport py3DSeedEditor
import show3
import qmisc
import pdb

# pouzivane promenne
plane = vtk.vtkPlane()
normal = None
coordinates = None
planew = None
iren = vtk.vtkRenderWindowInteractor()
renWin = vtk.vtkRenderWindow()
surface = vtk.vtkDataSetSurfaceFilter()
app = QApplication(sys.argv)
label = QtGui.QLabel()
myLayout = QGridLayout()



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
        MainWindow.resize(800, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.widget = QtGui.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(370, 49, 401, 441))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.toolButton = QtGui.QToolButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(140, 140, 71, 41))
        self.toolButton.setObjectName(_fromUtf8("toolButton"))
        self.toolButton_2 = QtGui.QToolButton(self.centralwidget)
        self.toolButton_2.setGeometry(QtCore.QRect(140, 210, 71, 41))
        self.toolButton_2.setObjectName(_fromUtf8("toolButton_2"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.toolButton.setText(_translate("MainWindow", "CUT", None))
        self.toolButton_2.setText(_translate("MainWindow", "PLANE", None))


class Viewer(object):


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

        
    def __init__(self, inputfile):
        self.vtk_filename = inputfile

        pass

##------------------------------------------------------------------------------------------
    def set_normal(self,normal):
        self.normal = normal

    def set_coordinates(self,coordinates):
        self.coordinates = coordinates
##------------------------------------------------------------------------------------------
    def generate_mesh(self,segmentation,voxelsize_mm,degrad = 5):
        segmentation = segmentation[::degrad,::degrad,::degrad]
        segmentation = segmentation[:,::-1,:]
        self.segment = segmentation
        print self.segment.shape
        print 'Voxelsize_mm'
        print voxelsize_mm
        print("Generuji data...")
        self.new_vox = voxelsize_mm * degrad
        print 'self_new'
        print self.new_vox
        mesh_data = seg2fem.gen_mesh_from_voxels_mc(self.segment, self.new_vox)
        print 'Voxer'
        print voxelsize_mm

        if True:
            for x in xrange (100):
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
        planeWidget = vtk.vtkImplicitPlaneWidget()
        planeWidget.SetInteractor(iren)
        planeWidget.SetPlaceFactor(1.5)
        planeWidget.SetInput(surface.GetOutput())
        planeWidget.PlaceWidget()
        planeWidget.TubingOff()
        planeWidget.OutsideBoundsOff()
        planeWidget.ScaleEnabledOff()
        planeWidget.OutlineTranslationOff()
        planeWidget.AddObserver("InteractionEvent", self.Cutter)

        planeWidget.On()
        #window.setLayout(grid)

        self.planew = planeWidget

        #window.show()
        #iren.Initialize()
        #renWin.Render()
        #iren.Start()
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
        # pokud bylo stisknuto tlacitko Cut pred Plane, vypise chybu
        try:
            self.set_normal(self.planew.GetNormal())
            self.set_coordinates(self.planew.GetOrigin())
            self.cutter()
            self.Rezani()
            #print(self.normal)
            #print(self.coordinates)
        except AttributeError:
            print('Neexistuje rovina rezu')
            print('Nejdrive vytvorte rovinu stisknutim tlacitka Plane')

    '''
    pripravena funkce pro nastavovani voxelu v grafickem okne
    - pravdepodobne nebude vyuzita :-/
    '''
    def Set_voxel_size(self):
        '''
        rozmer_x = QtGui.QInputDialog()
        rozmer_y = QtGui.QInputDialog()
        rozmer_z= QtGui.QInputDialog()
        #.rozmer_x.getInteger(window,'Voxel_size', 'Rozmer voxelu v x:', QLineEdit.Normal)
        #rozmer_y.getInteger(window, 'Voxel_size', 'Rozmer voxelu v y:', QLineEdit.Normal)
        #rozmer_z.getInteger(window, 'Voxel_size', 'Rozmer voxelu v z:', QLineEdit.Normal)
        okno.setWindowTitle("Voxel_size")
        layout = QtGui.QGridLayout()
        okno.setLayout(layout)
        rozmer_x.setOptions(QInputDialog.NoButtons)
        rozmer_y.setOptions(QInputDialog.NoButtons)
        rozmer_z.setOptions(QInputDialog.NoButtons)
        ok_button = QtGui.QPushButton()
        ok_button.setText('Ok')
        layout.addWidget(rozmer_x,0,2)
        layout.addWidget(rozmer_y,1,2)
        layout.addWidget(rozmer_z,2,2)
        layout.addWidget(ok_button,3,2)
        okno.connect(ok_button, QtCore.SIGNAL("clicked()"),(lambda y:lambda: self.close() )('Stisknuto : ok'))
        okno.show()
        renWin.Render()
        iren.Start()
        renWin.Finalize()

        #promen_z = QInputDialog()
        #winter = QtCore.QStringList()
        #winter = 'December, January, February'
        #print winter
        #promen_z.setComboBoxItems(winter)
        print(rozmer_x)
        print(rozmer_y)
        print(rozmer_z)

        '''
##------------------------------------------------------------------------------------------
    def Rez(self,a,b,c,d):
        mensi = 0
        vetsi = 0
        mensi_objekt = 0
        vetsi_objekt = 0
        print 'x: ',a,' y: ',b,' z: ',c
        print('Pocitani rezu...')
        data = self.segment
        print 'dimension'
        print data.shape
        dimension = data.shape
        for x in range(dimension[0]):
            for y in range(dimension[1]):
                for z in range(dimension[2]):
                    rovnice = a*x + b*y + c*z + d
                    #print self.data['segmentation'][x][y][z]
                    #pdb.set_trace()
                    if(rovnice < 0):
                        mensi = mensi+1
                        if(data[x][y][z] == 1):
                            mensi_objekt = mensi_objekt+1
                        data[x][y][z] = 0
                    else:
                        vetsi = vetsi+1
                        if(data[x][y][z] == 1):
                            vetsi_objekt = vetsi_objekt+1
                        #self.data['segmentation'][x][y][z] = False

                            
        print 'Mensi: ',mensi
        print 'Vetsi: ',vetsi
        print 'Mensi_objekt: ',mensi_objekt
        print 'Vetsi_objekt: ',vetsi_objekt
        print("Generuji data...")
        mesh_data = seg2fem.gen_mesh_from_voxels_mc(data, self.new_vox)
        
        if True:
            for x in xrange (100):
                mesh_data.coors = seg2fem.smooth_mesh(mesh_data)
        print("Done")
        vtk_file = "mesh_new.vtk"
        mesh_data.write(vtk_file)
        self.View(vtk_file,True)
                
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
        xx = self.coordinates[0]/self.new_vox[0]
        yy = self.coordinates[1]/self.new_vox[1]
        zz = self.coordinates[2]/self.new_vox[2]
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
        window = QtGui.QWidget()
        grid = QtGui.QGridLayout()
        window.setWindowTitle("3D liver")
        window.setLayout(grid)
        mesh = self.generate_mesh(data['segmentation'] == data['slab'][slab],data['voxelsize_mm'])
        if mode == 'View' or mode == None:
            accept = True
            self.View(mesh,accept)
        if mode == 'Cut':
            accept = False
            self.View(mesh,accept)
            self.buttons(window,grid)


        return self

##------------------------------------------------------------------------------------------
    def cutter(self):
        print 'Normal: '
        print self.normal
        print 'Coordinates: '
        print self.coordinates
##------------------------------------------------------------------------------------------
    def buttons(self,window,grid):
        '''
        window.resize(80,55)
        layout = QtGui.QVBoxLayout()
        buttons = QtGui.QDialogButtonBox(window)
        buttons.setGeometry(QtCore.QRect(0, 0, 100, 100))
        buttons.setOrientation(QtCore.Qt.Vertical)
        buttons.setStandardButtons(QtGui.QDialogButtonBox.Close|QtGui.QDialogButtonBox.Ok)
        buttons.setObjectName(("buttonBox"))
        grid.addWidget(buttons)
        '''
        '''
        # Button liver
        button_liver = QtGui.QPushButton()
        button_liver.setText(unicode('liver'))
        grid.addWidget(button_liver, 1, 0)
        window.connect(button_liver, QtCore.SIGNAL("clicked()"),(lambda y:lambda: self.callback(y) )('Stisknuto : liver'))
        #button_liver.clicked.connect(self.liver_view)
        button_liver.show()

        # Button vein
        button_vein = QtGui.QPushButton()
        button_vein.setText(unicode('vein'))
        grid.addWidget(button_vein, 2, 0)
        window.connect(button_vein, QtCore.SIGNAL("clicked()"),(lambda y:lambda: self.callback(y) )('Stisknuto : vein'))
        #button_vein.clicked.connect(self.vein_view)
        button_vein.show()
        '''

        # Button plane
        button_plane = QtGui.QPushButton()
        button_plane.setText(unicode('plane'))
        grid.addWidget(button_plane, 3, 0)
        window.connect(button_plane, QtCore.SIGNAL("clicked()"),(lambda y:lambda: self.callback(y) )('Stisknuto : plane'))
        button_plane.clicked.connect(self.Plane)
        button_plane.show()

        # Button cut
        button_cut = QtGui.QPushButton()
        button_cut.setText(unicode('cut'))
        grid.addWidget(button_cut, 4, 0)
        button_cut.clicked.connect(self.liver_cut)
        button_cut.show()

        #iren.Initialize()
        window.show()
        renWin.Render()
        iren.Start()
        # vypina okno View
        renWin.Finalize()


##-----------------------------------------------------------------------------------------
    def View(self,filename,accept):

        # Renderer and InteractionStyle
        ren = vtk.vtkRenderer()
        renWin.AddRenderer(ren)

        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(MyInteractorStyle())

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

        backProp = vtk.vtkProperty()

        clipActor = vtk.vtkActor()
        clipActor.SetMapper(clipMapper)
        clipActor.GetProperty().SetColor(1.0,0.0,0.0)
        clipActor.SetBackfaceProperty(backProp)

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
        ren.AddActor(clipActor)
        #ren.AddActor(cutActor)
        ren.AddActor(actor)

        # pri rezani se nezobrazi okno protoze iren se inicializuje pouze v buttons, nutno
        # ho inicializovat i tady
        if accept:
            iren.Initialize()
            renWin.Render()
            iren.Start()
            renWin.Finalize()

##------------------------------------------------------------------------------------------


class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
        def __init__(self,parent=None):
            self.parent = parent
            self.AddObserver("LeftButtonPressEvent",self.LeftButtonPressEvent)
            self.AddObserver("RightButtonPressEvent",self.RightButtonPressEvent)

        def LeftButtonPressEvent(self,obj,event):
            print "Left Button pressed"
            self.Rotate()
            self.OnLeftButtonDown()

        def RightButtonPressEvent(self,obj,event):
            print "Right Button pressed"
            self.Pan()
            self.OnRightButtonDown()

        e = '%prog [options]\n' + __doc__.rstrip()
help = {
    'in_file': 'input pkl file',
}


def main():

    parser = argparse.ArgumentParser(description=__doc__) # 'Simple VTK Viewer')

    parser.add_argument('-pkl','--picklefile', default=None,
                      help='File as .pkl')
    parser.add_argument('-vtk','--vtkfile', default=None,
                      help='File as .vtk')
    parser.add_argument('-mode','--mode', default=None,
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

    viewer = Ui_MainWindow()
    viw = QtGui.QMainWindow()
    viewer.setupUi(viw)

    viw.show()
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


    app.exec_()
    sys.exit(app.exec_())
    #print viewer.getPlane()

if __name__ == "__main__":
    main()

##------------------------------------------------------------------------------------------

