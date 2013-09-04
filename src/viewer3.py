#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple VTK Viewer.

Example:

$ viewer.py -f head.vtk
"""
from optparse import OptionParser
import sys
import vessel_cut

from PyQt4 import QtCore, QtGui
from PyQt4 import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk import *
from Tkinter import *
plane = vtk.vtkPlane()
normal = None
coordinates = None
planew = None


class normal_and_coordinates():
    
    def set_normal(self):
        return normal
        
    def set_coordinates(self):
        return coordinates


        
class QVTKViewer(QDialog):
    """
    Simple VTK Viewer.
    QVTKViewer(segmentation)
    QVTKViewer(segmentation, voxelsize_mm) # zobrazí vše, co je větší než nula
    QVTKViewer(segmentation, voxelsize_mm, slab) # umožňuje přepínat mezi více rovinami

    qv = QVTKViewer(segmentation, voxelsize_mm, slab, mode='select_plane')
    point = qv.getPlane()
    """

    
    #def __init__(self, inputdata, voxelsize_mm=None, slab=None, mode='view', callbackfcn=None):
    """
    Args:
        inputdata: 3D numpy array 
        voxelsize_mm: Array with voxel dimensions (default=None)
        slab: Dictionary with description of labels used in inputdata
        mode: 'view' or 'select_plane'
        callbackfcn: function which may affect segmentation

    """
    def __init__(self, vtk_filename):
        
        def callback(button):
            print button
            
        def Cutter(obj, event):
            global plane, selectActor
            obj.GetPlane(plane)
            
        def Plane():
            planeWidget = vtk.vtkImplicitPlaneWidget() 
            planeWidget.SetInteractor(iren) 
            planeWidget.SetPlaceFactor(1.5)
            planeWidget.SetInput(surface.GetOutput())
            planeWidget.PlaceWidget()
            planeWidget.TubingOff()
            planeWidget.OutsideBoundsOff()
            planeWidget.ScaleEnabledOff()
            planeWidget.OutlineTranslationOff()
            planeWidget.AddObserver("InteractionEvent", Cutter)
            
            planeWidget.On()
            window.setLayout(grid)

            self.planew = planeWidget
            
            window.show()
            iren.Initialize()
            renWin.Render()
            iren.Start()

        def liver_view():
            print('Zobrazuji liver')
            vessel_cut.View('liver')

        def vein_view():
            print('Zobrazuji vein')
            vessel_cut.View('porta')

        def liver_cut():
            global normal
            global coordinates
            normal = self.planew.GetNormal()
            coordinates = self.planew.GetOrigin()
            print(normal)
            print(coordinates)
        
        """
        Initiate Viwer

        Parameters
        ----------
        vtk_filename : str
            Input VTK filename
        """

        # Renderer and InteractionStyle
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(MyInteractorStyle())
        
        # VTK file
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(vtk_filename)
        reader.Update()

        # VTK surface
        surface=vtk.vtkDataSetSurfaceFilter()
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
        clipActor.GetProperty().SetColor(1,0,0)
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
        cutActor.GetProperty().SetColor((1,0,0))

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInput(clipper.GetClippedOutput())
        mapper.ScalarVisibilityOff()
        

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOn()
        #sirka linek u objektu 
        actor.GetProperty().SetLineWidth(0.1)
        actor.GetProperty().SetRepresentationToWireframe()
        ren.AddActor(clipActor)
        ren.AddActor(cutActor)
        ren.AddActor(actor)

        window = QtGui.QWidget()
        grid = QtGui.QGridLayout()


        # Button liver
        button_liver = QtGui.QPushButton()
        button_liver.setText(unicode('liver'))
        grid.addWidget(button_liver, 1, 0)
        window.connect(button_liver, QtCore.SIGNAL("clicked()"),(lambda y:lambda: callback(y) )('Stisknuto : liver'))
        button_liver.clicked.connect(liver_view)
        button_liver.show()

        # Button vein
        button_vein = QtGui.QPushButton()
        button_vein.setText(unicode('vein'))
        grid.addWidget(button_vein, 2, 0)
        window.connect(button_vein, QtCore.SIGNAL("clicked()"),(lambda y:lambda: callback(y) )('Stisknuto : vein'))
        button_vein.clicked.connect(vein_view)
        button_vein.show()
        
        # Button plane
        button_plane = QtGui.QPushButton()
        button_plane.setText(unicode('plane'))
        grid.addWidget(button_plane, 3, 0)
        window.connect(button_plane, QtCore.SIGNAL("clicked()"),(lambda y:lambda: callback(y) )('Stisknuto : plane'))
        button_plane.clicked.connect(Plane)
        button_plane.show()

        # Button cut
        button_cut = QtGui.QPushButton()
        button_cut.setText(unicode('cut'))
        grid.addWidget(button_cut, 4, 0)
        window.connect(button_cut, QtCore.SIGNAL("clicked()"),(lambda y:lambda: callback(y) )('Stisknuto : cut'))
        button_cut.clicked.connect(liver_cut)
        button_cut.show()


        window.setLayout(grid)
        window.show()
      
        # set interaction and Interaction style
        iren.Initialize()
        renWin.Render()
        iren.Start()


class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
        def __init__(self,parent=None):
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
    parser = OptionParser(description='Simple VTK Viewer')
    parser.add_option('-f','--filename', action='store',
                      dest='in_filename', default=None,
                      help=help['in_file'])
    (options, args) = parser.parse_args()

    if options.in_filename is None:
        raise IOError('No input data!')

    app = QApplication(sys.argv)

    # odkomentovat dva řádky
    #data = misc.obj_from_file(args.inputfile, filetype = 'pickle')
    #viewer = QVTKViewer(data['segmentation'], data['voxelsize_mm'], data['slab'])

    # zakomentovat jeden řádek

    viewer = QVTKViewer(options.in_filename)
    app.exec_()
    sys.exit(app.exec_())
    #print viewer.getPlane()

if __name__ == "__main__":
    main()
    
