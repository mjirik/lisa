#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple VTK Viewer.

Example:

$ viewer.py -f head.vtk
"""
from optparse import OptionParser
import sys

from PyQt4.QtGui import QApplication, QDialog, QGridLayout, QPushButton
import vtk
from vtk.qt4 import QVTKRenderWindowInteractor


class QVTKViewer(QDialog):
    """
    Simple VTK Viewer.
    """

    def initUI(self):

        grid = QGridLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        grid.addWidget(self.vtkWidget, 0, 0, 1, 1)

        btn_close = QPushButton("close", self)
        btn_close.clicked.connect(self.close)
        grid.addWidget(btn_close, 1, 0, 1, 1)

        self.setLayout(grid)
        self.setWindowTitle('VTK Viewer')
        self.show()

    def __init__(self, vtk_filename=None, vtk_data=None):
        """
        Initiate Viwer

        Parameters
        ----------
        vtk_filename : str
            Input VTK filename
        """

        QDialog.__init__(self)
        self.initUI()

        ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(ren)
        iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        if vtk_filename is not None:
            # VTK file
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(vtk_filename)
            reader.Update()
            vtkdata = reader.GetOutput()

        if vtk_data is not None:
            vtkdata = vtk_data

        # VTK surface
        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInput(vtkdata)
        surface.Update()

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInput(surface.GetOutput())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOff()
        # actor.GetProperty().SetEdgeColor(1,1,1)
        # actor.GetProperty().SetLineWidth(0.1)
        ren.AddActor(actor)

        # annot. cube
        axesActor = vtk.vtkAnnotatedCubeActor()
        axesActor.SetXPlusFaceText('R')
        axesActor.SetXMinusFaceText('L')
        axesActor.SetYMinusFaceText('H')
        axesActor.SetYPlusFaceText('F')
        axesActor.SetZMinusFaceText('A')
        axesActor.SetZPlusFaceText('P')
        axesActor.GetTextEdgesProperty().SetColor(1, 1, 0)
        axesActor.GetCubeProperty().SetColor(0, 0, 1)
        self.axes = vtk.vtkOrientationMarkerWidget()
        self.axes.SetOrientationMarker(axesActor)
        self.axes.SetInteractor(iren)
        self.axes.EnabledOn()
        self.axes.InteractiveOn()

        ren.ResetCamera()
        iren.Initialize()


usage = '%prog [options]\n' + __doc__.rstrip()
help = {
    'in_file': 'input VTK file with unstructured mesh',
}


def main():
    parser = OptionParser(description='Simple VTK Viewer')
    parser.add_option('-f', '--filename', action='store',
                      dest='in_filename', default=None,
                      help=help['in_file'])
    (options, args) = parser.parse_args()

    if options.in_filename is None:
        raise IOError('No VTK data!')

    app = QApplication(sys.argv)
    viewer = QVTKViewer(options.in_filename)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
