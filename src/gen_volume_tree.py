#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Generator of histology report

"""
import logging
logger = logging.getLogger(__name__)

import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))


import argparse
import numpy as np
import scipy.ndimage
import datawriter

import datareader
import py3DSeedEditor as se

import gen_vtk_tree

import vtk
from vtk.util import numpy_support

from datetime import datetime

class TreeVolumeGenerator:
    def __init__(self):
        self.data = None
        self.data3d = None
        self.voxelsize_mm = [1, 1, 1]
        self.shape = None

    def importFromYaml(self, filename):
        data = misc.obj_from_file(filename=filename, filetype='yaml')
        self.data = data
    
    def add_cylinder(self, cyl_id): # TODO - otestovat zda spravne zpracuje velikost voxelu
        """
        Funkce na vygenerovani 3d dat jednoho segmentu do 3D dat
        """
        cyl_data = self.data['Graph'][cyl_id]
        cyl_data3d = np.ones(self.shape, dtype=np.int)
        
        # prvni a koncovy bod, v mm
        p1 = cyl_data['nodeA_XYZ_mm']
        p2 = cyl_data['nodeB_XYZ_mm']
        # prvni a koncovy bod, ve pixelech
        p1 = [p1[0]/self.voxelsize_mm[0],p1[1]/self.voxelsize_mm[1],p1[2]/self.voxelsize_mm[2]]
        p2 = [p2[0]/self.voxelsize_mm[0],p2[1]/self.voxelsize_mm[1],p2[2]/self.voxelsize_mm[2]]
        
        # vzdalenost mezi prvnim a koncovim bodem
        pdiff = [p1[0]-p2[0],p1[1]-p2[1],p1[2]-p2[2]] 
        
        # generovani hodnot pro osu segmentu
        num_points = max(pdiff)*10 # na jeden "pixel" je 10 bodu primky
        xvalues = np.linspace(p1[0], p2[0], num_points)
        yvalues = np.linspace(p1[1], p2[1], num_points)
        zvalues = np.linspace(p1[2], p2[2], num_points)
        
        # drawing a line
        for i in range(0,len(xvalues)):
            cyl_data3d[int(xvalues[i])][int(yvalues[i])][int(zvalues[i])] = 0
            
        # drawinf a segment to data3d
        self.data3d[scipy.ndimage.distance_transform_edt(cyl_data3d,self.voxelsize_mm) <= cyl_data['radius_mm']] = 1

    def generateTree(self):
        """
        Funkce na vygenerování objemu stromu ze zadaných dat.


        """
        self.data3d = np.zeros(self.shape, dtype=np.int)
        
        for br in self.data['Graph']:
            logger.debug("CylinderId: "+str(br))
            self.add_cylinder(br)
    
    def generateTree_vtk(self):
        """
        Funkce na vygenerování objemu stromu ze zadaných dat.
        Veze pro generování pomocí VTK -> funguje špatně

        """
        #get vtkPolyData
        tree_data = gen_vtk_tree.process_tree(self.data['Graph'])
        polyData = gen_vtk_tree.gen_tree(tree_data)
        
        bounds = polyData.GetBounds()
        
        white_image = vtk.vtkImageData()
        white_image.SetSpacing(self.voxelsize_mm)
        white_image.SetDimensions(self.shape)
        white_image.SetExtent([0, self.shape[0]-1, 0, self.shape[1]-1, 0, self.shape[2]-1])
        #origin = [(bounds[0] + self.shape[0])/2, (bounds[1] + self.shape[1])/2, (bounds[2] + self.shape[2])/2]
        #white_image.SetOrigin(origin) #neni potreba?
        #white_image.SetScalarTypeToUnsignedChar()
        white_image.AllocateScalars()
        
        #fill the image with foreground voxels: (still black until stecil)
        inval = 255
        outval = 0  
        count = white_image.GetNumberOfPoints()
        for i in range(0,count):
            white_image.GetPointData().GetScalars().SetTuple1(i, inval)

        pol2stencil = vtk.vtkPolyDataToImageStencil()
        pol2stencil.SetInput(polyData)
        
        #pol2stencil.SetOutputOrigin(origin) # TOHLE S TIM DELA BORDEL
        pol2stencil.SetOutputSpacing(self.voxelsize_mm)
        pol2stencil.SetOutputWholeExtent(white_image.GetExtent())
        pol2stencil.Update()
        
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInput(white_image)
        imgstenc.SetStencil(pol2stencil.GetOutput())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(outval)
        imgstenc.Update()
        
        # VTK -> Numpy
        vtk_img_data = imgstenc.GetOutput()
        vtk_data = vtk_img_data.GetPointData().GetScalars()
        numpy_data = numpy_support.vtk_to_numpy(vtk_data)
        numpy_data = numpy_data.reshape(self.shape[0], self.shape[1], self.shape[2])
        numpy_data = numpy_data.transpose(2,1,0)
        
        self.data3d = numpy_data

    def saveToFile(self, outputfile):
        dw = datawriter.DataWriter()
        dw.Write3DData(self.data3d, outputfile)


if __name__ == "__main__":
    import misc
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    #ch = logging.StreamHandler()
    #logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(
        description='Histology analyser reporter'
    )
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        required=True,
        help='input file, yaml file'
    )
    parser.add_argument(
        '-o', '--outputfile',
        default=None,
        help='output file, .raw, .dcm, .tiff, given by extension '
    )
    parser.add_argument(
        '-vs', '--voxelsize',
        default=[1.0, 1.0, 1.0],
        type=int,
        metavar='N',
        nargs='+',
        help='size of voxel'
    )
    parser.add_argument(
        '-ds', '--datashape',
        default=[100, 100, 100],
        type=int,
        metavar='N',
        nargs='+',
        help='size of output data in pixels for each axis'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    startTime = datetime.now()
    
    hr = TreeVolumeGenerator()
    hr.importFromYaml(args.inputfile)
    hr.voxelsize_mm = args.voxelsize
    hr.shape = args.datashape
    hr.generateTree()
    
    logger.debug("TimeUsed:"+str(datetime.now()-startTime))
    
#vizualizace
    pyed = se.py3DSeedEditor(hr.data3d)
    pyed.show()
#ukládání do souboru
    if args.outputfile is not None:
        hr.saveToFile(args.outputfile)
