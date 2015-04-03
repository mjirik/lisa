#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Module is used for visualization of segmentation stored in pkl file.
"""

import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))
import logging
logger = logging.getLogger(__name__)

# from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication
import argparse


import numpy as np
import seg2fem
import misc
import viewer


def export_to_lar(segmentation, degrad):


    for xBlock in xrange(imageHeight/imageDx):
        # print "Working task: " +str(startImage) + "-" + str(endImage) + " [Xblock]"
        for yBlock in xrange(imageWidth/imageDy):
            # print "Working task: " +str(startImage) + "-" + str(endImage) + " [Yblock]"
            xStart, yStart = xBlock * imageDx, yBlock * imageDy
            xEnd, yEnd = xStart+imageDx, yStart+imageDy

            image = theImage[:, xStart:xEnd, yStart:yEnd]
            nz,nx,ny = image.shape

            # Compute a quotient complex of chains with constant field
            # ------------------------------------------------------------

            chains3D_old = [];
            chains3D = None
            hasSomeOne = False
            if (calculateout != True):
                chains3D = np.zeros(nx*ny*nz, dtype=np.int32)

            zStart = startImage - beginImageStack;

            if (calculateout == True):
                chains3D_old = cch.setList(nx,ny,nz, colorIdx, image,saveTheColors)
            else:
                hasSomeOne,chains3D = cch.setListNP(nx,ny,nz, colorIdx, image,saveTheColors)

            # print "Working task: " +str(startImage) + "-" + str(endImage) + " [hasSomeOne: " + str(hasSomeOne) +"]"

            # Compute the boundary complex of the quotient cell
            # ------------------------------------------------------------
            objectBoundaryChain = None
            if (calculateout == True) and (len(chains3D_old) > 0):
                objectBoundaryChain = larBoundaryChain(bordo3,chains3D_old)

            # Save
            if (calculateout == True):
                if (objectBoundaryChain != None):
                    writeOffsetToFile( fileToWrite, np.array([zStart,xStart,yStart], dtype=int32) )
                    fileToWrite.write( bytearray( np.array(objectBoundaryChain.toarray().astype('b').flatten()) ) )
            else:
                    if (hasSomeOne != False):
                            writeOffsetToFile( fileToWrite, np.array([zStart,xStart,yStart], dtype=int32) )
                            fileToWrite.write( bytearray( np.array(chains3D, dtype=np.dtype('b')) ) )

def showSegmentation(
        segmentation,
        voxelsize_mm=np.ones([3, 1]),
        degrad=4,
        label=1,
        smoothing=True
        ):
    """
    Funkce vrací trojrozměrné porobné jako data['segmentation']
    v data['slab'] je popsáno, co která hodnota znamená
    """
    labels = []

    segmentation = segmentation[::degrad, ::degrad, ::degrad]

    # import pdb; pdb.set_trace()
    mesh_data = seg2fem.gen_mesh_from_voxels_mc(segmentation, voxelsize_mm*degrad)
    if smoothing:
        mesh_data.coors = seg2fem.smooth_mesh(mesh_data)
    else:
        mesh_data = seg2fem.gen_mesh_from_voxels_mc(segmentation, voxelsize_mm * 1.0e-2)
        # mesh_data.coors +=
    vtk_file = "mesh_geom.vtk"
    mesh_data.write(vtk_file)
    QApplication(sys.argv)
    view = viewer.QVTKViewer(vtk_file)
    view.exec_()

    return labels

if __name__ == "__main__":
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(
        description='\
            3D visualization of segmentation\n\
            \npython show_segmentation.py\n\
            \npython show_segmentation.py -i resection.pkl -l 2 3 4 -d 4')
    parser.add_argument(
        '-i', '--inputfile',
        default='organ.pkl',
        help='input file')
    parser.add_argument(
        '-d', '--degrad', type=int,
        default=4,
        help='data degradation, default 4')
    parser.add_argument(
        '-l', '--label', type=int, metavar='N', nargs='+',
        default=[1],
        help='segmentation labels, default 1')
    args = parser.parse_args()

    data = misc.obj_from_file(args.inputfile, filetype='pickle')
    # args.label = np.array(eval(args.label))
    # print args.label
    # import pdb; pdb.set_trace()
    ds = np.zeros(data['segmentation'].shape, np.bool)
    for i in range(0, len(args.label)):
        ds = ds | (data['segmentation'] == args.label[i])

    export_to_lar(ds, args.degrad)

    showSegmentation(ds, degrad=args.degrad)
