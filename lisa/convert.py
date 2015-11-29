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

import vtk

import numpy as np
from dicom2fem import seg2fem
# import misc
import viewer
import io3d

def vtk2stl(fn_in, fn_out, mesh_data):

    reader = vtk.vtkDataSetReader()
    reader.SetFileName(fn_in)
    reader.Update()

    gfilter = vtk.vtkGeometryFilter()
    # if vtk.VTK_MAJOR_VERSION <= 5:
    #     gfilter.SetInput(reader.GetOutput())
    # else:
    #     import pdb; pdb.set_trace()
    #     gfilter.SetInputConnection(reader.GetOutputPort())
    # gfilter.SetInputConnection(reader.GetOutputPort())
    import pdb; pdb.set_trace()
    gfilter.SetInputConnection(mesh_data)

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(fn_out)
    writer.SetInput(gfilter.GetOutput())
    writer.Write()

def convert_segmentation(
        segmentation,
        voxelsize_mm=np.ones([3, 1]),
        degrad=4,
        label=1,
        smoothing=False
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
    # from dicom2fem import vtk2stl
    vtk2stl(vtk_file, "mesh.stl", mesh_data)
    # QApplication(sys.argv)
    # view = viewer.QVTKViewer(vtk_file)
    # view.exec_()

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
        '-l', '--labels', type=int, metavar='N', nargs='+',
        default=[1],
        help='segmentation labels, default 1')
    parser.add_argument(
        '-s', '--show', action='store_true',
        help='Show mode')
    args = parser.parse_args()

    dr = io3d.DataReader()
    data = dr.Get3DData(args.inputfile, dataplus_format=True)
    # args.label = np.array(eval(args.label))
    # print args.label
    # import pdb; pdb.set_trace()
    ds = np.zeros(data['segmentation'].shape, np.bool)
    print 'labels: ', np.unique(data['segmentation'])
    print np.sum(data['segmentation'] == 0)
    print args.labels
    # for i in range(0, len(args.label)):
    print args.labels[0] + 1
    for lab in args.labels:
        print "print zpracovavam ", lab
        dadd = (data['segmentation'] == lab)
        print np.sum(dadd)

        ds = ds | dadd

    print np.unique(ds)
    if args.show:
        import sed3
        ed = sed3.sed3(ds.astype(np.double))
        ed.show()


    convert_segmentation(ds, degrad=args.degrad)
