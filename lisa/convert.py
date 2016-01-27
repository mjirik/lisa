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
import dicom2fem
from data_manipulation import select_labels


def seg2stl(
        segmentation,
        voxelsize_mm=np.ones([3, 1]),
        degrad=4,
        labels=[1],
        smoothing=True,
        outputfile="output.stl",
        tempfile="mesh_geom.vtk",
        ):
    """
    Funkce vrací trojrozměrné porobné jako data['segmentation']
    v data['slab'] je popsáno, co která hodnota znamená
    """
    print np.unique(segmentation)
    segmentation = select_labels(segmentation, labels)

    # print 'labels: ', np.unique(data['segmentation'])
    # print np.sum(data['segmentation'] == 0)
    # print args.labels
    # for i in range(0, len(args.label)):

    segmentation = segmentation[::degrad, ::degrad, ::degrad]

    print np.unique(segmentation)

    # import pdb; pdb.set_trace()
    if smoothing:
        mesh_data = seg2fem.gen_mesh_from_voxels(segmentation, voxelsize_mm*degrad*1e-3, etype='t', mtype='s')
        mesh_data.coors = dicom2fem.seg2fem.smooth_mesh(mesh_data)
    else:
        mesh_data = dicom2fem.seg2fem.gen_mesh_from_voxels_mc(segmentation, voxelsize_mm * degrad * 1.0e-2)
        # mesh_data.coors +=
    mesh_data.write(tempfile)
    dicom2fem.vtk2stl.vtk2stl(tempfile, outputfile)
    # QApplication(sys.argv)
    # view = viewer.QVTKViewer(vtk_file)
    # view.exec_()


if __name__ == "__main__":
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(
        description='\
            convert segmentation stored in pklz file into stl\n\
            \npython convert.py -i resection.pkl -l 2 3 4 -d 4')
    parser.add_argument(
        '-i', '--inputfile',
        default='organ.pkl',
        help='input file')
    parser.add_argument(
        '-o', '--outputfile',
        default='output.stl',
        help='output file')
    parser.add_argument(
        '-t', '--tempfile',
        default='mesh_geom.vtk',
        help='temp file used in processing')
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
    ds = data['segmentation']

    if args.show:
        dsel = select_labels(ds, args.labels)
        import sed3
        ed = sed3.sed3(dsel.astype(np.double))
        ed.show()


    seg2stl(ds, labels=args.labels, degrad=args.degrad, outputfile=args.outputfile, tempfile=args.tempfile)
