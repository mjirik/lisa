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
from dicom2fem import seg2fem
import misc
import viewer


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
        # mesh_data.coors = seg2fem.smooth_mesh(mesh_data)

    else:
        mesh_data = seg2fem.gen_mesh_from_voxels_mc(segmentation, voxelsize_mm * degrad * 1.0e-2)
        # mesh_data.coors +=
    vtk_file = "mesh_geom.vtk"
    mesh_data.write(vtk_file)
    QApplication(sys.argv)
    view = viewer.QVTKViewer(vtk_file)
    print ('show viewer')
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
    import io3d
    data = io3d.read(args.inputfile, dataplus_format=True)
    # args.label = np.array(eval(args.label))
    # print args.label
    # import pdb; pdb.set_trace()
    ds = np.zeros(data['segmentation'].shape, np.bool)
    for i in range(0, len(args.label)):
        ds = ds | (data['segmentation'] == args.label[i])

    showSegmentation(ds, degrad=args.degrad, voxelsize_mm=data['voxelsize_mm'])
