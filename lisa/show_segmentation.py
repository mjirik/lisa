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
try:
    import dicom2fem
    import dicom2fem.seg2fem
    # from dicom2fem import seg2fem
    from dicom2fem.seg2fem import gen_mesh_from_voxels_mc, smooth_mesh
except:

    print('dicom2fem not found')
    # logger.warning('dicom2fem not found')
    from seg2mesh import gen_mesh_from_voxels, smooth_mesh


def showSegmentation(
        segmentation,
        voxelsize_mm=np.ones([3, 1]),
        degrad=4,
        label=1,
        smoothing=True,
        vtk_file=None,
        qt_app=None,
        show=True,
        resize_mm=None
        ):
    """
    Funkce vrací trojrozměrné porobné jako data['segmentation']
    v data['slab'] je popsáno, co která hodnota znamená
    """

    if vtk_file is None:
        vtk_file = "mesh_geom.vtk"
    vtk_file = os.path.expanduser(vtk_file)

    labels = []

    segmentation = segmentation[::degrad, ::degrad, ::degrad]
    voxelsize_mm = voxelsize_mm * degrad

    _stats(segmentation)
    if resize_mm is not None:
        logger.debug("resize begin")
        print("resize")
        new_voxelsize_mm = np.asarray([resize_mm, resize_mm, resize_mm])
        import imtools
        segmentation = imtools.misc.resize_to_mm(segmentation, voxelsize_mm=voxelsize_mm, new_voxelsize_mm=new_voxelsize_mm)
        voxelsize_mm = new_voxelsize_mm
        logger.debug("resize begin")
    _stats(segmentation)

    # import pdb; pdb.set_trace()
    mesh_data = gen_mesh_from_voxels_mc(segmentation, voxelsize_mm)
    if smoothing:
        mesh_data.coors = smooth_mesh(mesh_data)
        # mesh_data.coors = seg2fem.smooth_mesh(mesh_data)

    else:
        mesh_data = gen_mesh_from_voxels_mc(segmentation, voxelsize_mm * 1.0e-2)
        # mesh_data.coors +=
    mesh_data.write(vtk_file)
    if qt_app is None:
        qt_app = QApplication(sys.argv)
        logger.debug("qapp constructed")
    if show:
        import viewer
        view = viewer.QVTKViewer(vtk_file)
        print('show viewer')
        view.exec_()

    return labels

def _stats(data):
    print("stats")
    un = np.unique(data)
    for lab in un:
        print(lab, " : ", np.sum(data==lab))

def main():
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
        '-o', '--outputfile',
        default='~/lisa_data/mesh_geom.vtk',
        help='output file')
    parser.add_argument(
        '-d', '--degrad', type=int,
        default=4,
        help='data degradation, default 4')
    parser.add_argument(
        '-r', '--resize', type=float,
        default=None,
        help='resize voxel to defined size in milimeters, default is None')
    parser.add_argument(
        '-l', '--label', type=int, metavar='N', nargs='+',
        default=[1],
        help='segmentation labels, default 1')
    args = parser.parse_args()

    # data = misc.obj_from_file(args.inputfile, filetype='pickle')
    import io3d
    data = io3d.read(args.inputfile, dataplus_format=True)
    # args.label = np.array(eval(args.label))
    # print args.label
    # import pdb; pdb.set_trace()
    ds = np.zeros(data['segmentation'].shape, np.bool)
    for i in range(0, len(args.label)):
        ds = ds | (data['segmentation'] == args.label[i])

    outputfile = os.path.expanduser(args.outputfile)

    showSegmentation(ds, degrad=args.degrad, voxelsize_mm=data['voxelsize_mm'], vtk_file=outputfile, resize_mm=args.resize)

if __name__ == "__main__":
    main()
