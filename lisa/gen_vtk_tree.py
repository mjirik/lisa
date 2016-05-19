#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import vtk
import numpy as nm
import yaml
import sys
import argparse

#
# def get_cylinder(upper, height, radius,
#                  direction,
#                  resolution=10):
#
#     src = vtk.vtkCylinderSource()
#     src.SetCenter((0, height/2, 0))
#     src.SetHeight(height + radius/2.0)
#     src.SetRadius(radius)
#     src.SetResolution(resolution)
#
#     rot1 = vtk.vtkTransform()
#     fi = nm.arccos(direction[1])
#
#     rot1.RotateWXYZ(-nm.rad2deg(fi), 0.0, 0.0, 1.0)
#     u = nm.abs(nm.sin(fi))
#     rot2 = vtk.vtkTransform()
#     if u > 1.0e-6:
#
#         # sometimes d[0]/u little bit is over 1
#         d0_over_u = direction[0] / u
#         if d0_over_u > 1:
#             psi = 0
#         elif d0_over_u < -1:
#             psi = 2 * nm.pi
#         else:
#             psi = nm.arccos(direction[0] / u)
#
#         logger.debug('d0 '+str(direction[0])+'  u '+str(u)+' psi '+str(psi))
#         if direction[2] < 0:
#             psi = 2 * nm.pi - psi
#
#         rot2.RotateWXYZ(-nm.rad2deg(psi), 0.0, 1.0, 0.0)
#
#     tl = vtk.vtkTransform()
#     tl.Translate(upper)
#
#     tr1a = vtk.vtkTransformFilter()
#     tr1a.SetInput(src.GetOutput())
#     tr1a.SetTransform(rot1)
#
#     tr1b = vtk.vtkTransformFilter()
#     tr1b.SetInput(tr1a.GetOutput())
#     tr1b.SetTransform(rot2)
#
#     tr2 = vtk.vtkTransformFilter()
#     tr2.SetInput(tr1b.GetOutput())
#     tr2.SetTransform(tl)
#
#     tr2.Update()
#
#     return tr2.GetOutput()
#
#
# def gen_tree(tree_data):
#
#     points = vtk.vtkPoints()
#     polyData = vtk.vtkPolyData()
#     polyData.Allocate(1000, 1)
#     polyData.SetPoints(points)
#     poffset = 0
#
#     for br in tree_data:
#         cyl = get_cylinder(br['upperVertex'],
#                            br['length'],
#                            br['radius'],
#                            br['direction'],
#                            resolution=16)
#
#         for ii in xrange(cyl.GetNumberOfPoints()):
#             points.InsertPoint(poffset + ii, cyl.GetPoint(ii))
#
#         for ii in xrange(cyl.GetNumberOfCells()):
#             cell = cyl.GetCell(ii)
#             cellIds = cell.GetPointIds()
#             for jj in xrange(cellIds.GetNumberOfIds()):
#                 oldId = cellIds.GetId(jj)
#                 cellIds.SetId(jj, oldId + poffset)
#
#             polyData.InsertNextCell(cell.GetCellType(),
#                                     cell.GetPointIds())
#
#         poffset += cyl.GetNumberOfPoints()
#
#     return polyData
#
#
# def process_tree(indata):
#     scale = 1e-3
#     scale = 1
#
#     outdata = []
#     for key in indata:
#         ii = indata[key]
#         logger.debug(ii)
#         br = {}
#         try:
#             # old version of yaml tree
#             vA = ii['upperVertexXYZmm']
#             vB = ii['lowerVertexXYZmm']
#             radi = ii['radius']
#             lengthEstimation = ii['length']
#         except:
#             # new version of yaml
#             try:
#                 vA = ii['nodeA_ZYX_mm']
#                 vB = ii['nodeB_ZYX_mm']
#                 radi = ii['radius_mm']
#                 lengthEstimation = ii['lengthEstimation']
#             except:
#                 continue
#
#         br['upperVertex'] = nm.array(vA) * scale
#         br['radius'] = radi * scale
#         br['real_length'] = lengthEstimation * scale
#
#         vv = nm.array(vB) * scale - br['upperVertex']
#         br['direction'] = vv / nm.linalg.norm(vv)
#         br['length'] = nm.linalg.norm(vv)
#         outdata.append(br)
#
#     return outdata

def main():
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    # fh = logging.FileHandler('log.txt')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.debug('start')

    # input parser
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'inputfile',
        default=None,
        help='input file'
    )
    parser.add_argument(
        'outputfile',
        default='output.vtk',
        nargs='?',
        help='output file'
    )
    parser.add_argument(
        '-l','--label',
        default=None,
        help='text label of vessel tree. f.e. "porta" or "hepatic_veins". \
        First label is used if it is set to None'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)

    import imtools.gen_vtk_tree
    imtools.gen_vtk_tree.vt_file_2_vtk_file(args.inputfile, args.outputfile, args.label)


if __name__ == "__main__":
    main()
