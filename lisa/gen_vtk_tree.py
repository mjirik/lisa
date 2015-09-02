#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import vtk
import numpy as nm
import yaml
import sys


def get_cylinder(upper, height, radius,
                 direction,
                 resolution=10):

    src = vtk.vtkCylinderSource()
    src.SetCenter((0, height/2, 0))
    src.SetHeight(height + radius/2.0)
    src.SetRadius(radius)
    src.SetResolution(resolution)

    rot1 = vtk.vtkTransform()
    fi = nm.arccos(direction[1])

    rot1.RotateWXYZ(-nm.rad2deg(fi), 0.0, 0.0, 1.0)
    u = nm.abs(nm.sin(fi))
    rot2 = vtk.vtkTransform()
    if u > 1.0e-6:

        # sometimes d[0]/u little bit is over 1
        d0_over_u = direction[0] / u
        if d0_over_u > 1:
            psi = 0
        elif d0_over_u < -1:
            psi = 2 * nm.pi
        else:
            psi = nm.arccos(direction[0] / u)

        logger.debug('d0 '+str(direction[0])+'  u '+str(u)+' psi '+str(psi))
        if direction[2] < 0:
            psi = 2 * nm.pi - psi

        rot2.RotateWXYZ(-nm.rad2deg(psi), 0.0, 1.0, 0.0)

    tl = vtk.vtkTransform()
    tl.Translate(upper)

    tr1a = vtk.vtkTransformFilter()
    tr1a.SetInput(src.GetOutput())
    tr1a.SetTransform(rot1)

    tr1b = vtk.vtkTransformFilter()
    tr1b.SetInput(tr1a.GetOutput())
    tr1b.SetTransform(rot2)

    tr2 = vtk.vtkTransformFilter()
    tr2.SetInput(tr1b.GetOutput())
    tr2.SetTransform(tl)

    tr2.Update()

    return tr2.GetOutput()


def gen_tree(tree_data):

    points = vtk.vtkPoints()
    polyData = vtk.vtkPolyData()
    polyData.Allocate(1000, 1)
    polyData.SetPoints(points)
    poffset = 0

    for br in tree_data:
        cyl = get_cylinder(br['upperVertex'],
                           br['length'],
                           br['radius'],
                           br['direction'],
                           resolution=16)

        for ii in xrange(cyl.GetNumberOfPoints()):
            points.InsertPoint(poffset + ii, cyl.GetPoint(ii))

        for ii in xrange(cyl.GetNumberOfCells()):
            cell = cyl.GetCell(ii)
            cellIds = cell.GetPointIds()
            for jj in xrange(cellIds.GetNumberOfIds()):
                oldId = cellIds.GetId(jj)
                cellIds.SetId(jj, oldId + poffset)

            polyData.InsertNextCell(cell.GetCellType(),
                                    cell.GetPointIds())

        poffset += cyl.GetNumberOfPoints()

    return polyData


def process_tree(indata):
    scale = 1e-3
    scale = 1

    outdata = []
    for key in indata:
        ii = indata[key]
        logger.debug(ii)
        br = {}
        try:
            # old version of yaml tree
            vA = ii['upperVertexXYZmm']
            vB = ii['lowerVertexXYZmm']
            radi = ii['radius']
            lengthEstimation = ii['length']
        except:
            # new version of yaml
            try:
                vA = ii['nodeA_ZYX_mm']
                vB = ii['nodeB_ZYX_mm']
                radi = ii['radius_mm']
                lengthEstimation = ii['lengthEstimation']
            except:
                continue

        br['upperVertex'] = nm.array(vA) * scale
        br['radius'] = radi * scale
        br['real_length'] = lengthEstimation * scale

        vv = nm.array(vB) * scale - br['upperVertex']
        br['direction'] = vv / nm.linalg.norm(vv)
        br['length'] = nm.linalg.norm(vv)
        outdata.append(br)

    return outdata


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    infile = sys.argv[1]
    if len(sys.argv) >= 3:
        outfile = sys.argv[2]

    else:
        outfile = 'output.vtk'

    yaml_file = open(infile, 'r')
    tree_raw_data = yaml.load(yaml_file)

    if 'graph' in tree_raw_data:
        trees = tree_raw_data['graph']
    else:
        trees = tree_raw_data['Graph']

    tkeys = trees.keys()

    tree_data = process_tree(trees[tkeys[0]])
    # try:
    #     trees = process_tree(tree_raw_data['graph']['porta'])
    # except:
    #     = process_tree(tree_raw_data['Graph']['porta'])
    polyData = gen_tree(tree_data)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(outfile)
    writer.SetInput(polyData)
    writer.Write()

if __name__ == "__main__":
    main()
