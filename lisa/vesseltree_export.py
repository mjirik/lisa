#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Modul is used for skeleton binary 3D data analysis
"""

import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))

import logging
logger = logging.getLogger(__name__)
import argparse

import io3d

import traceback

import numpy as np
import scipy.ndimage

def vt2esofspy(vesseltree, outputfilename="tracer.txt"):

    import io3d.misc
    if (type(vesseltree) == str) and os.path.isfile(vesseltree):
        vt = io3d.misc.obj_from_file(vesseltree)
    else:
        vt = vesseltree
    print vt['general']
    print vt.keys()
    vtgm = vt['graph']['microstructure']
    lines = []
    vs = vt['general']['voxel_size_mm']
    sh = vt['general']['shape_px']

    lines.append("#Tracer+\n")
    lines.append("#voxelsize mm %f %f %f\n" % (vs[0], vs[1], vs[2]))
    lines.append("#shape %i %i %i\n" % (sh[0], sh[1], sh[2]))
    lines.append(str(len(vtgm) * 2)+"\n")

    i = 1
    for id in vtgm:
        # edge['']
        try:
            nda = vtgm[id]['nodeA_ZYX']
            ndb = vtgm[id]['nodeB_ZYX']
            lines.append("%i\t%i\t%i\t%i\n" % (nda[0], nda[1], nda[2], i))
            lines.append("%i\t%i\t%i\t%i\n" % (ndb[0], ndb[1], ndb[2], i))
            i += 1
        except:
            pass


    lines.append("%i\t%i\t%i\t%i" % (0, 0, 0, 0))
    lines[3] = str(i - 1) + "\n"
    with open(outputfilename, 'wt') as f:
        f.writelines(lines)


    print "urra"


if __name__ == "__main__":
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(description='Vessel tree export to esofspy')
    parser.add_argument('-o', '--output', default="tracer.txt",
                        help='output file name')
    parser.add_argument('-i', '--input', default=None,
                        help='input')
    args = parser.parse_args()

    vt2esofspy(args.input, outputfilename=args.output)
