
#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/"))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/py3DSeedEditor/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)


import datareader
import misc
import qmisc
import py3DSeedEditor

def show():
    reader = datareader.DataReader()
    data3d_a_path = os.path.join(path_to_script, '../../../data/medical/data_orig/sliver07/training-part1/liver-orig001.mhd')
    data3d_a, metadata_a = reader.Get3DData(data3d_a_path)

    data3d_b_path = os.path.join(path_to_script, '../../../data/medical/data_orig/sliver07/training-part1/liver-seg001.mhd')
    data3d_b, metadata_b = reader.Get3DData(data3d_b_path)

    #data3d_b_path = os.path.join(inputdata['basedir'], inputdata['data'][i]['ourseg'])
    #obj_b = misc.obj_from_file(data3d_b_path, filetype='pickle')
    #data_b, metadata_b = reader.Get3DData(data3d_b_path)

    #data3d_b = qmisc.uncrop(obj_b['segmentation'], obj_b['crinfo'],data3d_a.shape)


    #import pdb; pdb.set_trace()
    #data3d_a = (data3d_a > 1024).astype(np.int8)
    #data3d_b = (data3d_b > 0).astype(np.int8)

    #if args.visualization:
    pyed = py3DSeedEditor.py3DSeedEditor(data3d_a, contour =
    data3d_b)
    pyed.show()




show()
