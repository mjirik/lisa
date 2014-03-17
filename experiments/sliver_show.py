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
import argparse

import logging
logger = logging.getLogger(__name__)


import datareader
import misc
import qmisc
import py3DSeedEditor



def show(data3d_a_path, data3d_b_path, ourSegmentation):
    reader = datareader.DataReader()
    #data3d_a_path = os.path.join(path_to_script, data3d_a_path)
    data3d_a, metadata_a = reader.Get3DData(data3d_a_path)

    if data3d_b_path is not None:
        data3d_b_path = os.path.join(path_to_script, data3d_b_path)
        data3d_b, metadata_b = reader.Get3DData(data3d_b_path)

        pyed = py3DSeedEditor.py3DSeedEditor(data3d_a, contour =
        data3d_b)
        pyed.show()

    if ourSegmentation != None:
        ourSegmentation= os.path.join(path_to_script, ourSegmentation)
        data_our = misc.obj_from_file(ourSegmentation, 'pickle')
        #data3d_our = data_our['segmentation']
        data3d_our = qmisc.uncrop(data_our['segmentation'], data_our['crinfo'],data3d_a.shape)
#@TODO dodělat uncrop  a podobné kratochvíle

    #data3d_b_path = os.path.join(inputdata['basedir'], inputdata['data'][i]['ourseg'])
    #obj_b = misc.obj_from_file(data3d_b_path, filetype='pickle')
    #data_b, metadata_b = reader.Get3DData(data3d_b_path)

    #data3d_b = qmisc.uncrop(obj_b['segmentation'], obj_b['crinfo'],data3d_a.shape)


    #import pdb; pdb.set_trace()
    #data3d_a = (data3d_a > 1024).astype(np.int8)
    #data3d_b = (data3d_b > 0).astype(np.int8)

    #if args.visualization:

    if ourSegmentation != None:
        pyed = py3DSeedEditor.py3DSeedEditor(data3d_a, contour =
        data3d_our)
        pyed.show()

def main():

    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    data3d_a_path = None # os.path.join(path_to_script, '../../../data/medical/data_orig/sliver07/training-part1/liver-orig001.mhd')
    data3d_b_path = None #os.path.join(path_to_script, '../../../data/medical/data_orig/sliver07/training-part1/liver-seg001.mhd')
    parser = argparse.ArgumentParser(
            description='Visualization of sliver data and our segmentation')
    parser.add_argument('-sd', '--sliverData',
            help='path to input sliver data', default=data3d_a_path)
    parser.add_argument('-ss', '--sliverSegmentation',
            help='path to input sliver segmentation', default=data3d_b_path)
    parser.add_argument('-os', '--ourSegmentation',
            help='path to out pklz or pkl segmentation', default=None)
    args = parser.parse_args()

    show(args.sliverData, args.sliverSegmentation, args.ourSegmentation)



if __name__ == "__main__":
    main()
