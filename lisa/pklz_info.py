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

import misc
import py3DSeedEditor

def main():

    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    data3d_a_path = None # os.path.join(path_to_script, '../../../data/medical/data_orig/sliver07/training-part1/liver-orig001.mhd')
    data3d_b_path = None #os.path.join(path_to_script, '../../../data/medical/data_orig/sliver07/training-part1/liver-seg001.mhd')
    parser = argparse.ArgumentParser(
            description='Information about pkl/pklz file')
    parser.add_argument('pklzFile',
            help='path to data' )
    args = parser.parse_args()

    data = misc.obj_from_file(args.pklzFile, 'pickle')
    print data

    try:
        pyed = py3DSeedEditor.py3DSeedEditor(data['data3d'], contour=data['segmentation'])
        pyed.show()
    except:
        try:

            pyed = py3DSeedEditor.py3DSeedEditor(data['data3d'])
            pyed.show()
        except:
            print "Problem with visualization"





if __name__ == "__main__":
    main()
