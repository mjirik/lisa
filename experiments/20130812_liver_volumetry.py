#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/"))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/pycat/extern/py3DSeedEditor/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)


#import apdb
#  apdb.set_trace();
#import scipy.io
import numpy as np
import scipy
#from scipy import sparse
import traceback

# ----------------- my scripts --------
import py3DSeedEditor
#import dcmreaddata1 as dcmr
import dcmreaddata as dcmr
import pycut
import argparse
#import py3DSeedEditor

import segmentation
import qmisc
import misc
import organ_segmentation
import experiments
import datareader


def sample_input_data():
    inputdata = {'basedir':'/home/mjirik/data/medical/',
            'data': [
               # {'sliverseg':'data_orig/sliver07/training-part1/liver-seg001.mhd', 'ourseg':'data_processed/organ_small-liver-orig001.mhd.pkl'},
               # {'sliverseg':'data_orig/sliver07/training-part1/liver-seg002.mhd', 'ourseg':'data_processed/organ_small-liver-orig002.mhd.pkl'},
               # {'sliverseg':'data_orig/sliver07/training-part1/liver-seg003.mhd', 'ourseg':'data_processed/organ_small-liver-orig003.mhd.pkl'},
               # {'sliverseg':'data_orig/sliver07/training-part1/liver-seg004.mhd', 'ourseg':'data_processed/organ_small-liver-orig004.mhd.pkl'},
                {'sliverseg':'data_orig/sliver07/training-part1/liver-seg005.mhd', 'ourseg':'data_processed/organ_small-liver-orig005.mhd.pkl'},
                ]
            }


    sample_data_file = os.path.join(path_to_script, "20130812_liver_volumetry_sample.yaml")
    #print sample_data_file, path_to_script
    misc.obj_to_file(inputdata, sample_data_file, filetype='yaml')

def compare_volumes(vol1, vol2, voxelsize_mm):
    volume1_mm3 = np.sum(vol1 > 0) * np.prod(voxelsize_mm)
    volume2_mm3 = np.sum(vol2 > 0) * np.prod(voxelsize_mm)
    print 'vol1 [mm3]: ', volume1_mm3
    print 'vol2 [mm3]: ', volume2_mm3

    df = vol1 - vol2
    df1 = np.sum(df == 1) * np.prod(voxelsize_mm)
    df2 = np.sum(df == -1) * np.prod(voxelsize_mm)

    print 'err+ [mm3]: ', df1, ' err+ [%]: ', df1/volume1_mm3
    print 'err- [mm3]: ', df2, ' err- [%]: ', df2/volume1_mm3
    #pyed = py3DSeedEditor.py3DSeedEditor(df, contour =
    # vol2)
    #pyed.show()

    evaluation = {
            'volume1_mm3': volume1_mm3,
            'volume2_mm3': volume2_mm3,
            'err1_mm3':df1,
            'err2_mm3':df2,
            'err1_percent': df1/volume1_mm3,
            'err2_percent': df2/volume1_mm3,

            }
    return evaluation


def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    sample_input_data()
    # input parser
    print 'file' , __file__
    data_file = os.path.join(path_to_script, "20130812_liver_volumetry_sample.yaml")
    inputdata = misc.obj_from_file(data_file, filetype='yaml')
    
    
    evaluation_all = {
            'file1': [],
            'file2': [],
            'volume1_mm3': [],
            'volume2_mm3': [],
            'err1_mm3': [],
            'err2_mm3': [],
            'err1_percent': [],
            'err2_percent': []

            }
    for i in range(0,len(inputdata['data'])):

        reader = datareader.DataReader()
        data3d_a_path = os.path.join(inputdata['basedir'], inputdata['data'][i]['sliverseg'])
        data3d_a, metadata_a = reader.Get3DData(data3d_a_path)


        data3d_b_path = os.path.join(inputdata['basedir'], inputdata['data'][i]['ourseg'])
        obj_b = misc.obj_from_file(data3d_b_path, filetype='pickle')
        #data_b, metadata_b = reader.Get3DData(data3d_b_path)

        data3d_b = qmisc.uncrop(obj_b['segmentation'], obj_b['crinfo'],data3d_a.shape)


        #import pdb; pdb.set_trace()
        data3d_a = (data3d_a > 1024).astype(np.int8)
        data3d_b = (data3d_b > 0).astype(np.int8)

        pyed = py3DSeedEditor.py3DSeedEditor(data3d_b, contour =
        data3d_a)
        pyed.show()




        evaluation_one = compare_volumes(data3d_a , data3d_b , metadata_a['voxelsizemm'])
        evaluation_all['file1'] = data3d_a_path
        evaluation_all['file2'] = data3d_b_path
        evaluation_all['volume1_mm3'].append(evaluation_one['volume1_mm3'])
        evaluation_all['volume2_mm3'].append(evaluation_one['volume2_mm3'])
        evaluation_all['err1_mm3'].append(evaluation_one['err1_mm3'])
        evaluation_all['err2_mm3'].append(evaluation_one['err2_mm3'])
        evaluation_all['err1_percent'].append(evaluation_one['err1_percent'])
        evaluation_all['err2_percent'].append(evaluation_one['err2_percent'])


    print evaluation_all




    #igc = pycat.ImageGraphCut(data3d, zoom = 0.5)
    #igc.interactivity()

    #igc.make_gc()
    #igc.show_segmentation()

    # volume
    #volume_mm3 = np.sum(oseg.segmentation > 0) * np.prod(oseg.voxelsize_mm)

    #pyed = py3DSeedEditor.py3DSeedEditor(oseg.data3d, contour =
    # oseg.segmentation)
    #pyed.show()

#    if args.show_output:
#        oseg.show_output()
#
#    savestring = raw_input('Save output data? (y/n): ')
#    #sn = int(snstring)
#    if savestring in ['Y', 'y']:
#
#        data = oseg.export()
#
#        misc.obj_to_file(data, "organ.pkl", filetype='pickle')
#        misc.obj_to_file(oseg.get_ipars(), 'ipars.pkl', filetype='pickle')
#    #output = segmentation.vesselSegmentation(oseg.data3d,
    # oseg.orig_segmentation)

if __name__ == "__main__":
    main()
