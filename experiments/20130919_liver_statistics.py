#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
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


# okraj v pixelech využitý pro snížení výpočetní náročnosti během vyhodnocování AvgD
CROP_MARGIN = [20]



def sample_input_data():
    inputdata = {'basedir':'/home/mjirik/data/medical/',
            'data': [
                {'sliverseg':'data_orig/sliver07/training-part1/liver-seg001.mhd', 'sliverorig':'data_orig/sliver07/training-part1/liver-orig001.mhd'},
                {'sliverseg':'data_orig/sliver07/training-part1/liver-seg002.mhd', 'sliverorig':'data_orig/sliver07/training-part1/liver-orig002.mhd'},
                ]
            }


    sample_data_file = os.path.join(path_to_script, "20130919_liver_statistics_sample.yaml")
    #print sample_data_file, path_to_script
    misc.obj_to_file(inputdata, sample_data_file, filetype='yaml')

#def voe_metric(vol1, vol2, voxelsize_mm):


def compare_volumes(vol1, vol2, voxelsize_mm):
    """
    vol1: reference
    vol2: segmentation
    """
    volume1 = np.sum(vol1 > 0)
    volume2 = np.sum(vol2 > 0)
    volume1_mm3 = volume1 * np.prod(voxelsize_mm)
    volume2_mm3 = volume2 * np.prod(voxelsize_mm)
    print 'vol1 [mm3]: ', volume1_mm3
    print 'vol2 [mm3]: ', volume2_mm3

    df = vol1 - vol2
    df1 = np.sum(df == 1) * np.prod(voxelsize_mm)
    df2 = np.sum(df == -1) * np.prod(voxelsize_mm)

    print 'err- [mm3]: ', df1, ' err- [%]: ', df1/volume1_mm3*100
    print 'err+ [mm3]: ', df2, ' err+ [%]: ', df2/volume1_mm3*100

    #VOE[%]
    intersection = np.sum(df != 0).astype(float)
    union = (np.sum(vol1 > 0) + np.sum(vol2 > 0)).astype(float)
    voe = 100*( (intersection / union))
    print 'VOE [%]', voe


    #VD[%]
    vd = 100* (volume2-volume1).astype(float)/volume1.astype(float)
    print 'VD [%]', vd
    #import pdb; pdb.set_trace()

    #pyed = py3DSeedEditor.py3DSeedEditor(vol1, contour=vol2)
    #pyed.show()


    #get_border(vol1)
    avgd, rmsd, maxd = distance_matrics(vol1, vol2, voxelsize_mm)
    print 'AvgD [mm]', avgd
    print 'RMSD [mm]', rmsd
    print 'MaxD [mm]', maxd
    evaluation = {
            'volume1_mm3': volume1_mm3,
            'volume2_mm3': volume2_mm3,
            'err1_mm3':df1,
            'err2_mm3':df2,
            'err1_percent': df1/volume1_mm3*100,
            'err2_percent': df2/volume1_mm3*100,
            'voe': voe,
            'vd': vd,
            'avgd': avgd,
            'rmsd': rmsd,
            'maxd': maxd
            }
    return evaluation

def distance_matrics(vol1, vol2, voxelsize_mm):
    # crop data to reduce comutation time
    crinfo = qmisc.crinfo_from_specific_data(vol1, CROP_MARGIN)
    vol1 = qmisc.crop(vol1, crinfo)
    vol2 = qmisc.crop(vol2, crinfo)


    border1 = get_border(vol1)
    border2 = get_border(vol2)
    

    b1dst = scipy.ndimage.morphology.distance_transform_edt(border1,sampling=voxelsize_mm)


    dst_b1_to_b2 = border2*b1dst
    #import pdb; pdb.set_trace()
#    pyed = py3DSeedEditor.py3DSeedEditor(dst_b1_to_b2, contour=vol1)
#    pyed.show()
    avgd = np.average(dst_b1_to_b2[np.nonzero(dst_b1_to_b2)])
    rmsd = np.average(dst_b1_to_b2[np.nonzero(dst_b1_to_b2)]**2)
    maxd = np.max(dst_b1_to_b2)


    return avgd, rmsd, maxd

def get_border(image3d):
    from scipy import ndimage
    kernel = np.ones([3,3,3])
    conv = scipy.ndimage.convolve(image3d, kernel)
    conv[conv==27] = 0
    conv = conv * image3d

    conv = conv > 0
    

    #pyed = py3DSeedEditor.py3DSeedEditor(conv, contour =
    #image3d)
    #pyed.show()
    
    return conv

def write_csv(data, filename= "20130919_liver_statistics_sample.yaml"):
    import csv
    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(
                csvfile,
                delimiter=';',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
                )
        for label in data:
            spamwriter.writerow([label]+data[label])
            #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])


def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    parser = argparse.ArgumentParser(
            description='Compare two segmentation. Evaluation is similar\
            to MICCAI 2007 workshop.  Metrics are described in\
            www.sliver07.com/p7.pdf')
    parser.add_argument('-si', '--sampleInput',  action='store_true',
            help='generate sample intput data', default=False)
    parser.add_argument('-v', '--visualization',  action='store_true',
            help='Turn on visualization', default=False)
    args = parser.parse_args()

    if args.sampleInput:
        sample_input_data()
    # input parser
    data_file = os.path.join(path_to_script,  "20130919_liver_statistics_sample.yaml")
    inputdata = misc.obj_from_file(data_file, filetype='yaml')
    
    
    evaluation_all = {
            'file1': [],
            'file2': [],
            'volume1_mm3': [],
            'volume2_mm3': [],
            'err1_mm3': [],
            'err2_mm3': [],
            'err1_percent': [],
            'err2_percent': [],
            'voe': [],
            'vd': [],
            'avgd': [],
            'rmsd': [],
            'maxd': []

            }
    for i in range(0,len(inputdata['data'])):

        reader = datareader.DataReader()
        data3d_a_path = os.path.join(inputdata['basedir'], inputdata['data'][i]['sliverseg'])
        data3d_a, metadata_a = reader.Get3DData(data3d_a_path)


        data3d_b_path = os.path.join(inputdata['basedir'], inputdata['data'][i]['sliverorig'])
        data3d_b, metadata_b = reader.Get3DData(data3d_b_path)
        #data_b, metadata_b = reader.Get3DData(data3d_b_path)



        #import pdb; pdb.set_trace()
        data3d_seg = (data3d_a > 1024).astype(np.int8)
        data3d_orig = data3d_b

        if args.visualization:
            pyed = py3DSeedEditor.py3DSeedEditor(
                    data3d_orig,
                    contour=data3d_seg
                    )
            
            pyed.show()




#        evaluation_one = compare_volumes(data3d_a , data3d_b , metadata_a['voxelsize_mm'])
#        evaluation_all['file1'].append(data3d_a_path)
#        evaluation_all['file2'].append(data3d_b_path)
#        evaluation_all['volume1_mm3'].append(evaluation_one['volume1_mm3'])
#        evaluation_all['volume2_mm3'].append(evaluation_one['volume2_mm3'])
#        evaluation_all['err1_mm3'].append(evaluation_one['err1_mm3'])
#        evaluation_all['err2_mm3'].append(evaluation_one['err2_mm3'])
#        evaluation_all['err1_percent'].append(evaluation_one['err1_percent'])
#        evaluation_all['err2_percent'].append(evaluation_one['err2_percent'])
#        evaluation_all['voe'].append(evaluation_one['voe'])
#        evaluation_all['vd'].append(evaluation_one['vd'])
#        evaluation_all['avgd'].append(evaluation_one['avgd'])
#        evaluation_all['rmsd'].append(evaluation_one['rmsd'])
#        evaluation_all['maxd'].append(evaluation_one['maxd'])
#
#
    print evaluation_all
    write_csv(evaluation_all)
    #import pdb; pdb.set_trace()





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
