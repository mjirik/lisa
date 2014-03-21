#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Evaluation of liver volume error inspired by Sliver07.
"""

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

import logging
logger = logging.getLogger(__name__)


#import apdb
#  apdb.set_trace();
#import scipy.io
import numpy as np
import scipy
#from scipy import sparse
#import traceback

# ----------------- my scripts --------
import py3DSeedEditor
#import dcmreaddata1 as dcmr
#import dcmreaddata as dcmr
#import pycut
import argparse
#import py3DSeedEditor

#import segmentation
import qmisc
import misc
#import organ_segmentation
#import experiments
import datareader

# okraj v pixelech využitý pro snížení výpočetní náročnosti během vyhodnocování
# AvgD
CROP_MARGIN = [20]


def sample_input_data():
    inputdata = {'basedir':'/home/mjirik/data/medical/', # noqa
            'data': [
                {'sliverseg':'data_orig/sliver07/training-part1/liver-seg001.mhd', 'ourseg':'data_processed/organ_small-liver-orig001.mhd.pkl'},  # noqa
                {'sliverseg':'data_orig/sliver07/training-part1/liver-seg002.mhd', 'ourseg':'data_processed/organ_small-liver-orig002.mhd.pkl'},  # noqa
                {'sliverseg':'data_orig/sliver07/training-part1/liver-seg003.mhd', 'ourseg':'data_processed/organ_small-liver-orig003.mhd.pkl'},  # noqa
                {'sliverseg':'data_orig/sliver07/training-part1/liver-seg004.mhd', 'ourseg':'data_processed/organ_small-liver-orig004.mhd.pkl'},  # noqa
                {'sliverseg':'data_orig/sliver07/training-part1/liver-seg005.mhd', 'ourseg':'data_processed/organ_small-liver-orig005.mhd.pkl'},  # noqa
                {'sliverseg':'data_orig/sliver07/training-part2/liver-seg006.mhd', 'ourseg':'data_processed/organ_small-liver-orig006.mhd.pkl'},  # noqa
                {'sliverseg':'data_orig/sliver07/training-part2/liver-seg007.mhd', 'ourseg':'data_processed/organ_small-liver-orig007.mhd.pkl'},  # noqa
                {'sliverseg':'data_orig/sliver07/training-part2/liver-seg008.mhd', 'ourseg':'data_processed/organ_small-liver-orig008.mhd.pkl'},  # noqa
                {'sliverseg':'data_orig/sliver07/training-part2/liver-seg009.mhd', 'ourseg':'data_processed/organ_small-liver-orig009.mhd.pkl'},  # noqa
                ]
            }

    sample_data_file = os.path.join(path_to_script,
                                    "20130812_liver_volumetry_sample.yaml")
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
    logger.debug('vol1 [mm3]: ' + str(volume1_mm3))
    logger.debug('vol2 [mm3]: ' + str(volume2_mm3))

    df = vol1 - vol2
    df1 = np.sum(df == 1) * np.prod(voxelsize_mm)
    df2 = np.sum(df == -1) * np.prod(voxelsize_mm)

    logger.debug('err- [mm3]: ' + str(df1) + ' err- [%]: '
                 + str(df1 / volume1_mm3 * 100))
    logger.debug('err+ [mm3]: ' + str(df2) + ' err+ [%]: '
                 + str(df2 / volume1_mm3 * 100))

    #VOE[%]
    intersection = np.sum(df != 0).astype(float)
    union = (np.sum(vol1 > 0) + np.sum(vol2 > 0)).astype(float)
    voe = 100 * ((intersection / union))
    logger.debug('VOE [%]' + str(voe))

    #VD[%]
    vd = 100 * (volume2 - volume1).astype(float) / volume1.astype(float)
    logger.debug('VD [%]' + str(vd))
    #import pdb; pdb.set_trace()

    #pyed = py3DSeedEditor.py3DSeedEditor(vol1, contour=vol2)
    #pyed.show()

    #get_border(vol1)
    avgd, rmsd, maxd = distance_matrics(vol1, vol2, voxelsize_mm)
    logger.debug('AvgD [mm]' + str(avgd))
    logger.debug('RMSD [mm]' + str(rmsd))
    logger.debug('MaxD [mm]' + str(maxd))
    evaluation = {
        'volume1_mm3': volume1_mm3,
        'volume2_mm3': volume2_mm3,
        'err1_mm3': df1,
        'err2_mm3': df2,
        'err1_percent': df1 / volume1_mm3 * 100,
        'err2_percent': df2 / volume1_mm3 * 100,
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
    logger.debug(str(crinfo) + ' m1 ' + str(np.max(vol1)) +
                 ' m2 ' + str(np.min(vol2)))
    logger.debug("crinfo " + str(crinfo))
    vol1 = qmisc.crop(vol1, crinfo)
    vol2 = qmisc.crop(vol2, crinfo)

    border1 = get_border(vol1)
    border2 = get_border(vol2)

    #pyed = py3DSeedEditor.py3DSeedEditor(vol1, contour=vol1)
    #pyed.show()
    b1dst = scipy.ndimage.morphology.distance_transform_edt(
        1 - border1,
        sampling=voxelsize_mm
    )

    dst_b1_to_b2 = border2 * b1dst
    #import ipdb; ipdb.set_trace() # BREAKPOINT
    #pyed = py3DSeedEditor.py3DSeedEditor(dst_b1_to_b2, contour=vol1)
    #pyed.show()
    #print np.nonzero(border1)
    # avgd = np.average(dst_b1_to_b2[np.nonzero(border2)])
    avgd = np.average(dst_b1_to_b2[border2])
    rmsd = np.average(dst_b1_to_b2[border2] ** 2)
    maxd = np.max(dst_b1_to_b2)

    return avgd, rmsd, maxd


def get_border(image3d):
    import scipy.ndimage
    kernel = np.ones([3, 3, 3])
    conv = scipy.ndimage.convolve(image3d, kernel)
    conv[conv == 27] = 0
    conv = conv * image3d

    conv = conv > 0

    #pyed = py3DSeedEditor.py3DSeedEditor(conv, contour =
    #image3d)
    #pyed.show()

    return conv


def write_csv(data, filename='20130812_liver_volumetry.csv'):
    logger.debug(filename)
    import csv
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=';',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )
        write_sum_to_csv(data, writer)
        for label in data:
            writer.writerow([label] + data[label])
            #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])


def write_sum_to_csv(evaluation, writer):
    avg, var = make_sum(evaluation)
    key = evaluation.keys()
    writer.writerow([' - '] + key)
    writer.writerow(['var'] + var)
    writer.writerow(['avg'] + avg)
    writer.writerow([])


def eval_all(inputdata, visualization=False):
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
        'processing_time': [],
        'organ_interactivity_counter': [],
        'maxd': []

    }
    for i in range(0, len(inputdata['data'])):

        reader = datareader.DataReader()
        data3d_a_path = os.path.join(inputdata['basedir'],
                                     inputdata['data'][i]['sliverseg'])
        data3d_a, metadata_a = reader.Get3DData(data3d_a_path)

        data3d_b_path = os.path.join(inputdata['basedir'],
                                     inputdata['data'][i]['ourseg'])
        obj_b = misc.obj_from_file(data3d_b_path, filetype='pickle')
        #data_b, metadata_b = reader.Get3DData(data3d_b_path)

        data3d_b = qmisc.uncrop(obj_b['segmentation'],
                                obj_b['crinfo'], data3d_a.shape)

        #import pdb; pdb.set_trace()
        #data3d_a = (data3d_a > 1024).astype(np.int8)
        data3d_a = (data3d_a > 0).astype(np.int8)
        data3d_b = (data3d_b > 0).astype(np.int8)

        if visualization:
            pyed = py3DSeedEditor.py3DSeedEditor(data3d_a,  # + (4 * data3d_b)
                                                 contour=data3d_b)
            pyed.show()

        evaluation_one = compare_volumes(data3d_a, data3d_b,
                                         metadata_a['voxelsize_mm'])
        evaluation_all['file1'].append(data3d_a_path)
        evaluation_all['file2'].append(data3d_b_path)
        evaluation_all['volume1_mm3'].append(evaluation_one['volume1_mm3'])
        evaluation_all['volume2_mm3'].append(evaluation_one['volume2_mm3'])
        evaluation_all['err1_mm3'].append(evaluation_one['err1_mm3'])
        evaluation_all['err2_mm3'].append(evaluation_one['err2_mm3'])
        evaluation_all['err1_percent'].append(evaluation_one['err1_percent'])
        evaluation_all['err2_percent'].append(evaluation_one['err2_percent'])
        evaluation_all['voe'].append(evaluation_one['voe'])
        evaluation_all['vd'].append(evaluation_one['vd'])
        evaluation_all['avgd'].append(evaluation_one['avgd'])
        evaluation_all['rmsd'].append(evaluation_one['rmsd'])
        evaluation_all['maxd'].append(evaluation_one['maxd'])
        evaluation_all['processing_time'].append(obj_b['processing_time'])
        evaluation_all['organ_interactivity_counter'].append(
            obj_b['organ_interactivity_counter'])

    return evaluation_all


def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    default_data_file = os.path.join(path_to_script,
                                     "20130812_liver_volumetry.yaml")

    parser = argparse.ArgumentParser(
        description='Compare two segmentation. Evaluation is similar\
        to MICCAI 2007 workshop.  Metrics are described in\
        www.sliver07.com/p7.pdf')
    parser.add_argument('-d', '--debug',  action='store_true',
                        help='run in debug mode', default=False)
    parser.add_argument('-si', '--sampleInput',  action='store_true',
                        help='generate sample intput data', default=False)
    parser.add_argument('-v', '--visualization',  action='store_true',
                        help='Turn on visualization', default=False)
    parser.add_argument('-i', '--inputfile', help='input yaml file',
                        default=default_data_file)
    parser.add_argument('-o', '--outputfile',
                        help='output file without extension',
                        default='20130812_liver_volumetry')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('debug mode logging')

    if args.sampleInput:
        sample_input_data()
    # input parser
    data_file = args.inputfile
    inputdata = misc.obj_from_file(data_file, filetype='yaml')

    evaluation_all = eval_all(inputdata, args.visualization)

    logger.debug(str(evaluation_all))
    logger.debug('eval all')

    logger.debug(make_sum(evaluation_all))
    write_csv(evaluation_all, filename=args.outputfile+'.csv')
    misc.obj_to_file(evaluation_all, args.outputfile+'.pkl', filetype='pkl')
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


def make_sum(evaluation):
    var = []
    avg = []
    for key in evaluation.keys():
        avgi = 'nan'
        vari = 'nan'
        try:
            avgi = np.average(evaluation[key])
            vari = np.var(evaluation[key])
        except Exception:
            print "problem with key: ", key
            #print evaluation[key]
        avg.append(avgi)
        var.append(vari)
    return avg, var


if __name__ == "__main__":
    main()
