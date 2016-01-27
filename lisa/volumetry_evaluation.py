#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Evaluation of liver volume error inspired by Sliver07.
Input is YAML file with dataset metadata.

Function compare_volumes can be used to compare two data.

Compute Sliver07 score::

eval_data = lisa.volumetry_evaluation.compare_volumes(
    ndimage_3d_segmentation_1,
    ndimage_3d_segmentation_2,
    voxelsize_mm)

score = lisa.volumetry_evaluation.sliver_score_one_couple(eval_data)

"""

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/"))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/sed3/"))
# sys.path.append(os.path.join(path_to_script, "../extern/"))
# import featurevector

import logging

logger = logging.getLogger(__name__)


# import apdb
# apdb.set_trace();
# import scipy.io
import numpy as np
import scipy
# from scipy import sparse
# import traceback

# ----------------- my scripts --------
import sed3
# import dcmreaddata1 as dcmr
# import dcmreaddata as dcmr
import argparse

# import segmentation
import qmisc
import misc
# import organ_segmentation
# import experiments
from io3d import datareader

# okraj v pixelech využitý pro snížení výpočetní náročnosti během vyhodnocování
# AvgD
CROP_MARGIN = [20]


def evaluate_and_write_to_file(
        inputYamlFile,
        directoryPklz,
        directorySliver,
        outputfile,
        visualization,
        return_dir_lists=False,
        special_evaluation_function=None,
        return_all_data=False
):
    """
    Function computes yaml file (if there are given input sliver and pklz
    directories). Based on yaml file are compared sliver segmentations and
    our pklz files.
    """
    dirlists = None
    if (directoryPklz is not None) and (directorySliver is not None):
        dirlists = generate_input_yaml(
            directorySliver,
            directoryPklz,
            yaml_filename=inputYamlFile,
            return_dir_lists=return_dir_lists
        )

    # input parser
    data_file = inputYamlFile
    inputdata = misc.obj_from_file(data_file, filetype='yaml')

    evaluation_all = eval_all_from_dataset_metadata(
        inputdata, visualization,
        special_evaluation_function=special_evaluation_function
    )

    logger.debug(str(evaluation_all))
    logger.debug('eval all')

    logger.debug(make_sum(evaluation_all))
    write_csv(evaluation_all, filename=outputfile + '.csv')
    misc.obj_to_file(evaluation_all, outputfile + '.pkl', filetype='pkl')

    retval = []

    if return_dir_lists:
        retval.append(dirlists)
    if return_all_data:
        retval.append(evaluation_all)
    if len(retval) > 0:
        return retval

        # import pdb; pdb.set_trace()

        # volume
        # volume_mm3 = np.sum(oseg.segmentation > 0) * np.prod(oseg.voxelsize_mm)

        # pyed = sed3.sed3(oseg.data3d, contour =
        # oseg.segmentation)
        # pyed.show()

    # if args.show_output:
    # oseg.show_output()
    #
    # savestring = raw_input('Save output data? (y/n): ')
    # sn = int(snstring)
    # if savestring in ['Y', 'y']:
    #
    #        data = oseg.export()
    #
    #        misc.obj_to_file(data, "organ.pkl", filetype='pickle')
    #        misc.obj_to_file(oseg.get_ipars(), 'ipars.pkl', filetype='pickle')
    # output = segmentation.vesselSegmentation(oseg.data3d,
    # oseg.orig_segmentation)


def eval_all_from_dataset_metadata(inputdata, visualization=False,
                                   special_evaluation_function=None):
    """
    set metadata
    """
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
        print "inputdata ", inputdata['data'][i].values()
        try:
            # if there is defined overlay
            data3d_a = reader.GetOverlay()[inputdata['data'][i][
                'overlay_number']]
            logger.info('overlay loaded')
            print 'overlay loaded'
        except:
            logger.debug("overlay not loaded")
            pass

        logger.debug('data A shape ' + str(data3d_a.shape))

        data3d_b_path = os.path.join(inputdata['basedir'],
                                     inputdata['data'][i]['ourseg'])
        obj_b = misc.obj_from_file(data3d_b_path, filetype='pickle')
        # data_b, metadata_b = reader.Get3DData(data3d_b_path)

        if 'crinfo' in obj_b.keys():
            data3d_b = qmisc.uncrop(obj_b['segmentation'],
                                    obj_b['crinfo'], data3d_a.shape)
        else:
            data3d_b = obj_b['segmentation']

        # import pdb; pdb.set_trace()
        # data3d_a = (data3d_a > 1024).astype(np.int8)
        data3d_a = (data3d_a > 0).astype(np.int8)
        data3d_b = (data3d_b > 0).astype(np.int8)

        if visualization:
            pyed = sed3.sed3(data3d_a,  # + (4 * data3d_b)
                             contour=data3d_b)
            pyed.show()

        evaluation_one = compare_volumes(data3d_a, data3d_b,
                                         metadata_a['voxelsize_mm'])
        if special_evaluation_function is not None:
            evaluation_one.update(
                special_evaluation_function(
                    data3d_a, data3d_b, metadata_a['voxelsize_mm']
                ))
        evaluation_all['file1'].append(data3d_a_path)
        evaluation_all['file2'].append(data3d_b_path)
        for key in evaluation_one.keys():
            if key not in evaluation_all.keys():
                evaluation_all[key] = []

            evaluation_all[key].append(evaluation_one[key])
        # evaluation_all['volume1_mm3'].append(evaluation_one['volume1_mm3'])
        # evaluation_all['volume2_mm3'].append(evaluation_one['volume2_mm3'])
        # evaluation_all['err1_mm3'].append(evaluation_one['err1_mm3'])
        # evaluation_all['err2_mm3'].append(evaluation_one['err2_mm3'])
        # evaluation_all['err1_percent'].append(evaluation_one['err1_percent'])
        # evaluation_all['err2_percent'].append(evaluation_one['err2_percent'])
        # evaluation_all['voe'].append(evaluation_one['voe'])
        # evaluation_all['vd'].append(evaluation_one['vd'])
        # evaluation_all['avgd'].append(evaluation_one['avgd'])
        # evaluation_all['rmsd'].append(evaluation_one['rmsd'])
        # evaluation_all['maxd'].append(evaluation_one['maxd'])
        if 'processing_time' in obj_b.keys():
            # this is only for compatibility with march2014 data
            processing_time = obj_b['processing_time']
            organ_interactivity_counter = obj_b['organ_interactivity_counter']
        else:
            try:
                processing_time = obj_b['processing_information']['organ_segmentation']['processing_time']  # noqa
                organ_interactivity_counter = obj_b['processing_information']['organ_segmentation'][
                    'organ_interactivity_counter']  # noqa
            except:
                processing_time = 0
                organ_interactivity_counter = 0
        evaluation_all['processing_time'].append(processing_time)
        evaluation_all['organ_interactivity_counter'].append(
            organ_interactivity_counter)

    return evaluation_all


def compare_volumes_boundingbox(vol1, vol2, voxelsize_mm):
    import qmisc

    crinfo = qmisc.crinfo_from_specific_data(vol1, [20, 20, 20])
    vol1[
    crinfo[0][0]:crinfo[0][1],
    crinfo[1][0]:crinfo[1][1],
    crinfo[2][0]:crinfo[2][1]
    ] = 1

    volume1 = np.sum(vol1 > 0)
    volume2 = np.sum(vol2 > 0)
    volume1_mm3 = volume1 * np.prod(voxelsize_mm)
    volume2_mm3 = volume2 * np.prod(voxelsize_mm)
    volume_avg_mm3 = (volume1_mm3 + volume2_mm3) * 0.5

    df = vol1 - vol2
    df1 = np.sum(df == 1) * np.prod(voxelsize_mm)
    df2 = np.sum(df == -1) * np.prod(voxelsize_mm)

    evaluation = {
        'box_err1_mm3': df1,
        'box_err2_mm3': df2,
        'box_err1_percent': df1 / volume_avg_mm3 * 100,
        'box_err2_percent': df2 / volume_avg_mm3 * 100,
    }
    return evaluation

def compare_volumes_sliver(vol1, vol2, voxelsize_mm, use_logger=False):
    """
    Computes statistics from similarity of vol1 and vol2. Return the same as
    compare_volumes with additional information with sliver score
    :param vol1:
    :param vol2:
    :param voxelsize_mm:
    :param use_logger:
    :return:
    """

    evaluation = compare_volumes(vol1, vol2, voxelsize_mm, use_logger)
    sliver_evaluation = sliver_score_one_couple(
        evaluation,
        keys=[
            'sliver_vd_pts',
            'sliver_voe_pts',
            'sliver_avgd_pts',
            'sliver_rmsd_pts',
            'sliver_maxd_pts'
        ])
    overall = sliver_overall_score_for_one_couple(sliver_evaluation)

    evaluation.update(sliver_evaluation)
    evaluation['sliver_overall_pts'] = overall
    return evaluation

def compare_volumes(vol1, vol2, voxelsize_mm, use_logger=False):
    """
    computes metrics, no sliver computed here, see compare_volumes_sliver

    vol1: reference
    vol2: segmentation
    """
    volume1 = np.sum(vol1 > 0)
    volume2 = np.sum(vol2 > 0)
    volume1_mm3 = volume1 * np.prod(voxelsize_mm)
    volume2_mm3 = volume2 * np.prod(voxelsize_mm)
    volume_avg_mm3 = (volume1_mm3 + volume2_mm3) * 0.5
    if use_logger:
        logger.debug('vol1 [mm3]: ' + str(volume1_mm3))
        logger.debug('vol2 [mm3]: ' + str(volume2_mm3))

    df = vol1 - vol2
    df1 = np.sum(df == 1) * np.prod(voxelsize_mm)
    df2 = np.sum(df == -1) * np.prod(voxelsize_mm)

    if use_logger:
        logger.debug('err- [mm3]: ' + str(df1) + ' err- [%]: '
                     + str(df1 / volume_avg_mm3 * 100))
        logger.debug('err+ [mm3]: ' + str(df2) + ' err+ [%]: '
                     + str(df2 / volume_avg_mm3 * 100))

    # VOE[%]
    intersection = np.sum(df != 0).astype(np.float)
    union = (np.sum(vol1 > 0) + np.sum(vol2 > 0)).astype(float)
    voe = 100 * (intersection / union)
    if use_logger:
        logger.debug('VOE [%]' + str(voe))

    # VD[%]
    vd = 100 * (volume2 - volume1).astype(float) / volume1.astype(float)
    if use_logger:
        logger.debug('VD [%]' + str(vd))
    # import pdb; pdb.set_trace()

    # pyed = sed3.sed3(vol1, contour=vol2)
    # pyed.show()

    # get_border(vol1)
    avgd, rmsd, maxd = distance_matrics(vol1, vol2, voxelsize_mm)
    if use_logger:
        logger.debug('AvgD [mm]' + str(avgd))
        logger.debug('RMSD [mm]' + str(rmsd))
        logger.debug('MaxD [mm]' + str(maxd))
    evaluation = {
        'volume1_mm3': volume1_mm3,
        'volume2_mm3': volume2_mm3,
        'err1_mm3': df1,
        'err2_mm3': df2,
        'err1_percent': df1 / volume_avg_mm3 * 100,
        'err2_percent': df2 / volume_avg_mm3 * 100,
        'vd': vd,
        'voe': voe,
        'avgd': avgd,
        'rmsd': rmsd,
        'maxd': maxd
    }

    return evaluation


def distance_matrics(vol1, vol2, voxelsize_mm):
    # crop data to reduce comutation time
    crinfo = qmisc.crinfo_from_specific_data(vol1 + vol2, CROP_MARGIN)
    logger.debug(str(crinfo) + ' m1 ' + str(np.max(vol1)) +
                 ' m2 ' + str(np.min(vol2)))
    logger.debug("crinfo " + str(crinfo))
    vol1 = qmisc.crop(vol1, crinfo)
    vol2 = qmisc.crop(vol2, crinfo)

    border1 = _get_border(vol1)
    border2 = _get_border(vol2)

    # pyed = sed3.sed3(vol1, contour=vol1)
    # pyed.show()
    b1dst = scipy.ndimage.morphology.distance_transform_edt(
        1 - border1,
        sampling=voxelsize_mm
    )
    b2dst = scipy.ndimage.morphology.distance_transform_edt(
        1 - border2,
        sampling=voxelsize_mm
    )

    dst_b1_to_b2 = border2 * b1dst
    dst_b2_to_b1 = border1 * b2dst
    dst_12 = dst_b1_to_b2[border2]
    dst_21 = dst_b2_to_b1[border1]
    dst_both = np.append(dst_12, dst_21)

    # sum_d12 = np.sum(dst_12)
    # sum_d21 = np.sum(dst_21)
    # len_d12 = len(dst_12)
    # len_d21 = len(dst_21)
    # import ipdb; ipdb.set_trace() # BREAKPOINT
    # pyed = sed3.sed3(dst_b1_to_b2, contour=vol1)
    # pyed.show()
    # print np.nonzero(border1)
    # avgd = np.average(dst_b1_to_b2[np.nonzero(border2)])
    # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
    # avgd = (sum_d21 + sum_d12)/float(len_d21 + len_d12)
    # rmsd = ((np.sum(dst_12**2) + dst_21**2)/float(len_d21 + len_d12))**0.5
    avgd = np.average(dst_both)

    # there is not clear what is correct
    # rmsd = np.average(dst_both ** 2)**0.5
    # rmsd = (np.average(dst_12) + np.average(dst_21))**0.5
    rmsd = np.average(dst_both ** 2)
    # rmsd = np.average(dst_b1_to_b2[border2] ** 2)
    maxd = max(np.max(dst_b1_to_b2), np.max(dst_b2_to_b1))
    # old
    # avgd = np.average(dst_b1_to_b2[border2])
    # rmsd = np.average(dst_b1_to_b2[border2] ** 2)
    # maxd = np.max(dst_b1_to_b2)

    return avgd, rmsd, maxd


def _get_border(image3d):
    import scipy.ndimage

    kernel = np.ones([3, 3, 3])
    conv = scipy.ndimage.convolve(image3d, kernel)
    conv[conv == 27] = 0
    conv = conv * image3d

    conv = conv > 0

    # pyed = sed3.sed3(conv, contour =
    # image3d)
    # pyed.show()

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
            # spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])


def write_sum_to_csv(evaluation, writer):
    avg, var = make_sum(evaluation)
    key = evaluation.keys()
    writer.writerow([' - '] + key)
    writer.writerow(['var'] + var)
    writer.writerow(['avg'] + avg)
    writer.writerow([])


def sliver_overall_score_for_one_couple(score):
    """
    Computes overall score:
    """
    score_list = score.values()
    overall_score = np.average(np.array(score_list))

    return overall_score


def sliverScore(measure, metric_type):
    """
    deprecated name of sliver_score. Use new sliver_score() function.
    """
    print "Deprecated function sliverScore(). Use sliver_score()."
    return sliver_score(measure, metric_type)


def sliver_score(measure, metric_type):
    """
    Based on sliver metodics
    http://sliver07.org/p7.pdf

    Slope and intercept comutations:
    https://docs.google.com/spreadsheet/ccc?key=0AkBzbxly5bqfdEJaOWJJUEh5ajVJM05YWGdaX1k5aFE#gid=0   # noqa

    """
    slope = -1
    intercept = 100

    if metric_type is 'vd':
        slope = -3.90625
    elif metric_type is 'voe':
        slope = -5.31914893617021
    elif metric_type is 'avgd':
        slope = -25
    elif metric_type is 'rmsd':
        slope = -14.7058823529412
    elif metric_type is 'maxd':
        slope = -1.31578947368421

    score = intercept + np.abs(measure) * slope

    try:
        score[score < 0] = 0
    except:
        # if score is scalar
        if score < 0:
            score = 0

    return score


def sliver_score_one_couple(
        data,
        keys=['vd', 'voe', 'avgd', 'rmsd', 'maxd']):
    """
    Convert data from compare_volumes() function.
    :param keys: set output keys
    """
    score = {
        keys[0]: sliver_score(data['vd'], 'vd'),
        keys[1]: sliver_score(data['voe'], 'voe'),
        keys[2]: sliver_score(data['avgd'], 'avgd'),
        keys[3]: sliver_score(data['rmsd'], 'rmsd'),
        keys[4]: sliver_score(data['maxd'], 'maxd'),
    }
    return score


def sliverScoreAll(data):
    """
    Computers score by Sliver07
    http://sliver07.org/p7.pdf

    input: dataset = [{'vd':[1.1, ..., 0.1], 'voe':[...], 'avgd':[...], ...}
                      {'vd':[...], 'voe':[...], ...}
                     ]
    return: scoreTotal, scoreMetrics, scoreAll
        Order of scoreMetrics is [vd, voe, avgd, rmsd, maxd]


    """

    scoreAll = []
    scoreTotal = []
    scoreMetrics = []
    for dat in data:
        score = {
            'vd': sliver_score(dat['vd'], 'vd'),
            'voe': sliver_score(dat['voe'], 'voe'),
            'avgd': sliver_score(dat['avgd'], 'avgd'),
            'rmsd': sliver_score(dat['rmsd'], 'rmsd'),
            'maxd': sliver_score(dat['maxd'], 'maxd'),
        }
        scoreAll.append(score)

        metrics = [np.mean(score['vd']),
                   np.mean(score['voe']),
                   np.mean(score['avgd']),
                   np.mean(score['rmsd']),
                   np.mean(score['maxd'])
                   ]
        scoreMetrics.append(metrics)

        total = np.mean(metrics)
        scoreTotal.append(total)

    return scoreTotal, scoreMetrics, scoreAll


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
            # print evaluation[key]
        avg.append(avgi)
        var.append(vari)
    return avg, var


def main():
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # logger.debug('input params')

    default_data_file = os.path.join(path_to_script,
                                     "20130812_liver_volumetry.yaml")

    parser = argparse.ArgumentParser(
        description='Compare two segmentation. Evaluation is similar\
        to MICCAI 2007 workshop.  Metrics are described in\
        www.sliver07.com/p7.pdf')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='run in debug mode', default=False)
    parser.add_argument('-si', '--sampleInput', action='store_true',
                        help='generate sample intput data', default=False)
    parser.add_argument('-v', '--visualization', action='store_true',
                        help='Turn on visualization', default=False)
    parser.add_argument('-y', '--inputYamlFile', help='input yaml file',
                        default=default_data_file)
    parser.add_argument('-ds', '--directorySliver',
                        help='input SLiver directory. If this and\
                        directoryPklz is not None, yaml file is generated',
                        default=None)
    parser.add_argument('-dp', '--directoryPklz', help='input pklz directory',
                        default=None)
    parser.add_argument('-o', '--outputfile',
                        help='output file without extension',
                        default='20130812_liver_volumetry')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('debug mode logging')

    if args.sampleInput:
        sample_input_data()

    evaluate_and_write_to_file(
        args.inputYamlFile,
        args.directoryPklz,
        args.directorySliver,
        args.outputfile,
        args.visualization
    )


def generate_input_yaml(sliver_dir, pklz_dir,
                        sliver_ext='*seg0*.mhd', pklz_ext='*0*.pklz',
                        yaml_filename=None,
                        return_dir_lists=False
                        ):
    """
    Function pair files from different directory by numer in format g0XX.
    It is ok for seg001 and orig001 too.
    If there is given some yaml_filename, it is created.
    """
    import glob
    import re

    if not os.path.exists(sliver_dir):
        raise IOError("Directory with reference data does not exists")
    if not os.path.exists(pklz_dir):
        raise IOError("Directory with input pklz data does not exists")

    onlyfiles1 = glob.glob(os.path.join(sliver_dir, sliver_ext))
    onlyfiles2 = glob.glob(os.path.join(pklz_dir, pklz_ext))
    onlyfiles1.sort()
    onlyfiles2.sort()
    if len(onlyfiles1) == 0:
        raise IOError("Directory with reference data appears to be empty")
    if len(onlyfiles2) == 0:
        raise IOError("Directory with pklz data appears to be empty")

    logger.debug('sliver files \n' + str(onlyfiles1))
    logger.debug('pklz files \n' + str(onlyfiles2))

    data = []
    for flns in onlyfiles1:
        base, flnsh = os.path.split(os.path.normpath(flns))
        pattern = re.search('(g0[0-9]{2})', flnsh)
        if pattern:
            pattern = pattern.group(1)
        logger.debug('pattern1 ' + pattern)

        for flnp in onlyfiles2:
            base, flnph = os.path.split(os.path.normpath(flnp))
            pt = re.match('.*' + pattern + '.*', flnph)
            if pt:
                data.append({
                    'sliverseg': flns,
                    'ourseg': flnp
                })

    inputdata = {
        'basedir': '',
        'data': data
    }

    retval = []

    if yaml_filename is None:
        retval.append(inputdata)
    else:
        misc.obj_to_file(inputdata, yaml_filename, filetype='yaml')

    if return_dir_lists:
        retval.append(onlyfiles1)
        retval.append(onlyfiles2)

    if len(retval) > 1:
        return tuple(retval)
    elif len(retval) == 1:
        return retval[0]


def sample_input_data():
    inputdata = {'basedir': '/home/mjirik/data/medical/',  # noqa
                 'data': [
                     {'sliverseg': 'data_orig/sliver07/training-part1/liver-seg001.mhd',
                      'ourseg': 'data_processed/organ_small-liver-orig001.mhd.pkl'},  # noqa
                     {'sliverseg': 'data_orig/sliver07/training-part1/liver-seg002.mhd',
                      'ourseg': 'data_processed/organ_small-liver-orig002.mhd.pkl'},  # noqa
                     {'sliverseg': 'data_orig/sliver07/training-part1/liver-seg003.mhd',
                      'ourseg': 'data_processed/organ_small-liver-orig003.mhd.pkl'},  # noqa
                     {'sliverseg': 'data_orig/sliver07/training-part1/liver-seg004.mhd',
                      'ourseg': 'data_processed/organ_small-liver-orig004.mhd.pkl'},  # noqa
                     {'sliverseg': 'data_orig/sliver07/training-part1/liver-seg005.mhd',
                      'ourseg': 'data_processed/organ_small-liver-orig005.mhd.pkl'},  # noqa
                     {'sliverseg': 'data_orig/sliver07/training-part2/liver-seg006.mhd',
                      'ourseg': 'data_processed/organ_small-liver-orig006.mhd.pkl'},  # noqa
                     {'sliverseg': 'data_orig/sliver07/training-part2/liver-seg007.mhd',
                      'ourseg': 'data_processed/organ_small-liver-orig007.mhd.pkl'},  # noqa
                     {'sliverseg': 'data_orig/sliver07/training-part2/liver-seg008.mhd',
                      'ourseg': 'data_processed/organ_small-liver-orig008.mhd.pkl'},  # noqa
                     {'sliverseg': 'data_orig/sliver07/training-part2/liver-seg009.mhd',
                      'ourseg': 'data_processed/organ_small-liver-orig009.mhd.pkl'},  # noqa
                 ]
                 }

    sample_data_file = os.path.join(path_to_script,
                                    "20130812_liver_volumetry_sample.yaml")
    # print sample_data_file, path_to_script
    misc.obj_to_file(inputdata, sample_data_file, filetype='yaml')

# def voe_metric(vol1, vol2, voxelsize_mm):


if __name__ == "__main__":
    main()
