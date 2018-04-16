#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Experiment s \"barevným\" modelem jater.

Pro spuštění zkuste help:

python experiments/20130919_liver_statistics.py --help

Měřené vlastnosti se přidávají do get_features().
Pro přidání dalších dat, editujte příslušný yaml soubor.

"""

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/"))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/sed3/"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/lbp/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector

import logging
logger = logging.getLogger(__name__)


#import apdb
#  apdb.set_trace();
#import scipy.io
import numpy as np
#import scipy
#from scipy import sparse
#import traceback
import itertools

# ----------------- my scripts --------
import sed3
#import dcmreaddata1 as dcmr
#import dcmreaddata as dcmr
import argparse
#import sed3

import misc
import datareader
import matplotlib.pyplot as plt
import experiments


def feat_hist_by_segmentation(data3d_orig, data3d_seg, visualization=True):
    bins = range(-1024, 1024, 1)
    bins = range(-512, 512, 1)
    hist1, bin_edges1 = np.histogram(data3d_orig[data3d_seg > 0], bins=bins)
    hist2, bin_edges2 = np.histogram(data3d_orig[data3d_seg <= 0], bins=bins)
    #import pdb; pdb.set_trace()
    if visualization:
        plt_liver = plt.step(bin_edges1[1:], hist1)
        plt_rest = plt.step(bin_edges2[1:], hist2)
        plt.legend([plt_liver, plt_rest], ['Liver', 'Other tissue'])
        #plt.plot(bin_edges1[1:], hist1, bin_edges2[1:], hist2)
        plt.show()
    fv_hist = {'hist1': hist1,
               'hist2': hist2,
               'bins': bins
               }
    return fv_hist


def feat_hist(data3d_orig):
    bins = range(-1024, 1024, 1)
    bins = range(-512, 512, 1)
    bins = range(-512, 512, 10)
    bins = range(-512, 512, 64)
    bins = range(-512, 512, 100)
    hist1, bin_edges1 = np.histogram(data3d_orig, bins=bins)
    return hist1



def get_features(data3d_orig, data3d_seg, feature_fcn, visualization=True):
    u"""
    Sem doplníme všechny naše měření.

    Pro ukázku jsem vytvořil měření
    histogramu.
    data3d_orig: CT data3d_orig
    data3d_seg: jedničky tam, kde jsou játra

    """

    featur = feature_fcn(data3d_orig)
    #featur = {}
    #featur['hist'] = feat_hist(data3d_orig, visualization)
    #featur['lbp'] = lbp(data3d_orig, data3d_seg, visualization)

    return featur


def sample_input_data():
    inputdata = {
        'basedir': '/home/mjirik/data/medical/',
        'data': [
            {'sliverseg': 'data_orig/sliver07/training-part1/liver-seg001.mhd', 'sliverorig': 'data_orig/sliver07/training-part1/liver-orig001.mhd'},
            {'sliverseg': 'data_orig/sliver07/training-part1/liver-seg002.mhd', 'sliverorig': 'data_orig/sliver07/training-part1/liver-orig002.mhd'},
        ]
    }

    sample_data_file = os.path.join(path_to_script,
                                    "20130919_liver_statistics.yaml")
    #print sample_data_file, path_to_script
    misc.obj_to_file(inputdata, sample_data_file, filetype='yaml')

#def voe_metric(vol1, vol2, voxelsize_mm):


def write_csv(data, filename="20130919_liver_statistics.yaml"):
    import csv
    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile,
                                delimiter=';',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL
                                )
        for label in data:
            spamwriter.writerow([label] + data[label])
            #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])


#def training_dataset_prepare
def read_data_orig_and_seg(inputdata, i):
    """ Loads data_orig and data_seg from yaml file
    """

    reader = datareader.DataReader()
    data3d_a_path = os.path.join(inputdata['basedir'],
                                 inputdata['data'][i]['sliverseg'])
    data3d_a, metadata_a = reader.Get3DData(data3d_a_path, dataplus_format=False)

    data3d_b_path = os.path.join(inputdata['basedir'],
                                 inputdata['data'][i]['sliverorig'])
    data3d_b, metadata_b = reader.Get3DData(data3d_b_path, dataplus_format=False)

    #import pdb; pdb.set_trace()
    data3d_seg = (data3d_a > 0).astype(np.int8)
    data3d_orig = data3d_b

    return data3d_orig, data3d_seg


def one_experiment_setting_for_whole_dataset(inputdata, tile_shape,
                                             feature_fcn, classif_fcn, train,
                                             visualization=False):
    fvall = []
    fv_tiles = []
    indata_len = len(inputdata['data'])
    indata_len = 3

    for i in range(0, indata_len):
        data3d_orig, data3d_seg = read_data_orig_and_seg(inputdata, i)

        feat_hist_by_segmentation(data3d_orig, data3d_seg, visualization)

        if visualization:
            pyed = sed3.sed3(data3d_orig,
                                                 contour=data3d_seg)
            pyed.show()

            #import pdb; pdb.set_trace()
        #fvall.insert(i, get_features(
        #    data3d_orig,
            #ltl = (labels_train_lin_float * 10).astype(np.int8)
            #labels_train = arrange_to_tiled_data(cidxs, tile_shape,
            #                                     d_shp, ltl)

            #pyed = sed3.sed3(labels_train, contour=labels)

# @TODO vracet něco inteligentního, fvall je prázdný
    return fvall


def make_product_list(list_of_feature_fcn, list_of_classifiers):
#   TODO work with list_of_feature_fcn and list_of_classifiers
    featrs_plus_classifs = itertools.product(list_of_feature_fcn,
                                             list_of_classifiers)
    return featrs_plus_classifs


def experiment(path_to_yaml, featrs_plus_classifs,
               tile_shape, visualization=False, train=False):

    inputdata = misc.obj_from_file(path_to_yaml, filetype='yaml')

    #import ipdb; ipdb.set_trace()  # noqa BREAKPOINT

    results = []

    for fpc in featrs_plus_classifs:
        feature_fcn = fpc[0]
        classif_fcn = fpc[1]

        fvall = one_experiment_setting_for_whole_dataset(
            inputdata, tile_shape,
            feature_fcn, classif_fcn, train, visualization)

        result = {'params': str(fpc), 'fvall': fvall}
        results.append(result)
        print(results)

    return results


def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    parser = argparse.ArgumentParser(
        description='Compute features on liver and other tissue.')
    parser.add_argument('-si', '--sampleInput', action='store_true',
                        help='generate sample intput data', default=False)
    parser.add_argument('-v', '--visualization',  action='store_true',
                        help='Turn on visualization', default=False)
    parser.add_argument('-i', '--input', help='input yaml file',
                        default="20130919_liver_statistics.yaml")
    parser.add_argument('-o', '--output', help='output file',
                        default="20130919_liver_statistics_results.pkl")
    parser.add_argument('-t', '--train', help='Training', default=False,
                        action='store_true'
                        )
    args = parser.parse_args()

    if args.sampleInput:
        sample_input_data()
    # input parser
    #path_to_yaml = os.path.join(path_to_script, args.input)
    path_to_yaml = args.input

    #write_csv(fvall)
    list_of_feature_fcn = [feat_hist]
    from sklearn import svm
    from sklearn.naive_bayes import GaussianNB

    list_of_classifiers = [svm.SVC, GaussianNB]
    tile_shape = [1, 100, 100]
    featrs_plus_classifs = make_product_list(list_of_feature_fcn,
                                             list_of_classifiers)

    result = experiment(path_to_yaml, featrs_plus_classifs,
                        tile_shape=tile_shape,
                        visualization=args.visualization, train=args.train)

# Ukládání výsledku do souboru
    output_file = os.path.join(path_to_script, args.output)
    misc.obj_to_file(result, output_file, filetype='pickle')

if __name__ == "__main__":
    main()
