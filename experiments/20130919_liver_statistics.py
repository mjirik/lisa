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
                             "../extern/py3DSeedEditor/"))
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
import py3DSeedEditor
#import dcmreaddata1 as dcmr
#import dcmreaddata as dcmr
import argparse
#import py3DSeedEditor

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


def lbp(data3d_orig, data3d_seg, visualization=True):
    import realtime_lbp as real_lib
    realLbp = real_lib.loadRealtimeLbpLibrary()
    lbpRef = np.zeros([1, 256])
    lbpRef = real_lib.realTimeLbpImNp(realLbp, data3d_orig[:, :, 1])
    return lbpRef


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


def get_features_in_tiles(data3d_orig, data3d_seg, tile_shape, feature_fcn):
    """
    Computes features for small blocks of image data (tiles).

    cindexes: indexes of tiles
    features_t: numpy array of features
    segmentation_cover: np array with coverages of tiles with segmentation
    (float 0 by 1)

    """
# @TODO here
    cindexes = cutter_indexes(data3d_orig.shape, tile_shape)
# create empty list of defined length
    features_t = [None] * len(cindexes)
    seg_cover_t = [None] * len(cindexes)
    print " ####    get fv", len(cindexes), " dsh ", data3d_orig.shape
    for i in range(0, len(cindexes)):
        cindex = cindexes[i]
        tile_orig = experiments.getArea(data3d_orig, cindex, tile_shape)
        tile_seg = experiments.getArea(data3d_seg, cindex, tile_shape)
        tf = get_features(tile_orig, tile_seg, feature_fcn,
                          visualization=False)
        sc = np.sum(tile_seg > 0).astype(np.float) / np.prod(tile_shape)
        features_t[i] = tf
        seg_cover_t[i] = sc
    return cindexes, features_t, seg_cover_t
        #if (tile_seg == 1).all():
        #    pass


def cutter_indexes(shape, tile_shape):
    """
    Make indexes for cutting.

    shape: shape of cutted data
    tile_shape: shape of tile

    """
    # TODO přepis r1?
    r0 = range(0, shape[0] - tile_shape[0] + 1, tile_shape[0])
    r1 = range(1, shape[1] - tile_shape[1] + 1, tile_shape[1])
    r2 = range(2, shape[2] - tile_shape[2] + 1, tile_shape[2])
    r1 = range(0, shape[1], tile_shape[1])
    r2 = range(0, shape[2], tile_shape[2])
    cut_iterator = itertools.product(r0[:-1], r1[:-1], r2[:-1])
    return list(cut_iterator)


# @TODO dodělat rozsekávač
def cut_tile(data3d, cindex, tile_shape):
    """ Function is similar to experiments.getArea(). """

    upper_corner = cindex + np.array(tile_shape)
    print cindex, "    tile shape ", tile_shape, ' uc ', upper_corner,\
        ' dsh ', data3d.shape

    return data3d[cindex[0]:upper_corner[0],
                  cindex[1]:upper_corner[1],
                  cindex[2]:upper_corner[2]
                  ]


def arrange_to_tiled_data(cindexes, tile_shape, data3d_shape, labels_lin):
    """ Creates 3D image with values of labels.  """

    labels = np.zeros(data3d_shape, dtype=type(labels_lin[0]))
    for i in range(0, len(cindexes)):
        cindex = cindexes[i]

# TODO labels shape
        labels = experiments.setArea(labels, cindex, tile_shape, labels_lin[i])
    return labels


def generate_input_yaml_metadata():
    pass


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
    data3d_a, metadata_a = reader.Get3DData(data3d_a_path)

    data3d_b_path = os.path.join(inputdata['basedir'],
                                 inputdata['data'][i]['sliverorig'])
    data3d_b, metadata_b = reader.Get3DData(data3d_b_path)

    #import pdb; pdb.set_trace()
    data3d_seg = (data3d_a > 0).astype(np.int8)
    data3d_orig = data3d_b

    return data3d_orig, data3d_seg


def experiment(path_to_yaml, list_of_feature_fcn, list_of_classifiers,
               tile_shape, visualization=False, test=False):

    inputdata = misc.obj_from_file(path_to_yaml, filetype='yaml')

#   TODO work with list_of_feature_fcn and list_of_classifiers
    featrs_plus_classifs = itertools.product(list_of_feature_fcn,
                                             list_of_classifiers)
    import ipdb; ipdb.set_trace()  # noqa BREAKPOINT

    results = []

    for fpc in featrs_plus_classifs:
        fpc
        fvall = []
        fv_tiles = []
        indata_len = len(inputdata['data'])
        indata_len = 3

        for i in range(0, indata_len):
            data3d_orig, data3d_seg = read_data_orig_and_seg(inputdata, i)

            if visualization:
                pyed = py3DSeedEditor.py3DSeedEditor(data3d_orig,
                                                     contour=data3d_seg)
                pyed.show()
                #import pdb; pdb.set_trace()
            #fvall.insert(i, get_features(
            #    data3d_orig,
            #    data3d_seg,
            #    visualization=args.visualization
            #    ))
            #feature_fcn = feat_hist
            feature_fcn = fpc[0]
            fv_t = get_features_in_tiles(data3d_orig, data3d_seg, tile_shape,
                                         feature_fcn)
            cidxs, features_t, seg_cover_t = fv_t

            labels_train_lin_float = np.array(seg_cover_t)
            labels_train_lin = labels_train_lin_float > 0.5

            #from sklearn import svm
            #clf = svm.SVC()
						if(test == True)
              clf = fpc[1]()
              clf.fit(features_t, labels_train_lin)
              labels_lin = clf.predict(features_t)

              d_shp = data3d_orig.shape

              labels = arrange_to_tiled_data(cidxs, tile_shape, d_shp,
                                           labels_lin)
              #ltl = (labels_train_lin_float * 10).astype(np.int8)
              #labels_train = arrange_to_tiled_data(cidxs, tile_shape,
              #                                     d_shp, ltl)

              #pyed = py3DSeedEditor.py3DSeedEditor(labels_train, contour=labels)
              pyed = py3DSeedEditor.py3DSeedEditor(data3d_seg, contour=labels)
              pyed.show()
            fv_tiles.insert(i, fv_t)

        result = {'params': str(fpc), 'fvall': fvall}
        results.append(result)
        print results

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
		parser.add_argument('-t', '--test', help='Testing', default=False)
    args = parser.parse_args()

    if args.sampleInput:
        sample_input_data()
    # input parser
    path_to_yaml = os.path.join(path_to_script, args.input)

    #write_csv(fvall)
    list_of_feature_fcn = [feat_hist]
    from sklearn import svm
    from sklearn.naive_bayes import GaussianNB

    list_of_classifiers = [svm.SVC, GaussianNB]
    tile_shape = [1, 100, 100]
    result = experiment(path_to_yaml, list_of_feature_fcn, list_of_classifiers,
                        tile_shape=tile_shape,
                        visualization=args.visualization, test=args.test)

# Ukládání výsledku do souboru
    output_file = os.path.join(path_to_script, args.output)
    misc.obj_to_file(result, output_file, filetype='pickle')

if __name__ == "__main__":
    main()
