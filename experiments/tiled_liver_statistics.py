# ! /usr/bin/python
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
# sys.path.append(os.path.join(path_to_script, "../extern/"))
# import featurevector

import logging
logger = logging.getLogger(__name__)

import json
# import apdb
#  apdb.set_trace();
# import scipy.io
import numpy as np
import scipy
# from scipy import sparse
# import traceback
import itertools

# ----------------- my scripts --------
import py3DSeedEditor
# import dcmreaddata1 as dcmr
# import dcmreaddata as dcmr
import argparse
# import py3DSeedEditor

import misc
import datareader
import matplotlib.pyplot as plt
import experiments
import texture_features as tfeat


def feat_hist_by_segmentation(data3d_orig, data3d_seg, voxelsize_mm=[1],
                              visualization=True):
    bins = range(-1024, 1024, 1)
    bins = range(-512, 512, 1)
    hist1, bin_edges1 = np.histogram(data3d_orig[data3d_seg > 0], bins=bins)
    hist2, bin_edges2 = np.histogram(data3d_orig[data3d_seg <= 0], bins=bins)
    # import pdb; pdb.set_trace()
    if visualization:
        plt_liver = plt.step(bin_edges1[1:], hist1)
        plt_rest = plt.step(bin_edges2[1:], hist2)
        plt.legend([plt_liver, plt_rest], ['Liver', 'Other tissue'])
        # plt.plot(bin_edges1[1:], hist1, bin_edges2[1:], hist2)
        plt.show()

    vvolume = np.prod(voxelsize_mm)
    fv_hist = {'hist1': hist1 * vvolume,
               'hist2': hist2 * vvolume,
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


def super_feat_hist(data3d_orig, parameter):
    return feat_hist(data3d_orig)
    # print parameter


def f_greycomatrix(data3d):
    import skimage
    skimage.feature.greycomatrix(data3d[:, :, 0], [5], )


def lbp(data3d_orig, data3d_seg, visualization=True):
    import realtime_lbp as real_lib
    realLbp = real_lib.loadRealtimeLbpLibrary()
    lbpRef = np.zeros([1, 256])
    lbpRef = real_lib.realTimeLbpImNp(realLbp, data3d_orig[:, :, 1])
    return lbpRef


def lbp3d(data3d_orig, filename, visualization=True):
    from lbp import lbp3d
    lib3d = lbp3d.load()
    f = open(filename, 'r')
    maskJSON = json.load(f)
    mask = maskJSON['mask']
    lbp3d.coordToPoints(
        mask,
        data3d_orig.shape[2],
        data3d_orig.shape[1])
    res = lbp3d.compute(
        lib3d,
        data3d_orig,
        mask)
    return res


def f_lbp3d(data3d_orig):
    return lbp3d(
        data3d_orig,
        '/home/petr/Dokumenty/git/lbpLibrary/masks/mask3D_8_4.json',
        True)


def get_features(data3d_orig, data3d_seg, feature_fcn, visualization=True):
    u"""
    Sem doplníme všechny naše měření.

    Pro ukázku jsem vytvořil měření
    histogramu.
    data3d_orig: CT data3d_orig
    data3d_seg: jedničky tam, kde jsou játra

    """

    featur = feature_fcn(data3d_orig)
    # featur = {}
    # featur['hist'] = feat_hist(data3d_orig, visualization)
    # featur['lbp'] = lbp(data3d_orig, data3d_seg, visualization)

    return featur


def get_features_in_tiles(data3d_orig, data3d_seg, tile_shape, feature_fcn):
    """
    Computes features for small blocks of image data (tiles).

    cindexes: indexes of tiles
    features_t: numpy array of features
    segmentation_cover: np array with coverages of tiles with segmentation
    (float 0 by 1)

    """
    cindexes = cutter_indexes(data3d_orig.shape, tile_shape)
# create empty list of defined length
    features_t = [None] * len(cindexes)
    seg_cover_t = [None] * len(cindexes)
    print " # ## #    get fv", len(cindexes), " dsh ", data3d_orig.shape
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
    # if (tile_seg == 1).all():
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
            {'sliverseg': 'data_orig/sliver07/training-part1/liver-seg001.mhd', 'sliverorig': 'data_orig/sliver07/training-part1/liver-orig001.mhd'},  # noqa
            {'sliverseg': 'data_orig/sliver07/training-part1/liver-seg002.mhd', 'sliverorig': 'data_orig/sliver07/training-part1/liver-orig002.mhd'},  # noqa
        ]
    }

    sample_data_file = os.path.join(path_to_script,
                                    "20130919_liver_statistics.yaml")
    # print sample_data_file, path_to_script
    misc.obj_to_file(inputdata, sample_data_file, filetype='yaml')

# def voe_metric(vol1, vol2, voxelsize_mm):


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
            # spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])


# def training_dataset_prepare
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

    # import pdb; pdb.set_trace()
    data3d_seg = (data3d_a > 0).astype(np.int8)
    data3d_orig = data3d_b

    return data3d_orig, data3d_seg, metadata_a['voxelsize_mm']


def data_preprocessing(data3d, voxelsize_mm, working_voxelsize_mm):
    # working_voxelsize_mm = np.array([2.0, 1.0, 1.0])
    zoom = voxelsize_mm / (1.0 * working_voxelsize_mm)

    data3d_res = scipy.ndimage.zoom(
        data3d,
        zoom,
        mode='nearest',
        order=1
    ).astype(np.int16)
    return data3d_res, working_voxelsize_mm


def data_postprocessing(segmentation_res, voxelsize_mm, working_voxelsize_mm):
    zoom = voxelsize_mm / (1.0 * working_voxelsize_mm)
    segm_orig_scale = scipy.ndimage.zoom(
        segmentation_res,
        1.0 / zoom,
        mode='nearest',
        order=0
    ).astype(np.int8)
    return segm_orig_scale


def save_labels(
        inputfile,
        data3d,
        segmentation,
        feature_fcn,
        classif_inst,
        voxelsize,
        tile_shape):
    path_directory = 'lisa_data/'
    subdirectory = 'experiments/'
    actual = os.getcwd()
    os.chdir(os.path.expanduser('~'))
    # Ukládání výsledku do souboru
    if(os.path.exists(path_directory) is False):
        os.makedirs(path_directory)
    path_subdirectory = os.path.join(path_directory, subdirectory)
    if(os.path.exists(path_subdirectory) is False):
        os.makedirs(path_subdirectory)
    # TODO : Main Saving Loop ...
    dataplus = []
    slab = {}
    slab['liver'] = 1
    slab['none'] = 0
    dataplus = {
        # 'segmentation': segmentation[:10, :10, :10].astype(np.int8),
        # 'data3d': data3d[:10, :10, :10].astype(np.int16),
        'segmentation': segmentation.astype(np.int8),
        'data3d': data3d.astype(np.int16),
        'processing_info': {
            'feature_fcn': str(feature_fcn),
            'classif_fcn': str(classif_inst.__class__.__name__)
        },
        'voxelsize_mm': voxelsize,
        'slab': slab}
    # inputfilename = path_leaf(inputfile)
    filename = feature_fcn.__name__ + '_' + \
        classif_inst.__class__.__name__ + '_' + inputfile
    filename = filename + '_' + \
        str(tile_shape[0]) + '_' + str(tile_shape[1]) + '_'
    filename = filename + str(tile_shape[2]) + '.pklz'
    path_to_file = os.path.join(path_subdirectory, filename)
    misc.obj_to_file(dataplus, path_to_file, filetype='pklz')
    os.chdir(actual)


def one_experiment_setting_training(inputdata, tile_shape,
                                    feature_fcn, classif_fcn,
                                    visualization=False):
    """
    Training of experiment.
    """
    features_t_all = []
    labels_train_lin_all = []
    indata_len = len(inputdata['data'])
    features_t_all = []
    # indata_len = 3

    for i in range(0, indata_len):
        data3d_orig, data3d_seg, voxelsize_mm = read_data_orig_and_seg(
            inputdata, i)

        if visualization:
            pyed = py3DSeedEditor.py3DSeedEditor(data3d_orig,
                                                 contour=data3d_seg)
            pyed.show()
        fv_t = get_features_in_tiles(data3d_orig, data3d_seg, tile_shape,
                                     feature_fcn)
        cidxs, features_t, seg_cover_t = fv_t
        labels_train_lin_float = np.array(seg_cover_t)
        labels_train_lin = (
            labels_train_lin_float > 0.5).astype(np.int8).tolist()

        features_t_all = features_t_all + features_t
        labels_train_lin_all = labels_train_lin_all + labels_train_lin
    clf = classif_fcn()
    clf.fit(features_t_all, labels_train_lin_all)
    # import ipdb; ipdb.set_trace()  # noqa BREAKPOINT
    return clf


def one_experiment_setting_testing(inputdata, tile_shape,
                                   feature_fcn, clf,
                                   visualization=False):
    indata_len = len(inputdata['data'])
    # indata_len = 3

    for i in range(0, indata_len):
        data3d_orig, data3d_seg, voxelsize_mm = read_data_orig_and_seg(
            inputdata, i)

        if visualization:
            pyed = py3DSeedEditor.py3DSeedEditor(data3d_orig,
                                                 contour=data3d_seg)
            pyed.show()
        fv_t = get_features_in_tiles(data3d_orig, data3d_seg, tile_shape,
                                     feature_fcn)
        cidxs, features_t, seg_cover_t = fv_t

        # labels_train_lin_float = np.array(seg_cover_t)
        # labels_train_lin = labels_train_lin_float > 0.5
        labels_lin = clf.predict(features_t)

        d_shp = data3d_orig.shape

        segmentation = arrange_to_tiled_data(cidxs, tile_shape, d_shp,
                                             labels_lin)
# @TODO změnil jsem to. Už zde není ukazatel na klasifikátor, ale přímo
# natrénovaný klasifikátor.
        save_labels(
            inputdata['data'][i]['sliverorig'], data3d_orig, segmentation,
            feature_fcn, clf, voxelsize_mm, tile_shape)
        # ltl = (labels_train_lin_float * 10).astype(np.int8)
        # labels_train = arrange_to_tiled_data(cidxs, tile_shape,
        #                                     d_shp, ltl)

        # pyed = py3DSeedEditor.py3DSeedEditor(labels_train, contour=labels)
        if visualization:
            pyed = py3DSeedEditor.py3DSeedEditor(
                data3d_seg,
                contour=segmentation)
            pyed.show()
    pass


def one_experiment_setting_for_whole_dataset(training_yaml, testing_yaml,
                                             tile_shape, feature_fcn,
                                             classif_fcn, train,
                                             visualization=False):
    fvall = []
    # fv_tiles = []
    clf = one_experiment_setting_training(training_yaml, tile_shape,
                                          feature_fcn, classif_fcn,
                                          visualization=False)

    logger.info('run testing')
    one_experiment_setting_testing(testing_yaml, tile_shape,
                                   feature_fcn, clf,
                                   visualization=visualization)

# @TODO vracet něco inteligentního, fvall je prázdný
    return fvall


def make_product_list(list_of_feature_fcn, list_of_classifiers):
    #   TODO work with list_of_feature_fcn and list_of_classifiers
    featrs_plus_classifs = itertools.product(list_of_feature_fcn,
                                             list_of_classifiers)
    return featrs_plus_classifs


def experiment(training_yaml_path, testing_yaml_path,  featrs_plus_classifs,
               tile_shape, visualization=False, train=False):

    training_yaml = misc.obj_from_file(training_yaml_path, filetype='yaml')
    testing_yaml = misc.obj_from_file(testing_yaml_path, filetype='yaml')

    # import ipdb; ipdb.set_trace()  # noqa BREAKPOINT

    results = []

    for fpc in featrs_plus_classifs:
        feature_fcn = fpc[0]
        classif_fcn = fpc[1]

        fvall = one_experiment_setting_for_whole_dataset(
            training_yaml, testing_yaml, tile_shape,
            feature_fcn, classif_fcn, train, visualization)

        result = {'params': str(fpc), 'fvall': fvall}
        results.append(result)
        print results

    return results


def main():

    # logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # logger.debug('input params')

    parser = argparse.ArgumentParser(
        description='Compute features on liver and other tissue.')
    parser.add_argument('-tr', '--training_yaml_path', help='input yaml file',
                        default="20130919_liver_statistics.yaml")
    parser.add_argument('-te', '--testing_yaml_path', help='input yaml file',
                        default=None)
    parser.add_argument('-si', '--sampleInput', action='store_true',
                        help='generate sample intput data', default=False)
    parser.add_argument('-v', '--visualization',  action='store_true',
                        help='Turn on visualization', default=False)
    parser.add_argument('-fc', '--features_classifs',  action='store_true',
                        help='Read features and classifs list from file',
                        default=False)
    parser.add_argument('-o', '--output', help='output file',
                        default="20130919_liver_statistics_results.pkl")
    parser.add_argument('-t', '--train', help='Training', default=False,
                        action='store_true'
                        )
    args = parser.parse_args()

    if args.sampleInput:
        sample_input_data()
    # input parser
    # path_to_yaml = os.path.join(path_to_script, args.input)
    # training_yaml_path = args.training_yaml_path
    # testing_yaml_path = args.testing_yaml_path
    if args.testing_yaml_path is None:
        print 'testing is same as training'
        args.testing_yaml_path = args.training_yaml_path

    # write_csv(fvall)
    # gf = tfeat.GaborFeatures()
    glcmf = tfeat.GlcmFeatures()

    list_of_feature_fcn = [
        # feat_hist,
        # gf.feats_gabor
        glcmf.feats_glcm
    ]
    from sklearn import svm
    from sklearn.naive_bayes import GaussianNB

    list_of_classifiers = [
            GaussianNB,
            svm.SVC
            ]
    tile_shape = [10, 50, 50]

    if args.features_classifs:
        import features_classifs
        featrs_plus_classifs = features_classifs.fc
    else:
        featrs_plus_classifs = make_product_list(list_of_feature_fcn,
                                                 list_of_classifiers)

    result = experiment(args.training_yaml_path, args.testing_yaml_path,
                        featrs_plus_classifs, tile_shape=tile_shape,
                        visualization=args.visualization, train=args.train)

# Ukládání výsledku do souboru
    output_file = os.path.join(path_to_script, args.output)
    misc.obj_to_file(result, output_file, filetype='pickle')

if __name__ == "__main__":
    main()
