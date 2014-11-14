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
sys.path.append(os.path.join(path_to_script, "../lisa/"))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/sed3/"))
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
import sed3
# import dcmreaddata1 as dcmr
# import dcmreaddata as dcmr
import argparse
# import sed3

import misc
import qmisc
from io3d import datareader
import matplotlib.pyplot as plt
from lisa import experiments
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
    # bins = range(-1024, 1024, 1)
    # bins = range(-512, 512, 1)
    # bins = range(-512, 512, 10)
    # bins = range(-512, 512, 64)
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


def get_features(data3d_orig, data3d_seg, feature_fcn, feature_fcn_params=[],
                 visualization=True):
    u"""
    Sem doplníme všechny naše měření.

    Pro ukázku jsem vytvořil měření
    histogramu.
    data3d_orig: CT data3d_orig
    data3d_seg: jedničky tam, kde jsou játra
    feature_fcn_plus_params: list of size 2: [feature_function, [3, 'real']]

    There are two possible feature_fcn formats
     * method
     * object with method 'features' and attribute 'description'
    """
    # feature_fcn, feature_fcn_params = feature_fcn_plus_params

    if hasattr(feature_fcn, 'description'):
        feature_fcn = feature_fcn.features

    featur = feature_fcn(data3d_orig, *feature_fcn_params)
    # featur = {}
    # featur['hist'] = feat_hist(data3d_orig, visualization)
    # featur['lbp'] = lbp(data3d_orig, data3d_seg, visualization)

    return featur


def get_features_in_tiles(
        data3d_orig,
        data3d_seg,
        tile_shape,
        feature_fcn,
        feature_fcn_params):
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
    logger.debug(
        " ##    get fv" + str(len(cindexes)) + " dsh " +
        str(data3d_orig.shape) + 'tile_shape ' + str(tile_shape))
    for i in range(0, len(cindexes)):
        cindex = cindexes[i]
        tile_orig = experiments.getArea(data3d_orig, cindex, tile_shape)
        tile_seg = experiments.getArea(data3d_seg, cindex, tile_shape)
        tf = get_features(tile_orig, tile_seg, feature_fcn, feature_fcn_params,
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
    # r0 = range(0, shape[0] - tile_shape[0] + 1, tile_shape[0])
    # r1 = range(1, shape[1] - tile_shape[1] + 1, tile_shape[1])
    # r2 = range(2, shape[2] - tile_shape[2] + 1, tile_shape[2])
    r0 = range(0, shape[0], tile_shape[0])
    r1 = range(0, shape[1], tile_shape[1])
    r2 = range(0, shape[2], tile_shape[2])
    cut_iterator = itertools.product(r0[:-1], r1[:-1], r2[:-1])
    return list(cut_iterator)


# @TODO dodělat rozsekávač
def cut_tile(data3d, cindex, tile_shape):
    """ Function is similar to experiments.getArea(). """

    upper_corner = cindex + np.array(tile_shape)
    # print cindex, "    tile shape ", tile_shape, ' uc ', upper_corner,\
    #    ' dsh ', data3d.shape

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
    # @TODO unused delete it
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
    # @TODO unused delete it
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
        feature_fcn_params,
        classif_inst,
        classif_fcn_plus_params,
        voxelsize,
        tile_shape,
        use_voxelsize_norm
        ):
    """
    classif_fcn_plus_params: used for directory name
    """
    path_directory = 'lisa_data/'
    subdirectory = 'experiments/'
    # actual = os.getcwd()
    path_directory = os.path.join(os.path.expanduser('~'), path_directory)
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

# there are two possible ways for fature_fcn. One is function, other is object
# with "description" attribute and function 'features'
    if hasattr(feature_fcn, 'description'):
        feature_fcn_description = feature_fcn.description
    else:
        feature_fcn_description = str(feature_fcn)

    dataplus = {
        # 'segmentation': segmentation[:10, :10, :10].astype(np.int8),
        # 'data3d': data3d[:10, :10, :10].astype(np.int16),
        'segmentation': segmentation.astype(np.int8),
        'data3d': data3d.astype(np.int16),
        'processing_information': {
            'feature_fcn': feature_fcn_description,
            'feature_fcn_params': str(feature_fcn_params),
            'classif_fcn_name': str(classif_inst.__class__.__name__),
            'classif_fcn': str(classif_inst)
        },
        'voxelsize_mm': voxelsize,
        'slab': slab}
    # inputfilename = path_leaf(inputfile)
    str_feat_params = __struct_to_string_for_filename(feature_fcn_params)
    str_clf_params = __struct_to_string_for_filename(str(classif_inst))
    str_clf_params = __struct_to_string_for_filename(
        str(classif_fcn_plus_params))

    vsnorm_string = "vsnF_"
    if use_voxelsize_norm:
        vsnorm_string = "vsnT_"

# construct filename
    experiment_dirname =\
        feature_fcn_description + '_' + \
        str_feat_params + '_' +\
        vsnorm_string +\
        classif_inst.__class__.__name__ + '_' + \
        str_clf_params + '_' +\
        str(tile_shape[0]) + '_' + str(tile_shape[1]) + '_' +\
        str(tile_shape[2])

    filename = __struct_to_string_for_filename(inputfile + '.pklz')

    path_to_file = os.path.join(path_subdirectory,
                                experiment_dirname, filename)
    misc.obj_to_file(dataplus, path_to_file, filetype='pklz')
    # os.chdir(actual)


# save info to dir
    info = {
        'feature_fcn': feature_fcn_description,
        'feature_fcn_params': str(feature_fcn_params),
        'classif_fcn': str(classif_inst)
        }
    infofilepath = os.path.join(path_subdirectory,
                                experiment_dirname, 'info.yaml')
    misc.obj_to_file(info, infofilepath, 'yaml')


def norm_voxelsize(data3d_orig, data3d_seg, voxelsize_mm,
                   working_voxelsize_mm):

    data3d_orig_res = qmisc.resize_to_mm(data3d_orig,
                                         voxelsize_mm, working_voxelsize_mm)
    data3d_seg_res = qmisc.resize_to_mm(data3d_seg,
                                        voxelsize_mm, working_voxelsize_mm)

    return data3d_orig_res, data3d_seg_res


def one_exp_set_training(inputdata, tile_shape,
                         feature_fcn_plus_params,
                         classif_fcn_plus_params,
                         use_voxelsize_norm=False,
                         working_voxelsize_mm=[1, 1, 1],
                         visualization=False):
    """
    Training of experiment.
    """
    features_t_all = []
    labels_train_lin_all = []
    indata_len = len(inputdata['data'])
    features_t_all = []
    # indata_len = 3
    classif_fcn, classif_fcn_params = classif_fcn_plus_params
    logger.debug('number of data files ' + str(indata_len))

    for i in range(0, indata_len):
        data3d_orig, data3d_seg, voxelsize_mm = read_data_orig_and_seg(
            inputdata, i)

        if use_voxelsize_norm:
            data3d_orig, data3d_seg = norm_voxelsize(
                data3d_orig, data3d_seg,
                voxelsize_mm, working_voxelsize_mm)

        if visualization:
            pyed = sed3.sed3(data3d_orig,
                                                 contour=data3d_seg)
            pyed.show()
        logger.debug('data shape ' + str(data3d_orig.shape))
        fv_t = get_features_in_tiles(data3d_orig, data3d_seg, tile_shape,
                                     feature_fcn_plus_params[0],
                                     feature_fcn_plus_params[1])
        cidxs, features_t, seg_cover_t = fv_t
        labels_train_lin_float = np.array(seg_cover_t)
        labels_train_lin = (
            labels_train_lin_float > 0.5).astype(np.int8).tolist()

        features_t_all = features_t_all + features_t
        labels_train_lin_all = labels_train_lin_all + labels_train_lin

    # if there are named variables use dict unpacking
    if isinstance(classif_fcn_params, dict):
        clf = classif_fcn(**classif_fcn_params)
    else:
        clf = classif_fcn(*classif_fcn_params)

    clf.fit(features_t_all, labels_train_lin_all)
    # import ipdb; ipdb.set_trace()  # noqa BREAKPOINT
    return clf


def __struct_to_string_for_filename(params):
    sarg = str(params)
    sarg = sarg.replace('{', '')
    sarg = sarg.replace('}', '')
    sarg = sarg.replace('[', '')
    sarg = sarg.replace(']', '')
    sarg = sarg.replace('<', '')
    sarg = sarg.replace('>', '')
    sarg = sarg.replace('"', '')
    sarg = sarg.replace("'", '')
    sarg = sarg.replace(" ", '')
    sarg = sarg.replace(",", '_')
    sarg = sarg.replace(":", '-')
    sarg = sarg.replace("\\", '-')
    sarg = sarg.replace("/", '-')
    if len(sarg) > 100:
        sarg = sarg[:100]
    return sarg


def one_exp_set_testing(
        inputdata, tile_shape,
        feature_fcn_plus_params,
        clf,
        classif_fcn_plus_params,
        use_voxelsize_norm=False,
        working_voxelsize_mm=[1, 1, 1],
        visualization=False):
    indata_len = len(inputdata['data'])
    # indata_len = 3
    feature_fcn, feature_fcn_params = feature_fcn_plus_params

    for i in range(0, indata_len):
        data3d_orig_orig, data3d_seg_orig, voxelsize_mm = \
            read_data_orig_and_seg(inputdata, i)
        data3d_orig = data3d_orig_orig
        data3d_seg = data3d_seg_orig

        if use_voxelsize_norm:
            data3d_orig, data3d_seg = norm_voxelsize(
                data3d_orig_orig, data3d_seg_orig,
                voxelsize_mm, working_voxelsize_mm)

        if visualization:
            pyed = sed3.sed3(data3d_orig,
                                                 contour=data3d_seg)
            pyed.show()

        fv_t = get_features_in_tiles(data3d_orig, data3d_seg, tile_shape,
                                     feature_fcn,
                                     feature_fcn_params)
        cidxs, features_t, seg_cover_t = fv_t
        # we are ignoring seg_cover_t which gives us information about
        # segmentation

        labels_lin = clf.predict(features_t)

        d_shp = data3d_orig.shape

        segmentation = arrange_to_tiled_data(cidxs, tile_shape, d_shp,
                                             labels_lin)

        if use_voxelsize_norm:
            segmentation = qmisc.resize_to_shape(
                segmentation,
                data3d_orig_orig.shape)
# @TODO změnil jsem to. Už zde není ukazatel na klasifikátor, ale přímo
# natrénovaný klasifikátor.
        save_labels(
            inputdata['data'][i]['sliverorig'], data3d_orig_orig, segmentation,
            feature_fcn, feature_fcn_params, clf,
            classif_fcn_plus_params, voxelsize_mm, tile_shape,
            use_voxelsize_norm)
        # ltl = (labels_train_lin_float * 10).astype(np.int8)
        # labels_train = arrange_to_tiled_data(cidxs, tile_shape,
        #                                     d_shp, ltl)

        # pyed = sed3.sed3(labels_train, contour=labels)
        if visualization:
            pyed = sed3.sed3(
                data3d_seg,
                contour=segmentation)
            pyed.show()
    pass


def one_exp_set_for_whole_dataset(
        training_yaml,
        testing_yaml,
        tile_shape,
        feature_fcn_plus_params,
        classif_fcn_plus_params,
        train,
        use_voxelsize_norm,
        working_voxelsize_mm,
        visualization=False):
    fvall = []
    # fv_tiles = []
    print "classif_fcn ", classif_fcn_plus_params
    print "feature_fcn ", feature_fcn_plus_params
    clf = one_exp_set_training(
        training_yaml, tile_shape,
        feature_fcn_plus_params,
        classif_fcn_plus_params,
        use_voxelsize_norm,
        working_voxelsize_mm,
        visualization=False)

    logger.info('run testing')
    one_exp_set_testing(testing_yaml, tile_shape,
                        feature_fcn_plus_params, clf,
                        classif_fcn_plus_params,
                        use_voxelsize_norm,
                        working_voxelsize_mm,
                        visualization=visualization)

# @TODO vracet něco inteligentního, fvall je prázdný
    return fvall


def make_product_list(list_of_feature_fcn, list_of_classifiers):
    #   TODO work with list_of_feature_fcn and list_of_classifiers
    featrs_plus_classifs = itertools.product(list_of_feature_fcn,
                                             list_of_classifiers)
    return featrs_plus_classifs


def experiment(training_yaml_path, testing_yaml_path,  featers_plus_classifs,
               tile_shape,
               use_voxelsize_norm,
               working_voxelsize_mm,
               visualization=False, train=False):

    training_yaml = misc.obj_from_file(training_yaml_path, filetype='yaml')
    testing_yaml = misc.obj_from_file(testing_yaml_path, filetype='yaml')

    # import ipdb; ipdb.set_trace()  # noqa BREAKPOINT

    results = []

    for fpc in featers_plus_classifs:
        feature_fcn = fpc[0]
        classif_fcn = fpc[1]

        fvall = one_exp_set_for_whole_dataset(
            training_yaml, testing_yaml, tile_shape,
            feature_fcn, classif_fcn,
            train,
            use_voxelsize_norm,
            working_voxelsize_mm,
            visualization)

        result = {'params': str(fpc), 'fvall': fvall}
        results.append(result)
        print results

    return results


def prepared_classifiers_by_string(names):
    from sklearn import svm
    from sklearn.naive_bayes import GaussianNB  # noqa
    from sklearn.mixture import GMM  # noqa
    from sklearn import tree  # noqa
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier  # noqa
    from sklearn.lda import LDA  # noqa
    from sklearn.qda import QDA  # noqa
    import classification
    dict_of_classifiers = {
        'GaussianNB': [GaussianNB, []],
        'SVC': [svm.SVC, []],
        'DT': [tree.DecisionTreeClassifier, []],
        'GMM2': [classification.GMMClassifier,
                 {'n_components': 2, 'covariance_type': 'full'}],
        'SVClin': [svm.SVC, {'kernel': 'linear'}],
        'SVCrbf': [svm.SVC, {'kernel': 'rbf'}],
        'SVCpoly': [svm.SVC, {'kernel': 'poly'}],
        'RF': [RandomForestClassifier, []],
        'LDA': [LDA, []],
        'QDA': [QDA, []],
        }

    selected_classifiers = [dict_of_classifiers[name] for name in names]
    return selected_classifiers


def prepared_texture_features_by_string(names):
    """
    There are two possible feature_fcn formats
     * method
     * object with method 'features' and attribute 'description'
    """
    gf = tfeat.GaborFeatures()  # noqa
    glcmf = tfeat.GlcmFeatures()  # noqa
    haralick = tfeat.HaralickFeatures()  # noqa
    hist_gf = tfeat.FeaturesCombinedFeatures(feat_hist, gf.feats_gabor)
    hist_glcm = tfeat.FeaturesCombinedFeatures(feat_hist, glcmf.feats_glcm)
    hist_glcm_gf = tfeat.FeaturesCombinedFeatures(
        feat_hist,
        glcmf.feats_glcm,
        gf.feats_gabor
    )

    dict_of_feature_fcn = {
        'hist': [feat_hist, []],
        'gf': [gf.feats_gabor, []],
        'glcm': [glcmf.feats_glcm, []],
        'haralick': [haralick.feats_haralick, [True]],
        'hist_gf': [hist_gf, []],
        'hist_glcm': [hist_glcm, []],
        'hist_glcm_gf': [hist_glcm_gf, []],
    }

    selected_classifiers = [dict_of_feature_fcn[name] for name in names]
    return selected_classifiers


def main():

    # logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # logger.debug('input params')

    parser = argparse.ArgumentParser(
        description='Compute features on liver and other tissue.')
    parser.add_argument('-tr', '--training_yaml_path',
                        help='Input yaml file.' +
                        " You can check sample with -si parameter.",
                        default="20130919_liver_statistics.yaml")
    parser.add_argument('-te', '--testing_yaml_path', help='Input yaml file.' +
                        " You can check sample with -si parameter.",
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
    parser.add_argument(
        '-cl', '--classifers',
        help='classifer by string: "SVC", or "GaussianNB", ...',
        nargs='+', type=str, default=['SVC']
    )
    parser.add_argument(
        '-fe', '--features',
        help='features by string: "hist", or "glcm", ...',
        nargs='+', type=str, default=['hist']
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
    gf = tfeat.GaborFeatures()  # noqa
    glcmf = tfeat.GlcmFeatures()  # noqa
    haralick = tfeat.HaralickFeatures()  # noqa

    list_of_feature_fcn = prepared_texture_features_by_string(args.features)

    list_of_classifiers = prepared_classifiers_by_string(args.classifers)
    tile_shape = [10, 50, 50]

    if args.features_classifs:
        import features_classifs
        featrs_plus_classifs = features_classifs.fc
    else:
        featrs_plus_classifs = make_product_list(list_of_feature_fcn,
                                                 list_of_classifiers)

    result = experiment(args.training_yaml_path, args.testing_yaml_path,
                        featrs_plus_classifs, tile_shape=tile_shape,
                        use_voxelsize_norm=True,
                        working_voxelsize_mm=[1, 1, 1],
                        visualization=args.visualization, train=args.train)

# Ukládání výsledku do souboru
    output_file = os.path.join(path_to_script, args.output)
    misc.obj_to_file(result, output_file, filetype='pickle')

if __name__ == "__main__":
    main()
