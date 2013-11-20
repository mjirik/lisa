#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Experiment s 'barevným' modelem jater. 
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
import itertools

# ----------------- my scripts --------
import py3DSeedEditor
#import dcmreaddata1 as dcmr
import dcmreaddata as dcmr
import argparse
#import py3DSeedEditor

import misc
import datareader
import matplotlib.pyplot as plt
import realtime_lbp as real_lib
import experiments




def feat_hist_by_segmentation(data3d_orig, data3d_seg, visualization=True):
    bins = range(-1024,1024,1)
    bins = range(-512,512,1)
    hist1, bin_edges1 = np.histogram(data3d_orig[data3d_seg>0], bins=bins)
    hist2, bin_edges2 = np.histogram(data3d_orig[data3d_seg<=0], bins=bins)
    #import pdb; pdb.set_trace()
    if visualization:
        plt_liver = plt.step(bin_edges1[1:], hist1)
        plt_rest = plt.step(bin_edges2[1:], hist2)
        plt.legend([plt_liver, plt_rest], ['Liver', 'Other tissue'])
        #plt.plot(bin_edges1[1:], hist1, bin_edges2[1:], hist2)
        plt.show()
    fv_hist = {
            'hist1':hist1,
            'hist2':hist2,
            'bins':bins
            }
    return fv_hist


def feat_hist(data3d_orig, visualization=True):
    bins = range(-1024,1024,1)
    bins = range(-512,512,1)
    bins = range(-512,512,10)
    hist1, bin_edges1 = np.histogram(data3d_orig, bins=bins)
    return hist1



def lbp(data3d_orig, data3d_seg, visualization=True):
    realLbp = real_lib.loadRealtimeLbpLibrary()  
    lbpRef = np.zeros([1,256])
    lbpRef = real_lib.realTimeLbpImNp(realLbp,data3d_orig[:,:,1])
    return lbpRef



def get_features(data3d_orig, data3d_seg, visualization=True):
    """
    Sem doplníme všechny naše měření. Pro ukázku jsem vytvořil měření
    histogramu.
    data3d_orig: CT data3d_orig
    data3d_seg: jedničky tam, kde jsou játra
    """
    featur = feat_hist(data3d_orig, visualization)
    #featur = {}
    #featur['hist'] = feat_hist(data3d_orig, visualization)
    #featur['lbp'] = lbp(data3d_orig, data3d_seg, visualization)




    return featur


def get_features_in_tiles(data3d_orig, data3d_seg, tile_shape):
    """
    cindexes: indexes of tiles
    features_t: numpy array of features
    segmentation_cover: np array with coverages of tiles with segmentation 
    (float 0 by 1)

    """
# @TODO here
    cindexes = cutter_indexes(data3d_orig.shape, tile_shape)
    output = {}
# create empty list of defined length
    features_t = [None]*len(cindexes)
    seg_cover_t = [None]*len(cindexes)
    print " ####    get fv", len(cindexes), " dsh ", data3d_orig.shape
    for i in range(0, len(cindexes)):
        cindex = cindexes[i]
        tile_orig = experiments.getArea(data3d_orig, cindex, tile_shape)
        tile_seg = experiments.getArea(data3d_seg, cindex, tile_shape)
        tf = get_features(tile_orig, tile_seg, visualization=False)
        sc = np.sum(tile_seg > 0).astype(np.float) / np.prod(tile_shape)
        features_t[i] = tf
        seg_cover_t[i] = sc
    return cindexes, features_t, seg_cover_t
        #if (tile_seg == 1).all(): 
        #    pass



def cutter_indexes(shape, tile_shape):
    """
    shape: shape of cutted data
    tile_shape: shape of tile

    """
    r0 = range(0,shape[0]-tile_shape[0] + 1, tile_shape[0])
    r1 = range(1,shape[1]-tile_shape[1] + 1, tile_shape[1])
    r2 = range(2,shape[2]-tile_shape[2] + 1, tile_shape[2])
    r1 = range(0,shape[1], tile_shape[1])
    r2 = range(0,shape[2], tile_shape[2])
    cut_iterator = itertools.product(r0[:-1], r1[:-1], r2[:-1])
    return list(cut_iterator)



# @TODO dodělat rozsekávač
def cut_tile(data3d, cindex, tile_shape ):
    """
    Function is similar to experiments.getArea()
    """
    upper_corner = cindex + np.array(tile_shape)
    print cindex, "    tile shaoe ", tile_shape, ' uc ', upper_corner, ' dsh ', data3d.shape

    return data3d[
            cindex[0]:upper_corner[0],
            cindex[1]:upper_corner[1],
            cindex[2]:upper_corner[2]
            ]








def sample_input_data():
    inputdata = {'basedir':'/home/mjirik/data/medical/',
            'data': [
                {'sliverseg':'data_orig/sliver07/training-part1/liver-seg001.mhd', 'sliverorig':'data_orig/sliver07/training-part1/liver-orig001.mhd'},
                {'sliverseg':'data_orig/sliver07/training-part1/liver-seg002.mhd', 'sliverorig':'data_orig/sliver07/training-part1/liver-orig002.mhd'},
                ]
            }


    sample_data_file = os.path.join(path_to_script, "20130919_liver_statistics.yaml")
    #print sample_data_file, path_to_script
    misc.obj_to_file(inputdata, sample_data_file, filetype='yaml')

#def voe_metric(vol1, vol2, voxelsize_mm):



def write_csv(data, filename= "20130919_liver_statistics.yaml"):
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


#def training_dataset_prepare


def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    parser = argparse.ArgumentParser(
            description='Compute features on liver and other tissue.' )
    parser.add_argument('-si', '--sampleInput',  action='store_true',
            help='generate sample intput data', default=False)
    parser.add_argument('-v', '--visualization',  action='store_true',
            help='Turn on visualization', default=False)
    args = parser.parse_args()

    if args.sampleInput:
        sample_input_data()
    # input parser
    data_file = os.path.join(path_to_script,  "20130919_liver_statistics.yaml")
    inputdata = misc.obj_from_file(data_file, filetype='yaml')
    

    fvall = []
    fv_tiles = []
    indata_len = len(inputdata['data'])
    indata_len = 3
    
    for i in range(0, indata_len):

        reader = datareader.DataReader()
        data3d_a_path = os.path.join(inputdata['basedir'], inputdata['data'][i]['sliverseg'])
        data3d_a, metadata_a = reader.Get3DData(data3d_a_path)


        data3d_b_path = os.path.join(inputdata['basedir'], inputdata['data'][i]['sliverorig'])
        data3d_b, metadata_b = reader.Get3DData(data3d_b_path)



        #import pdb; pdb.set_trace()
        data3d_seg = (data3d_a > 0).astype(np.int8)
        data3d_orig = data3d_b

        if args.visualization:
            pyed = py3DSeedEditor.py3DSeedEditor(
                    data3d_orig,
                    contour=data3d_seg
                    )
            
            
            pyed.show()
            #import pdb; pdb.set_trace()
        #fvall.insert(i, get_features(
        #    data3d_orig,
        #    data3d_seg,
        #    visualization=args.visualization
        #    ))
        tile_shape = [1,128,128]
        fv_t = get_features_in_tiles(data3d_orig, data3d_seg, tile_shape)
        cindexes, features_t, seg_cover_t = fv_t

        labels = np.array(seg_cover_t) > 0.5
        from sklearn import svm

        clf = svm.SVC()
        clf.fit(features_t, labels)
        classifed = clf.predict(features_t)
        import pdb; pdb.set_trace()
        fv_tiles.insert(i, fv_t)



#
#
    print fvall
    #write_csv(fvall)

# Ukládání výsledku do souboru
    output_file = os.path.join(path_to_script,  "20130919_liver_statistics_results.pkl")
    misc.obj_to_file(fvall,output_file, filetype='pickle')

if __name__ == "__main__":
    main()
