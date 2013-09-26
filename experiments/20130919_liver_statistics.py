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
import argparse
#import py3DSeedEditor

import misc
import datareader
import matplotlib.pyplot as plt




def feat_hist(data3d_orig, data3d_seg, visualization=True):
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




def get_features(data3d_orig, data3d_seg, visualization=True):
    """
    Sem doplníme všechny naše měření. Pro ukázku jsem vytvořil měření
    histogramu.
    data3d_orig: CT data3d_orig
    data3d_seg: jedničky tam, kde jsou játra
    """
    featur = {}
    featur['hist'] = feat_hist(data3d_orig, data3d_seg, visualization)




    return featur









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
    
    for i in range(0,len(inputdata['data'])):

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
        fvall.insert(i, get_features(
            data3d_orig,
            data3d_seg,
            visualization=args.visualization
            ))



#
#
    print fvall
    #write_csv(fvall)

# Ukládání výsledku do souboru
    output_file = os.path.join(path_to_script,  "20130919_liver_statistics_results.pkl")
    misc.obj_to_file(fvall,output_file, filetype='pickle')

if __name__ == "__main__":
    main()
