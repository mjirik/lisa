#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/"))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/sed3/"))
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
import sed3
# import dcmreaddata1 as dcmr
# try:
#    import dcmreaddata as dcmr
# except:
#    from imcut import dcmreaddata as dcmr

import argparse
#import sed3

from .. import misc
from .. import organ_segmentation
from .. import experiments


def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    params = {'datadir':None, 'working_voxelsize_mm':3}

    dirpath = os.path.join(path_to_script, "../../../data/medical/data_orig/volumetrie")

    experiment_results = {'params':params,'dirpath':dirpath, 'volume_l':{}}

    output_dirpath = os.path.join(path_to_script, '../../../data/medical')
    dirlist = experiments.get_subdirs(dirpath)

    for key in dirlist:
        print(key)

#        import pdb; pdb.set_trace()
        try:
            dirname = dirlist[key]['abspath']
            params['datadir'] = dirname

            oseg = organ_segmentation.OrganSegmentation(**params)

            oseg.interactivity()
            print("Volume " +
                    str(oseg.get_segmented_volume_size_mm3() / 1000000.0)
                    + ' [l]')

            volume_l = (oseg.get_segmented_volume_size_mm3() / 1000000.0)

            experiment_results['volume_l'][key] = volume_l

            head, teil = os.path.split(dirname)
            filename_organ = os.path.join(output_dirpath, teil)

            data = oseg.export()
            misc.obj_to_file(data, filename_organ + "-organ.pkl", filetype='pickle')
            #misc.obj_to_file(data, "organ.pkl", filetype='pickle')
            misc.obj_to_file(oseg.get_iparams(), filename_organ + '-iparams.pkl', filetype='pickle')
            misc.obj_to_file(experiment_results, filename_organ + "-info.yaml", filetype='yaml')

        except:
            print('Selhani, pokracujeme dal')
            print(traceback.format_exc())
            import pdb; pdb.set_trace()

    misc.obj_to_file(experiment_results, "results.yaml", filetype='yaml')
    #igc = pycat.ImageGraphCut(data3d, zoom = 0.5)
    #igc.interactivity()

    #igc.make_gc()
    #igc.show_segmentation()

    # volume
    #volume_mm3 = np.sum(oseg.segmentation > 0) * np.prod(oseg.voxelsize_mm)

    #pyed = sed3.sed3(oseg.data3d, contour =
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
