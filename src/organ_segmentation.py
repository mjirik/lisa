#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os
sys.path.append("../extern/pycat/")
sys.path.append("../extern/pycat/extern/py3DSeedEditor/")
#import featurevector

import logging
logger = logging.getLogger(__name__)


import pdb
#  pdb.set_trace();
#import scipy.io

# ----------------- my scripts --------
import dcmreaddata
import pycat


if __name__ == "__main__":

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
    data3d = dcmreaddata.dcm_read_from_dir()

    print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )

    igc = pycat.ImageGraphCut(data3d, zoom = 0.25)
    igc.interactivity()


    igc.make_gc()
    igc.show_segmentation()
