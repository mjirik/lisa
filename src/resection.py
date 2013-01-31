#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os
sys.path.append("../extern/pycat/")
sys.path.append("../extern/pycat/extern/py3DSeedEditor/")
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)
# ----------------- my scripts --------
import misc
import py3DSeedEditor

def resection():

    pass

if __name__ == "__main__":
    data = misc.obj_from_file("out", filetype = 'pickle')
    ds = data['segmentation'] == 3
    ped = py3DSeedEditor.py3DSeedEditor(ds)
    import pdb; pdb.set_trace()
