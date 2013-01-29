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


def resection():

    pass

if __name__ == "__main__":
    data = misc.obj_from_file("out", filetype = 'pickle')
    import pdb; pdb.set_trace()
