#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/py3DSeedEditor/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector

import logging
logger = logging.getLogger(__name__)

import inspect


import misc

def load_config(filename):
    if os.path.isfile(filename):
        cfg = misc.obj_from_file(filename, filetype='yaml')
    else:
        # default config
        pass

def get_default_function_config(p_fcn):
    #fcn_cfg = {p_fcn.__name__:inspect.getargspec(p_fcn)}
    fcn_cfg = inspect.getargspec(p_fcn)
    return fcn_cfg 

def subdict(bigdict, wanted_keys):
    return dict([(i, bigdict[i]) for i in wanted_keys if i in bigdict])


def save_config(cfg, pointer):
    pass
