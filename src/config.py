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


def get_config(filename, default_cfg):
    """
    Looks config file and update default_cfg values.

    If file does not exist it is created.
    """
    if os.path.isfile(filename):
        cfg = misc.obj_from_file(filename, filetype='yaml')
        default_cfg.update(cfg)
        cfg_out = default_cfg
    else:
        misc.obj_to_file(default_cfg, filename, filetype='yaml')
        cfg_out = default_cfg

        # default config
    return cfg_out


def get_function_keys(p_fcn):
    #fcn_cfg = {p_fcn.__name__:inspect.getargspec(p_fcn)}
    fcn_cfg = inspect.getargspec(p_fcn)
    return fcn_cfg[0]


def get_default_function_config(p_fcn):
    """
    Return dictionary with keys and default params of function.
    """
    #fcn_cfg = {p_fcn.__name__:inspect.getargspec(p_fcn)}
    fcn_argspec = inspect.getargspec(p_fcn)
    valid_keys = fcn_argspec[0][-len(fcn_argspec[3]):]
    fcn_cfg = {}
    for i in range(0, len(valid_keys)):
        fcn_cfg[valid_keys[i]] = fcn_argspec[3][i]

    return fcn_cfg


def subdict(bigdict, wanted_keys):
    ret = dict([(i, bigdict[i]) for i in wanted_keys if i in bigdict])
    # @TODO not working for wanted_keys len == 1, I dont know why, hack fallows
    if len(wanted_keys) == 1:
        ret = {bigdict[wanted_keys[0]]}
    return ret


def save_config(cfg, pointer):
    pass
