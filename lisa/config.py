#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
This is module for configuration support.

There are two config files in Lisa. One is in source directory. It is default
config file. It must not have parameter config_version. Second file is in
~/lisa_data directory. This file must have config_version parameter.

"""

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
# sys.path.append(os.path.join(path_to_script,
#                              "../extern/sed3/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector

import logging
logger = logging.getLogger(__name__)

import inspect


from io3d import misc
# import misc


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

def delete_config_records(filename, records_to_save=[]):
    """
    All records of config file are deleted except records_to_save.
    """
    if os.path.isfile(filename):
        cfg = misc.obj_from_file(filename, filetype='yaml')
        new_cfg = subdict(cfg, records_to_save)
        print cfg
        misc.obj_to_file(new_cfg, "test_" + filename , filetype='yaml')


def update_config_records(filename, new_cfg):
    """
    All records of config file are updated exept records_to_save.
    """
    if os.path.isfile(filename):
        cfg = misc.obj_from_file(filename, filetype='yaml')
        cfg.update(new_cfg)
        print cfg
        misc.obj_to_file(new_cfg, "test_" + filename , filetype='yaml')


def check_config_version_and_remove_old_records(filename, version,
                                                records_to_save):
    """
    Check if config file version is ok. If it is not all records except
    records_to_save are deleted and config_version in file is set to version.
    It is used to update user configuration.
    """
    if os.path.isfile(filename):
        cfg = misc.obj_from_file(filename, filetype='yaml')
        if ('config_version' in cfg and (cfg['config_version'] == version)):
# everything is ok, no need to panic
            return
        else:
# older version of config file
            cfg = misc.obj_from_file(filename, filetype='yaml')
            misc.obj_to_file(cfg, filename + '.old', filetype='yaml')
            print 'cfg ', cfg
            new_cfg = subdict(cfg, records_to_save)
            new_cfg['config_version'] = version
            print 'ncfg ', new_cfg
            misc.obj_to_file(new_cfg, filename, filetype='yaml')






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


def save_config(cfg, filename):
    misc.obj_to_file(cfg, filename, filetype='yaml')
