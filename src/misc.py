#! /usr/bin/python
# -*- coding: utf-8 -*-

#import sys
#import os

import logging
logger = logging.getLogger(__name__)

def obj_from_file(filename = 'annotation.yaml', filetype = 'yaml'):
    ''' Read object from file '''
# TODO solution for file extensions
    f = open(filename, 'rb')
    if filetype == 'yaml':
        import yaml
        obj = yaml.load(f)
    elif filetype == 'pickle':
        import pickle
        obj = pickle.load(f)
    else:
        logger.error('Unknown filetype')
    f.close()
    return obj


def obj_to_file(obj, filename = 'annotation.yaml', filetype = 'yaml'):
    '''Writes annotation in file
    '''
    #import json
    #with open(filename, mode='w') as f:
    #    json.dump(annotation,f)

    # write to yaml

    f = open(filename, 'wb')
    if filetype == 'yaml':
        import yaml
        yaml.dump(obj,f)
    elif filetype == 'pickle':
        import pickle
        pickle.dump(obj,f)
    else:
        logger.error('Unknown filetype')
    f.close
