#! /usr/bin/python
# -*- coding: utf-8 -*-

#import sys
#import os

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()

def suggest_file_name(file_path):
    """
    Try if exist path and add number to its end.
    """
    import os.path
    if os.path.exists(file_path):
        cislo = 2
        try:
            cislo = int(file_path[-1])
        except Exception as e:
            pass

        file_path = filepath + str(cislo)


    return file_path

def obj_from_file(filename = 'annotation.yaml', filetype = 'yaml'):
    ''' Read object from file '''
# TODO solution for file extensions
    if filetype == 'yaml':
        import yaml
        f = open(filename, 'rb')
        obj = yaml.load(f)
        f.close()
    elif filetype in ('pickle', 'pkl', 'pklz', 'picklezip'):
        fcontent = read_pkl_and_pklz(filename)
        import pickle
        obj = pickle.loads(fcontent)
    else:
        logger.error('Unknown filetype ' + filetype)
    return obj


def read_pkl_and_pklz(filename):
    """
    Try read zipped or not zipped pickle file
    """
    fcontent = None
    try:
        import gzip
        f = gzip.open(filename, 'rb')
        fcontent = f.read()
        f.close()
    except Exception as e:
        logger.warning("Input gzip exception: " + str(e))
        f = open(filename, 'rb')
        fcontent = f.read()
        f.close()

    return fcontent



def obj_to_file(obj, filename = 'annotation.yaml', filetype = 'yaml'):
    '''Writes annotation in file
    '''
    #import json
    #with open(filename, mode='w') as f:
    #    json.dump(annotation,f)

    # write to yaml

    if filetype == 'yaml':
        f = open(filename, 'wb')
        import yaml
        yaml.dump(obj,f)
        f.close
    elif filetype in ('pickle', 'pkl'):
        f = open(filename, 'wb')
        import pickle
        pickle.dump(obj,f)
        f.close
    elif filetype in ('picklezip', 'pklz'):
        import gzip
        import pickle
        f = gzip.open(filename, 'wb', compresslevel=1)
        #f = open(filename, 'wb')
        pickle.dump(obj,f)
        f.close
    else:
        logger.error('Unknown filetype ' + filetype)
