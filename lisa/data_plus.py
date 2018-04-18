#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2014 mjirik <mjirik@hp-mjirik>
#
# Distributed under terms of the MIT license.

"""
Module for work with dataplus format

"""

import logging
logger = logging.getLogger(__name__)
import argparse
import numpy as np


def default_slab():
    return {
        'none': 0,
        'liver': 1,
        'hearth': 10,
        'left kidney': 15,
        'right kidney': 16,
    }

def get_slab_value(slab, label, value=None):
    if label in slab.keys():
        return slab[label]
    else:
        if value is None:
            value = np.max(list(slab.values())) + 1
        slab[label] = value

# class Slab(dict):

def main():
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    # fh = logging.FileHandler('log.txt')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.debug('start')

    # input parser
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        required=True,
        help='input file'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)



if __name__ == "__main__":
    main()
