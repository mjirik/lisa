#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © %YEAR%  <>
#
# Distributed under terms of the %LICENSE% license.

"""
Training module. Default setup makes nothing. Use --all to make all
"""

import logging

logger = logging.getLogger(__name__)
import argparse


import organ_localizator
import organ_model

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
            # required=True,
            help='input file'
    )
    parser.add_argument(
            '-d', '--debug', action='store_true',
            help='Debug mode')
    parser.add_argument(
            '-lm', '--liver-model', action='store_true',
            help='Train liver model')
    parser.add_argument(
            '-ll', '--liver-localizator', action='store_true',
            help='Train liver localizator')
    parser.add_argument(
            '--all', action='store_true',
            help='Train all')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)


    if args.liver_localizator or args.all:
        organ_localizator.train_liver_localizator_from_sliver_data()


    if args.liver_model or args.all:
        organ_model.train_liver_model_from_sliver_data()


if __name__ == "__main__":
    main()