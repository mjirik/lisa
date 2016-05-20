#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © %YEAR%  <>
#
# Distributed under terms of the %LICENSE% license.

"""

"""

import logging

logger = logging.getLogger(__name__)
import argparse
import os.path as op

import config_default
import imtools
import imtools.sample_data
import glob
import os.path as op
import re


def sample_data_path():
    return op.expanduser("~/lisa_data/sample_data/")

def get_sample_data():
    keys = imtools.sample_data.data_urls.keys()
    imtools.sample_data.get_sample_data(keys, sample_data_path())


class DataIteratorOutput:
    def __init__(self, orig, ref, data, label, i):
        self.orig = orig
        self.ref = ref
        self.data = data
        self.label = label
        self.i = i

class DataDirIterator:
    def __init__(
        self,
        data_dir=None,
        reference_dir="~/data/medical/orig/sliver07/training/",
        common_pattern="*[0-9]",
        orig_pattern="*orig%s.mhd",
        seg_pattern="*seg%s.mhd",
        data_pattern="*%s*.pklz"

    ):
        """
        Iterate over all files in reference dir and return file names with defined patter.


        :param data_dir: other related data can be used by defining this parameter
        :param reference_dir: Data directory (f.e. sliver07 training data dir)
        :param common_pattern: used in all other patterns as a kernel. It can select specific files from dir
        :param orig_pattern: select files with orig data
        :param seg_pattern: select files with segmentation
        :param data_pattern: different dir related dir pattern
        """



        self.data_dir = data_dir

        sliver_reference_dir = op.expanduser(reference_dir)


        patterns = seg_pattern % (common_pattern)
        patterno = orig_pattern % (common_pattern)

        self.orig_fnames = glob.glob(sliver_reference_dir + patterno)
        self.ref_fnames = glob.glob(sliver_reference_dir + patterns)
        self.orig_fnames.sort()
        self.ref_fnames.sort()

        if self.data_dir is not None:
            self.data_dir = op.expanduser(data_dir)
            patternd = data_pattern % (common_pattern)
            self.data_fnames = glob.glob(data_dir + patternd)
            self.data_fnames.sort()
        else:
            self.data_fnames = [None] * len(self.orig_fnames)

        self.current = 0
        self.high = len(self.orig_fnames)

    def __iter__(self):
        return self

    def next(self): # Python 3: def __next__(self)
        if self.current >= self.high:
            raise StopIteration
        else:

            numeric_label = re.search(".*g(\d+)", self.orig_fnames[self.current]).group(1)
            out = DataIteratorOutput(
                self.orig_fnames[self.current],
                self.ref_fnames[self.current],
                self.data_fnames[self.current],
                numeric_label,
                self.current

            )
            self.current += 1
            return out


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
    # parser.add_argument(
    #     '-i', '--inputfile',
    #     default=None,
    #     required=True,
    #     help='input file'
    # )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)

    get_sample_data()

if __name__ == "__main__":
    main()