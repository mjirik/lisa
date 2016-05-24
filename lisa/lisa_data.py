#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © %YEAR% %USER% <%MAIL%>
#
# Distributed under terms of the %LICENSE% license.

"""
%HERE%
"""

import logging

logger = logging.getLogger(__name__)
import argparse
import os
import os.path as op


def create_lisa_data_dir_tree(oseg=None):

    odp = op.expanduser('~/lisa_data')
    if not op.exists(odp):
        os.makedirs(odp)

    if oseg is not None:
        # used for server sync
        oseg._output_datapath_from_server = op.join(oseg.output_datapath, 'sync', oseg.sftp_username, "from_server")
        # used for server sync
        oseg._output_datapath_to_server = op.join(oseg.output_datapath, 'sync', oseg.sftp_username, "to_server")
        odp = oseg.output_datapath
        if not op.exists(odp):
            os.makedirs(odp)
        odp = oseg._output_datapath_from_server
        if not op.exists(odp):
            os.makedirs(odp)
        odp = oseg._output_datapath_to_server
        if not op.exists(odp):
            os.makedirs(odp)

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