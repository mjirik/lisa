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



def make_icon():
    import platform

    system = platform.system()
    if system == 'Darwin':
        # MacOS
        __make_icon_osx()
        pass
    elif system == "Linux":
        __make_icon_linux()

def lidapath():
    return op.expanduser('~/lisa_data')

def __make_icon_osx():
    import wget
    wget.download(
        "https://raw.githubusercontent.com/mjirik/lisa/master/lisa/requirements_pip.txt",
        out=lidapath() + "lisa_gui"
    )
    home_path = os.path.expanduser('~')
    in_path = os.path.join(path_to_script, "applications/lisa_gui")
    dt_path = os.path.join(home_path, "Desktop")
    subprocess.call(['ln', '-s', in_path, dt_path])


def __make_icon_linux():

    in_path = os.path.join(path_to_script, "applications/lisa.desktop.in")
    in_path_ha = os.path.join(path_to_script, "applications/ha.desktop.in")
    print "icon input path:"
    print in_path, in_path_ha

    home_path = os.path.expanduser('~')

    if os.path.exists(os.path.join(home_path, 'Desktop')):
        desktop_path = os.path.join(home_path, 'Desktop')
    elif os.path.exists(os.path.join(home_path, 'Plocha')):
        desktop_path = os.path.join(home_path, 'Plocha')
    else:
        print "Cannot find desktop directory"
        desktop_path = None

    # copy desktop files to desktop
    if desktop_path is not None:
        out_path = os.path.join(desktop_path, "lisa.desktop")
        out_path_ha = os.path.join(desktop_path, "ha.desktop")

        # fi = fileinput.input(out_path, inplace=True)
        print "icon output path:"
        print out_path, out_path_ha
        file_copy_and_replace_lines(in_path, out_path)
        file_copy_and_replace_lines(in_path_ha, out_path_ha)

    # copy desktop files to $HOME/.local/share/applications/
    # to be accesable in application menu (Linux)
    local_app_path = os.path.join(home_path, '.local/share/applications')
    if os.path.exists(local_app_path) and os.path.isdir(local_app_path):
        out_path = os.path.join(local_app_path, "lisa.desktop")

        out_path_ha = os.path.join(local_app_path, "ha.desktop")

        print "icon output path:"
        print out_path, out_path_ha
        file_copy_and_replace_lines(in_path, out_path)
        file_copy_and_replace_lines(in_path_ha, out_path_ha)

    else:
        print "Couldnt find $HOME/.local/share/applications/."

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