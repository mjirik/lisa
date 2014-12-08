#! /usr/bin/python
# -*- coding: utf-8 -*-

import subprocess
import traceback


import logging
logger = logging.getLogger(__name__)







def update():
    release_type = 'devel'
    # release_type = 'stable'

    print ('Updating submodules')
    branch_name = subprocess.check_output(['git', 'branch'])
    # if we found stabel (find is not -1), we should use specific version
    if branch_name.find('* stable') != -1 or release_type == 'stable':
        print ('Stable version prerequisities')
        use_specifed_version = 1
    else:
        use_specifed_version = 0

    # update submodules codes
    # This can update only one package to actual or defined version
    # installed version can be checked with
    #     pip show sed3
    pipshell = 'pip install -U --no-deps --user '
    pppysegbase = ['pysegbase', 'pysegbase==1.0.13']
    ppio3d = ['io3d', 'io3d==1.0.5']
    ppsed3 = ['sed3', 'sed3==1.0.4']
    try:
        #import pdb; pdb.set_trace()
        subprocess.call('git pull', shell=True)
        subprocess.call('git submodule update --init --recursive', shell=True)
        # subprocess.call('pip install -U --no-deps pysegbase --user', shell=True)
        # subprocess.call('pip install -U --no-deps io3d --user', shell=True)
        # subprocess.call('pip install -U --no-deps sed3 --user', shell=True)
        subprocess.call(pipshell + pppysegbase[use_specifed_version], shell=True)
        subprocess.call(pipshell + ppio3d[use_specifed_version], shell=True)
        subprocess.call(pipshell + ppsed3[use_specifed_version], shell=True)
# skelet3d is not in pipy
        # subprocess.call('pip install -U --no-deps skelet3d --user', shell=True)
    except:
        print ('Probem with git submodules')
        print (traceback.format_exc())


def main():
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    update()

if __name__ == "__main__":
    main()
