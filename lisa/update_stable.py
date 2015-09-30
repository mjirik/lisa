#! /usr/bin/python
# -*- coding: utf-8 -*-

import subprocess
import traceback


import logging
logger = logging.getLogger(__name__)
import datetime


# def write_update_info_in_file():
#     import misc
#     now = datetime.datetime.now() 
def update_by_plan(filename=".update_plan.yaml", update_periode_days=10):
    """
    :return: True or False
    """
    retval = False
    now = datetime.datetime.now() 
    try:
        # this import is necessary here. This module requires scipy and it is 
        # not installed yet
        import misc
        data = misc.obj_from_file(filename)

        string_date = data['update_datetime']
        dt = datetime.datetime.strptime(string_date, "%Y-%m-%d %H:%M:%S.%f")
        dt_delta = datetime.timedelta(days=update_periode_days)
        if now > (dt + dt_delta):
            retval = True
            data['update_datetime'] = str(now)
            misc.obj_to_file(data, filename)

        # data['update_datetime'] = 
    except:
        retval = True
        data = {}
        data['update_datetime'] = str(now)
        misc.obj_to_file(data, filename)

    return retval

def update(dry_run=False):
    if update_by_plan():
        make_update(dry_run)

def make_update(dry_run=False):
    import os.path as op
    path_to_script = op.dirname(op.abspath(__file__))
    path_to_base = op.abspath(op.join(path_to_script, '../'))
    # release_type = 'devel'
    # release_type = 'stable'

    if not dry_run:
        try:
            subprocess.call('git pull', shell=True)
        except:
            logger.warning('Problem with git pull')
    conda_ok = True
    print ('Updating conda root modules')
    try:
        cmd = ["conda", "install", "--yes", "--file",
               op.join(path_to_base, "requirements_conda_root.txt")]
        if dry_run:
            cmd.append('--dry-run')
        subprocess.call(cmd)
    except:
        logger.warning('Problem with conda root update')
        traceback.print_exc()
        conda_ok = False

    print ('Updating conda modules')
    try:
        cmd = ["conda", "install", "--yes",
               # "-c", 'luispedro ',
               '-c', 'SimpleITK', "-c", "menpo", "--file",
               op.join(path_to_base, "requirements_conda.txt")]
        if dry_run:
            cmd.append('--dry-run')
        subprocess.call(cmd)
    except:
        logger.warning('Problem with conda update')
        traceback.print_exc()
        conda_ok = False
        # try:
        #     install_and_import('wget', '--user')

    print ('Updating pip modules')
    try:

        cmd = ["pip", "install", '-U', '--no-deps']
        if not conda_ok:
            cmd.append('--user')
        cmd = cmd + ["-r", op.join(path_to_base, "requirements_pip.txt")]
        if dry_run:
            cmd.insert(1, '-V')
        subprocess.call(cmd)
    except:
        logger.warning('Problem with pip update')
        traceback.print_exc()

    # branch_name = subprocess.check_output(['git', 'branch'])
    # # if we found stabel (find is not -1), we should use specific version
    # if branch_name.find('* stable') != -1 or release_type == 'stable':
    #     print ('Stable version prerequisities')
    #     use_specifed_version = 1
    # else:
    #     use_specifed_version = 0

    # update submodules codes
    # This can update only one package to actual or defined version
    # installed version can be checked with
    #     pip show sed3
    # pipshell = 'pip install -U --no-deps --user '
    # pppysegbase = ['pysegbase', 'pysegbase==1.0.13']
    # ppio3d = ['io3d', 'io3d==1.0.5']
    # ppsed3 = ['sed3', 'sed3==1.0.4']
    # if not dry_run:
    #     try:
    #         subprocess.call('git pull', shell=True)
    #         subprocess.call(
    #             'git submodule update --init --recursive', shell=True)
    #         subprocess.call(
    #             pipshell + pppysegbase[use_specifed_version], shell=True)
    #         subprocess.call(pipshell + ppio3d[use_specifed_version],
    #             shell=True)
    #         subprocess.call(pipshell + ppsed3[use_specifed_version],
    #             shell=True)
    #         # skelet3d is not in pipy
    #         # subprocess.call(
    #         #      'pip install -U --no-deps skelet3d --user', shell=True)
    #     except:
    #         print ('Probem with git submodules')
    #         print (traceback.format_exc())


def install_and_import(package, pip_params=[]):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        main_args = ['install'] + pip_params
        pip.main(main_args)
    finally:
        globals()[package] = importlib.import_module(package)


def main():
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    update()

if __name__ == "__main__":
    main()
