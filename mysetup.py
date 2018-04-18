#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
import zipfile
import subprocess
import sys
# import traceback

import logging
logger = logging.getLogger(__name__)

import argparse

if sys.version_info < (3, 0):
    import urllib as urllibr
else:
    import urllib.request as urllibr


# import funkcí z jiného adresáře
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))


def submodule_update():
    # update submodules codes
    print('Updating submodules')
    try:
        # import pdb; pdb.set_trace()
        subprocess.call('git submodule update --init --recursive', shell=True)
        # subprocess.call('git submodule update --init --recursive')

    except:
        print('Probem with git submodules')


def check_python_architecture(pythondir, target_arch_str):
    """
    functions check architecture of target python
    """
    pyth_str = subprocess.check_output(
        [pythondir + 'python', '-c',
         'import platform; print(platform.architecture()[0])'])
    if pyth_str[:2] != target_arch_str:
        raise Exception(
            "Wrong architecture of target python. Expected arch is"
            + target_arch_str)


def definitions_win64_py32():
    global urlpython, urlmsysgit, urlmingw, urlnumpy, urlscipy, urlsklearn,\
        urlmatplotlib, urlcython, urlgco_python, urlgco, urlcompiler,\
        pythondir, pythonversion, target_arch_str
    urlpython = "http://www.python.org/ftp/python/3.2.3/python-3.2.3.amd64.msi"
    urlmsysgit = "http://msysgit.googlecode.com/files/Git-1.8.0-preview20121022.exe"
    urlmingw = "http://downloads.sourceforge.net/project/mingw/Installer/mingw-get-inst/mingw-get-inst-20120426/mingw-get-inst-20120426.exe?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fmingw%2Ffiles%2Flatest%2Fdownload%3Fsource%3Dfiles&ts=1359726876&use_mirror=ignum",
    urlnumpy = "http://home.zcu.cz/~mjirik/liver/download/win64_py32/numpy-MKL-1.6.2.win-amd64-py3.2.exe"
    urlscipy = "http://home.zcu.cz/~mjirik/liver/download/win64_py32/scipy-0.11.0.win-amd64-py3.2.exe"
    urlsklearn = "http://home.zcu.cz/~mjirik/liver/download/win64_py32/scikit-learn-0.13.win-amd64-py3.2.exe"
    urlmatplotlib = "http://home.zcu.cz/~mjirik/liver/download/win64_py32/matplotlib-1.2.0.win-amd64-py3.2.exe"
    urlcython = "http://home.zcu.cz/~mjirik/liver/download/win64_py32/Cython-0.18.win-amd64-py3.2.exe"
    urlgco_python = "https://github.com/amueller/gco_python/archive/master.zip"
    urlgco = "http://vision.csd.uwo.ca/code/gco-v3.0.zip"
    urlcompiler = "http://home.zcu.cz/~mjirik/liver/download/win64_py32/cygwinccompiler.py"
    pythondir = "c:/python32/"
    pythonversion = (3, 2)
    target_arch_str = '64'


def definitions_win32_py27():
    global urlpython, urlmsysgit, urlmingw, urlnumpy, urlscipy, urlsklearn, urlmatplotlib, urlcython, urlgco_python, urlgco, urlcompiler, pythondir, pythonversion, target_arch_str
    urlpython = "http://www.python.org/ftp/python/2.7.3/python-2.7.3.msi"
    urlmsysgit = "http://msysgit.googlecode.com/files/Git-1.8.0-preview20121022.exe"
    urlmingw = "http://downloads.sourceforge.net/project/mingw/Installer/mingw-get-inst/mingw-get-inst-20120426/mingw-get-inst-20120426.exe?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fmingw%2Ffiles%2Flatest%2Fdownload%3Fsource%3Dfiles&ts=1359726876&use_mirror=ignum"
    urlnumpy = "http://home.zcu.cz/~mjirik/liver/download/win32_py27/numpy-MKL-1.6.2.win32-py2.7.exe"
    urlscipy = "http://home.zcu.cz/~mjirik/liver/download/win32_py27/scipy-0.11.0.win32-py2.7.exe"
    urlsklearn = "http://home.zcu.cz/~mjirik/liver/download/win32_py27/scikit-learn-0.13.win32-py2.7.exe"
    urlmatplotlib = "http://home.zcu.cz/~mjirik/liver/download/win32_py27/matplotlib-1.2.0.win32-py2.7.exe"
    urlcython = "http://home.zcu.cz/~mjirik/liver/download/win32_py27/Cython-0.17.4.win32-py2.7.exe"
    urlgco_python = "https://github.com/amueller/gco_python/archive/master.zip"
    urlgco = "http://vision.csd.uwo.ca/code/gco-v3.0.zip"
    urlcompiler = "http://home.zcu.cz/~mjirik/liver/download/win32_py27/cygwinccompiler.py"
    pythondir = "c:/python27/"
    pythonversion = (2,7)
    target_arch_str = '32'


def windows_install():
    try:
        os.mkdir("tmp")
    except:
        pass

    # problems with 64 bit version and mingw
    #definitions_win64_py32()
    definitions_win32_py27()

    # check correct python version
    if (sys.version_info[0:2] != pythonversion):

        print("Different python required \n installing python "
               + str(pythonversion))
        print(sys.version_info)

        local_file_name = 'tmp\python.msi'
        urllibr.urlretrieve(urlpython, local_file_name)
        a = "msiexec /i " + local_file_name
        print(a)
        subprocess.call(a)

        check_python_architecture(pythondir, target_arch_str)

        print("Please restart installer with new python")
        subprocess.call(pythondir + "python.exe mysetup.py")
        return
    else:
        print("You have python " + str(pythonversion))

    # install distribute and pip
    # this is not necessary for numpy, scipy etc., because we have binaries
    # but we need it for pydicom
    urldistribute = "http://python-distribute.org/distribute_setup.py"
    local_file_name = "./tmp/distribute_setup.py"
    urllibr.urlretrieve(urldistribute, local_file_name)
    subprocess.call(pythondir + "python.exe distribute_setup.py", cwd="./tmp/")

    urlpip = "https://raw.github.com/pypa/pip/master/contrib/get-pip.py"
    local_file_name = "./tmp/get-pip.py"
    urllibr.urlretrieve(urlpip, local_file_name)
    subprocess.call(pythondir + "python.exe get-pip.py", cwd="./tmp/")

    # numpy, scipy, matplotlib, scikit-learn, cython
    print("numpy, scipy, matplotlib, scikit-learn, cython install")
    #import pdb; pdb.set_trace()
    try:

        subprocess.call(
            pythondir +
            "Scripts/pip.exe install numpy scipy scikit-learn"
            + " matplotlib cython nose",
            cwd="./tmp/")
    except:
        print("alternative installation ")
        download_and_run(urlnumpy, "./tmp/numpy.exe")
        download_and_run(urlscipy, "./tmp/scipy.exe")
        download_and_run(urlsklearn, "./tmp/sklearn.exe")
        download_and_run(urlmatplotlib, "./tmp/matplotlib.exe")
        download_and_run(urlcython, "./tmp/cython.exe")

    # install pydicom
    subprocess.call(pythondir + "Scripts/pip.exe install pydicom",
                    cwd="./tmp/")
    subprocess.call(pythondir + "Scripts/pip.exe install pyyaml", cwd="./tmp/")


def windows_get_git():

    # instalace msys gitu
    print("MSYS Git install")
    #import pdb; pdb.set_trace()
    download_and_run(url=urlmsysgit, local_file_name='./tmp/msysgit.exe')


def windows_build_gco():
    # install MinGW
    print("MinGW install")
    download_and_run(url=urlmingw, local_file_name='./tmp/mingw.exe')

    # install gco_python
    print("gco_python install")
    # --- first we need gco_python source codes
    #try:
    #    os.mkdir("tmp/gco_python")
    #except:
    #    pass

    local_file_name = './tmp/gco_python.zip'
    urllibr.urlretrieve(urlgco_python, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./tmp/')

    # --- second we need gco sources
    try:
        os.mkdir("tmp/gco_python-master/gco_src")
    except:
        pass

    local_file_name = './tmp/gco_src.zip'
    urllibr.urlretrieve(urlgco, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./tmp/gco_python-master/gco_src')

    # --- now compile
    print("gco compilation")

    # there is a bug in python mingw compiler with -mno-cygwin, so here is simple patch

    local_file_name = pythondir + "Lib\distutils\cygwinccompiler.py"
    urllibr.urlretrieve(urlcompiler, local_file_name)

    # compilation with mingw
    subprocess.call(pythondir + "python.exe setup.py build_ext -i --compiler=mingw32", cwd="./tmp/gco_python-master/")
    # parametr -i by to mel nainstalovat sam, ale nedala to
    subprocess.call(pythondir + "python.exe setup.py build --compiler=mingw32", cwd="./tmp/gco_python-master/")
    subprocess.call(pythondir + "python.exe setup.py install --skip-build", cwd="./tmp/gco_python-master/")

    #subprocess.call('envinstall\envwindows.bat')


def remove(local_file_name):
    try:
        os.remove(local_file_name)
    except Exception as e:
        print("Cannot remove file '" + local_file_name + "'. Please remove\
        it manually.")
        print(e)


def downzip(url, destination='./sample_data/'):
    """
    Download, unzip and delete.
    """

    # url = "http://147.228.240.61/queetech/sample-data/jatra_06mm_jenjatra.zip"
    local_file_name = './sample_data/tmp.zip'
    urllibr.urlretrieve(url, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall(destination)
    remove(local_file_name)


def get_sample_data():
    """
    This function is deprecated. Now we use imtools.qmisc.get_sample_data
    :return:
    """
    # download sample data
    print('Downloading sample data')

    try:
        os.mkdir('sample_data')
    except:
        pass

    # Puvodni URL z mathworks
    # url =  "http://www.mathworks.com/matlabcentral/fileexchange/2762-dicom-example-files?download=true"
    # url = "http://www.mathworks.com/includes_content/domainRedirect/domainRedirect.html?uri=http%3A%2F%2Fwww.mathworks.com%2Fmatlabcentral%2Ffileexchange%2F2762-dicom-example-files%3Fdownload%3Dtrue%26nocookie%3Dtrue"
    url = "http://147.228.240.61/queetech/sample-data/head.zip"
    local_file_name = './sample_data/head.zip'

    urlobj = urllibr.urlopen(url)
    url = urlobj.geturl()
    urllibr.urlretrieve(url, local_file_name)

    datafile = zipfile.ZipFile(local_file_name)
    #datafile.setpassword('queetech')
    datafile.extractall('./sample_data/')
    remove(local_file_name)

# get jatra_06mm_jenjatra

    # url = "http://147.228.240.61/queetech/sample-data/jatra_06mm_jenjatraplus.zip"
    # local_file_name = './sample_data/jatra_06mm_jenjatraplus.zip'
    url = "http://147.228.240.61/queetech/sample-data/jatra_06mm_jenjatra.zip"
    local_file_name = './sample_data/jatra_06mm_jenjatra.zip'
    urllibr.urlretrieve(url, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./sample_data/')
    remove(local_file_name)
# get jatra 5mm
    url = "http://147.228.240.61/queetech/sample-data/jatra_5mm.zip"
    local_file_name = './sample_data/jatra_5mm.zip'
    urllibr.urlretrieve(url, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./sample_data/')
    remove(local_file_name)

# get experiment data
    url = "http://147.228.240.61/queetech/sample-data/exp.zip"
    local_file_name = './sample_data/exp.zip'
    urllibr.urlretrieve(url, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./sample_data/')
    remove(local_file_name)
# get sliver sample
    url = "http://147.228.240.61/queetech/sample-data/sliver_training_001.zip"
    local_file_name = './sample_data/sliver_training_001.zip'
    urllibr.urlretrieve(url, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./sample_data/')
    remove(local_file_name)

# get volumetry sample
    url = "http://147.228.240.61/queetech/sample-data/volumetrie.zip"
    local_file_name = './sample_data/volumetrie.zip'
    urllibr.urlretrieve(url, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./sample_data/')
    remove(local_file_name)

# get organ.pkl and vessels.pkl
    url = "http://147.228.240.61/queetech/sample-data/organ.pkl.zip"
    local_file_name = './sample_data/organ.pkl.zip'
    urllibr.urlretrieve(url, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./sample_data/')
    remove(local_file_name)

    url = "http://147.228.240.61/queetech/sample-data/vessels.pkl.zip"
    local_file_name = './sample_data/vessels.pkl.zip'
    urllibr.urlretrieve(url, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./sample_data/')
    remove(local_file_name)

# get biodur samples
    url = "http://147.228.240.61/queetech/sample-data/biodur_sample.zip"
    local_file_name = './sample_data/biodur_sample.zip'
    urllibr.urlretrieve(url, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./sample_data/')
    remove(local_file_name)

# get gensei samples
    url = "http://147.228.240.61/queetech/sample-data/gensei_slices.zip"
    local_file_name = './sample_data/gensei_slices.zip'
    urllibr.urlretrieve(url, local_file_name)
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./sample_data/')
    remove(local_file_name)

    downzip("http://147.228.240.61/queetech/sample-data/exp_small.zip")


def windows_get_gco():
    url = "http://147.228.240.61/queetech/install/pygco-py27-32bit/pygco.pyd"

    local_file_name = "C:\python27\Lib\site-packages\pygco.pyd"
    urllibr.urlretrieve(url, local_file_name)


def download_and_run(url, local_file_name):
    urllibr.urlretrieve(url, local_file_name)
    subprocess.call(local_file_name)


def get_conda_path():
    import os.path as op
    conda_pth = op.expanduser('~/anaconda/bin')
    if not op.exists(conda_pth):
        conda_pth = op.expanduser('~/miniconda/bin')
    return conda_pth


def file_copy_and_replace_lines(in_path, out_path):
    import shutil
    import fileinput

    # print "path to script:"
    # print path_to_script
    lisa_path = os.path.abspath(path_to_script)

    shutil.copy2(in_path, out_path)
    conda_path = get_conda_path()

    # print 'ip ', in_path
    # print 'op ', out_path
    # print 'cp ', conda_path
    for line in fileinput.input(out_path, inplace=True):
        # coma on end makes no linebreak
        line = line.replace("@{LISA_PATH}", lisa_path)
        line = line.replace("@{CONDA_PATH}", conda_path)
        print(line)


def make_icon():
    import platform

    system = platform.system()
    if system == 'Darwin':
        # MacOS
        __make_icon_osx()
        pass
    elif system == "Linux":
        __make_icon_linux()


def __make_icon_osx():
    home_path = os.path.expanduser('~')
    in_path = os.path.join(path_to_script, "applications/lisa_gui")
    dt_path = os.path.join(home_path, "Desktop")
    subprocess.call(['ln', '-s', in_path, dt_path])


def __make_icon_linux():

    in_path = os.path.join(path_to_script, "applications/lisa.desktop.in")
    in_path_ha = os.path.join(path_to_script, "applications/ha.desktop.in")
    print("icon input path:")
    print(in_path, in_path_ha)

    home_path = os.path.expanduser('~')

    if os.path.exists(os.path.join(home_path, 'Desktop')):
        desktop_path = os.path.join(home_path, 'Desktop')
    elif os.path.exists(os.path.join(home_path, 'Plocha')):
        desktop_path = os.path.join(home_path, 'Plocha')
    else:
        print("Cannot find desktop directory")
        desktop_path = None

    # copy desktop files to desktop
    if desktop_path is not None:
        out_path = os.path.join(desktop_path, "lisa.desktop")
        out_path_ha = os.path.join(desktop_path, "ha.desktop")

        # fi = fileinput.input(out_path, inplace=True)
        print("icon output path:")
        print(out_path, out_path_ha)
        file_copy_and_replace_lines(in_path, out_path)
        file_copy_and_replace_lines(in_path_ha, out_path_ha)

    # copy desktop files to $HOME/.local/share/applications/
    # to be accesable in application menu (Linux)
    local_app_path = os.path.join(home_path, '.local/share/applications')
    if os.path.exists(local_app_path) and os.path.isdir(local_app_path):
        out_path = os.path.join(local_app_path, "lisa.desktop")

        out_path_ha = os.path.join(local_app_path, "ha.desktop")

        print("icon output path:")
        print(out_path, out_path_ha)
        file_copy_and_replace_lines(in_path, out_path)
        file_copy_and_replace_lines(in_path_ha, out_path_ha)

    else:
        print("Couldnt find $HOME/.local/share/applications/.")


def main():
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(
        description=
        'Segment vessels from liver \n \npython organ_segmentation.py\n \n\
        python organ_segmentation.py -mroi -vs 0.6')
    parser.add_argument(
        '-d', '--get_sample_data', action='store_true',
        default=False,
        help='Get sample data')
    parser.add_argument(
        '-i','--install', action='store_true',
        default=False,
        help='Install')
    parser.add_argument('-icn','--make_icon', action='store_true',
            default=False,
            help='Creates desktop icon, works only in ubuntu')
    parser.add_argument('-g','--get_git', action='store_true',
            default=False,
            help='Get git in windows')

    parser.add_argument('--build_gco', action='store_true',
            default = False, help='Build gco_python in windows. Problematic step.')
    args = parser.parse_args()

#    if args.get_sample_data == False and args.install == False and args.build_gco == False:
## default setup is install and get sample data
#        args.get_sample_data = True
#        args.install = True
#        args.build_gco = False

    if args.get_sample_data:
        import lisa
        import lisa.dataset
        lisa.dataset.get_sample_data()
        # get_sample_data()

    if args.make_icon:
        make_icon()

    if args.install:
        print('Installing system environment')
        if sys.platform.startswith('linux'):

            subprocess.call('./envinstall/envubuntu.sh')
            #submodule_update()
        elif sys.platform.startswith('win'):
            windows_install()
            if args.build_gco:
                windows_build_gco()
            else:
                windows_get_gco()
            if args.get_git:
                windows_get_git()
                #submodule_update()


if __name__ == "__main__":
    main()
