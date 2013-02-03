import os
import zipfile
import subprocess
import sys

if sys.version_info < (3,0):
    import urllib as urllibr
else:
    import urllib.request as urllibr






def submodule_update():
    # update submodules codes
    print ('Updating submodules')
    try:
        #import pdb; pdb.set_trace()
        subprocess.call('git submodule update --init --recursive', shell=True)
        #subprocess.call('git submodule update --init --recursive')

    except:
        print ('Probem with git submodules')


def definitions_win64_py32():
    global urlpython, urlmsysgit, urlmingw, urlnumpy, urlscipy, urlsklearn, urlmatplotlib, urlcython, urlgco_python, urlgco, urlcompiler, pythondir, pythonversion
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
    pythonversion = (3,2)
    
    
def definitions_win32_py27():
    global urlpython, urlmsysgit, urlmingw, urlnumpy, urlscipy, urlsklearn, urlmatplotlib, urlcython, urlgco_python, urlgco, urlcompiler, pythondir, pythonversion
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
        
        print ("Different python required \n installing python " + str(pythonversion))
        print (sys.version_info)
        
        local_file_name = 'tmp\python.msi'
        urllibr.urlretrieve(urlpython, local_file_name)
        a = "msiexec /i " + local_file_name
        print (a)
        subprocess.call(a)

        print ("Please restart installer with new python")
        subprocess.call(pythondir + "python.exe mysetup.py")
        return
    else:
        print ("You have python " + str (pythonversion))

    
    

    # instalace msys gitu
    print ("MSYS Git install")
    #import pdb; pdb.set_trace()
    download_and_run(url = urlmsysgit, local_file_name = './tmp/msysgit.exe' )

    # install MinGW
    print ("MinGW install")
    download_and_run(url = urlmingw, local_file_name = './tmp/mingw.exe')


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
    print ("numpy, scipy, matplotlib, scikit-learn, cython install")
    #import pdb; pdb.set_trace()
    download_and_run(urlnumpy, "./tmp/numpy.exe")
    download_and_run(urlscipy, "./tmp/scipy.exe")
    download_and_run(urlsklearn, "./tmp/sklearn.exe")
    download_and_run(urlmatplotlib, "./tmp/matplotlib.exe")
    download_and_run(urlcython, "./tmp/cython.exe")

    
    # install pydicom
    subprocess.call(pythondir + "Scripts/pip.exe pydicom", cwd="./tmp/")
    subprocess.call(pythondir + "Scripts/pip.exe pyyaml", cwd="./tmp/")
    

    # install gco_python
    print ("gco_python install")
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
    print ("gco compilation")
    
    # there is a bug in python mingw compiler with -mno-cygwin, so here is simple patch
    
    local_file_name = pythondir + "Lib\distutils\cygwinccompiler.py"
    urllibr.urlretrieve(urlcompiler, local_file_name)
    
    # compilation with mingw
    subprocess.call(pythondir + "python.exe setup.py build_ext -i --compiler=mingw32", cwd="./tmp/gco_python-master/")
    # parametr -i by to mel nainstalovat sam, ale nedala to
    subprocess.call(pythondir + "python.exe setup.py build --compiler=mingw32", cwd="./tmp/gco_python-master/")    
    subprocess.call(pythondir + "python.exe setup.py install --skip-build", cwd="./tmp/gco_python-master/")    

    

    #subprocess.call('envinstall\envwindows.bat')

def download_and_run(url, local_file_name):
    urllibr.urlretrieve(url, local_file_name)
    subprocess.call(local_file_name)




# download sample data
print('Downloading sample data')

try:
    os.mkdir('sample_data')
except:
    pass

url =  "http://www.mathworks.com/matlabcentral/fileexchange/2762-dicom-example-files?download=true"

local_file_name = './sample_data/head.zip'

urllibr.urlretrieve(url, local_file_name)

datafile = zipfile.ZipFile(local_file_name)
datafile.extractall('./sample_data/')

print('Installing system environment')
if sys.platform.startswith('linux'):
    
    subprocess.call('./envinstall/envubuntu.sh')
    submodule_update()
elif sys.platform.startswith('win'):
    windows_install()
    submodule_update()
                        
