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
	urlpython = "http://www.python.org/ftp/python/3.2.3/python-3.2.3.amd64.msi"
	urlmsysgit = "http://msysgit.googlecode.com/files/Git-1.8.0-preview20121022.exe"
	urlmingw = "http://downloads.sourceforge.net/project/mingw/Installer/mingw-get-inst/mingw-get-inst-20120426/mingw-get-inst-20120426.exe?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fmingw%2Ffiles%2Flatest%2Fdownload%3Fsource%3Dfiles&ts=1359726876&use_mirror=ignum",
	urlnumpy = "http://home.zcu.cz/~mjirik/liver/download/numpy-MKL-1.6.2.win-amd64-py3.2.exe"
    urlscipy = "http://home.zcu.cz/~mjirik/liver/download/scipy-0.11.0.win-amd64-py3.2.exe"
    urlsklearn = "http://home.zcu.cz/~mjirik/liver/download/scikit-learn-0.13.win-amd64-py3.2.exe"
    urlmatplotlib = "http://home.zcu.cz/~mjirik/liver/download/matplotlib-1.2.0.win-amd64-py3.2.exe"
    urlcython = "http://home.zcu.cz/~mjirik/liver/download/Cython-0.18.win-amd64-py3.2.exe"
	urlgco_python = "https://github.com/amueller/gco_python/archive/master.zip"
	urlgco = "http://vision.csd.uwo.ca/code/gco-v3.0.zip"
	urlcompiler = "http://home.zcu.cz/~mjirik/liver/download/cygwinccompiler.py"
	
def windows_install():
    try:
        os.mkdir("tmp")
    except:
        pass
    definitions_win64_py32()
    """
    # check correct python version
    if sys.version_info < (3,2):
        
        print ("Different python required \n installing python 3.2 64-bit")
        print (sys.version_info)
        
        local_file_name = 'tmp\python.msi'
        urllibr.urlretrieve(urlpython, local_file_name)
        a = "msiexec /i " + local_file_name
        print (a)
        subprocess.call(a)

        print ("Please restart installer with new python")
        subprocess.call("c:\python32\python.exe mysetup.py")
        return
    else:
        print ("You have python 3.2")

        


    # instalace msys gitu
    print ("MSYS Git install")
    import pdb; pdb.set_trace()
    download_and_run(
        url = urlmsysgit,
        local_file_name = './tmp/git-1.8.0.exe'
        )

    # install MinGW
    print ("MinGW install")
    download_and_run(
        url = urlmingw
        local_file_name = './tmp/mingw.exe'
        )

    # numpy, scipy, matplotlib, scikit-learn, cython
    print ("numpy, scipy, matplotlib, scikit-learn, cython install")
    import pdb; pdb.set_trace()
    download_and_run(urlnumpy, "./tmp/numpy.exe")
    download_and_run(urlscipy, "./tmp/scipy.exe")
    download_and_run(urlsklearn, "./tmp/sklearn.exe")
    download_and_run(urlmatplotlib, "./tmp/matplotlib.exe")
    download_and_run(urlcython, "./tmp/cython.exe")

    """

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
    
    local_file_name = "c:\python32\Lib\distutils\cygwinccompiler.py"
    urllibr.urlretrieve(urlcompiler, local_file_name)
    
    import pdb; pdb.set_trace()
    subprocess.call("c:\python32\python.exe setup.py build_ext -i --compiler=mingw32", cwd="./tmp/gco_python-master/")
    # tohle je nepotrebne, protoze parametr -i to nainstaluje sam
    #subprocess.call("c:\python32\python.exe setup.py install --skip-build", cwd="./tmp/gco_python-master/")    
    import pdb; pdb.set_trace()
                        
    # pomoci mingw-get tohle moc nefunguje
    """
    url = "http://downloads.sourceforge.net/project/mingw/Installer/mingw-get/mingw-get-0.5-beta-20120426-1/mingw-get-0.5-mingw32-beta-20120426-1-bin.zip?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fmingw%2Ffiles%2FInstaller%2Fmingw-get%2Fmingw-get-0.5-beta-20120426-1%2Fmingw-get-0.5-mingw32-beta-20120426-1-bin.zip%2Fdownload%3Fuse_mirror%3Dgarr%26r%3D%26use_mirror%3Dgarr&ts=1358892903&use_mirror=switch"
    local_file_name = './tmp/mingw-get.zip'
    urllibr.urlretrieve(url, local_file_name)


    try:
        os.mkdir("./tmp/mingw-get")
    except:
        pass
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./tmp/mingw-get')

    subprocess.call("./tmp/mingw-get/bin/mingw-get.exe install mingw-get")
    subprocess.call("./tmp/mingw-get/bin/mingw-get.exe install gcc")
    subprocess.call("./tmp/mingw-get/bin/mingw-get.exe install g++")
    """
    # pomoci mingw-get-inst
    #print("C Compiler, maybe C++ Compiler, MSYS Basic System, MinGW Developer ToolKit")
    #url = "http://downloads.sourceforge.net/project/mingw/Installer/mingw-get-inst/mingw-get-inst-20120426/mingw-get-inst-20120426.exe?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fmingw%2Ffiles%2Flatest%2Fdownload%3Fsource%3Dfiles&ts=1358934475&use_mirror=freefr"
    #local_file_name = './tmp/mingw-get-inst.exe'
    #urllibr.urlretrieve(url, local_file_name)
    #subprocess.call(local_file_name)

    import pdb; pdb.set_trace()

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
                        
