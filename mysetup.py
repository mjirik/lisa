#import urllib.request
import urllib
import os
import zipfile
import subprocess
import sys




# download sample data
print('Downloading sample data')

try:
    os.mkdir('sample_data')
except:
    pass

url =  "http://www.mathworks.com/matlabcentral/fileexchange/2762-dicom-example-files?download=true"

local_file_name = './sample_data/head.zip'

urllib.urlretrieve(url, local_file_name)

datafile = zipfile.ZipFile(local_file_name)
datafile.extractall('./sample_data/')

print('Installing system environment')
if sys.platform.startswith('linux'):
    
    subprocess.call('./envinstall/envubuntu.sh')
elif sys.platform.startswith('win'):

    # instalace msys gitu
    url = "http://msysgit.googlecode.com/files/Git-1.8.0-preview20121022.exe"
    try:
        os.mkdir("tmp")
    except:
        pass
            
    local_file_name = './tmp/git-1.8.0.exe'
    urllib.urlretrieve(url, local_file_name)
    subprocess.call(local_file_name)

    # install MinGW
    url = "http://downloads.sourceforge.net/project/mingw/Installer/mingw-get/mingw-get-0.5-beta-20120426-1/mingw-get-0.5-mingw32-beta-20120426-1-bin.zip?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fmingw%2Ffiles%2FInstaller%2Fmingw-get%2Fmingw-get-0.5-beta-20120426-1%2Fmingw-get-0.5-mingw32-beta-20120426-1-bin.zip%2Fdownload%3Fuse_mirror%3Dgarr%26r%3D%26use_mirror%3Dgarr&ts=1358892903&use_mirror=switch"
    local_file_name = './tmp/mingw-get.zip'
    urllib.urlretrieve(url, local_file_name)

    try:
        os.mkdir("./tmp/mingw-get")
    except:
        pass
    datafile = zipfile.ZipFile(local_file_name)
    datafile.extractall('./tmp/mingw-get')

    subprocess.call("./tmp/mingw-get/bin/mingw-get.exe install mingw-get")
    subprocess.call("./tmp/mingw-get/bin/mingw-get.exe install gcc")
    subprocess.call("./tmp/mingw-get/bin/mingw-get.exe install g++")

    import pdb; pdb.set_trace()

    subprocess.call('./envinstall/envwindows.bat')


# update submodules codes
print ('Updating submodules')
try:
    #import pdb; pdb.set_trace()
    subprocess.call('git submodule update --init --recursive', shell=True)
    #subprocess.call('git submodule update --init --recursive')

except:
    print ('Probem with git submodules')
