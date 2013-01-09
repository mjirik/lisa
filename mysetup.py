#import urllib.request
import urllib
import os
import zipfile
import subprocess
import sys

# update submodules codes
print ('Updating submodules')
try:
    #import pdb; pdb.set_trace()
    subprocess.call('git submodule update --init --recursive', shell=True)
    #subprocess.call('git submodule update --init --recursive')

except:
    print ('Probem with git submodules')


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
    subprocess.call('./envinstall/enwindows.bat')

