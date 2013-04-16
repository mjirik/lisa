#!/bin/bash 

tmpd=~/tmp/livertmp


# get path to this script
directoryx="$(dirname -- $(readlink -fn -- "$0"; echo x))"
directory="${directoryx%x}"


command -v apt-get >/dev/null 2>&1 || { echo >&2 "I require apt-get but it's not installed..";exit;} 
sudo apt-get install python-numpy python-scipy python-matplotlib python-sklearn python-dicom python-yaml
# gco_python install
# -------------------
# 
# see  http://peekaboo-vision.blogspot.cz/2012/05/graphcuts-for-python-pygco.html


echo 'Install gco_python'

sudo apt-get install cython
mkdir $tmpd
cd $tmpd

git clone https://github.com/amueller/gco_python.git
cd gco_python

make

#this will crash but dowload  gco_src is ok

#you need to include stddef.h in GCoptimization.h

sed -i '111 i\#include "stddef.h"' gco_src/GCoptimization.h

#again

make

#and final install
sudo python setup.py install


#example
#python example.py

