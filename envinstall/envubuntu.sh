#!/bin/bash 

tmpdir=~/tmp/winpython


# get path to this script
directoryx="$(dirname -- $(readlink -fn -- "$0"; echo x))"
directory="${directoryx%x}"

sudo apt-get install python-numpy python-scipy python-matplotlib python-sklearn

# gco_python install
# -------------------
# 
# see  http://peekaboo-vision.blogspot.cz/2012/05/graphcuts-for-python-pygco.html


echo 'Install gco_python'

sudo apt-get install cython
mkdir $tmpdir
cd $tmpdir

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

