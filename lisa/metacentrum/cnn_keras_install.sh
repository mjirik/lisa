#!/usr/bin/bash

module add python27-modules-gcc

DIR=~/keras_104
mkdir -p $DIR
cd $DIR

pip install virtualenv --root ./virtualenv --process-dependency-links --ignore-installed
export PYTHONPATH=$PYTHONPATH:$DIR/virtualenv/software/python-2.7.6/gcc/lib/python2.7/site-packages

$DIR/virtualenv/software/python-2.7.6/gcc/bin/virtualenv keras-1.0.4

module rm python27-modules-gcc

module add python-2.7.6-gcc
export PYTHONPATH=$PYTHONPATH:$DIR/virtualenv/software/python-2.7.6/gcc/lib/python2.7/site-packages
source $DIR/keras-1.0.4/bin/activate

PYTHONUSERBASE=$DIR/keras-1.0.4/
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python2.7/site-packages:$PYTHONPATH

pip install --upgrade keras --ignore-installed --process-dependency-links
pip install nose
pip install jupyter
pip install matplotlib
