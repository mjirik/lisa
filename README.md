LISA 
=============

LIver Surgery Analyser.



Requirements
------------

Installing requires you to have installed:

* GIT - distributed version control system (http://git-scm.com)
* numpy (http://www.numpy.org)
* scipy (http://scipy.org)
* scikit-learn (http://scikit-learn.org)
* Cython - C-extension for Python (http://cython.org)
* pyqt - Python bindings for Qt application framework
(http://www.riverbankcomputing.com/software/pyqt)
* pygco - Graphcuts for Python (https://github.com/amueller/gco_python)
* pydicom - package for working with DICOM files
(http://code.google.com/p/pydicom)
* pyqt - QT4 for python (https://wiki.python.org/moin/PyQt)

Simple install instructions fallows. In case of problem see (https://github.com/mjirik/lisa/blob/master/INSTALL.md)

Linux:

use package manager of your distribution

    sudo apt-get install python git python-numpy python-scipy python-matplotlib python-sklearn python-dicom cython python-yaml sox python-insighttoolkit3 python-qt4 python-setuptools make

SimpleITK is not in ubuntu packages. You can use easy_install

    sudo easy_install -U SimpleITK
    
For pygco use following (more info https://github.com/mjirik/pyseg_base/blob/master/INSTALL)

    git clone https://github.com/amueller/gco_python.git
    cd gco_python
    make
    sudo python setup.py install


On Window, you can use Python XY (http://code.google.com/p/pythonxy/) and
packages by Christoph Gohlke (http://www.lfd.uci.edu/~gohlke/pythonlibs/)

On Mac, see notes.txt 
(https://github.com/mjirik/liver-surgery/blob/master/notes.txt)  
and pyseg_base install notes 
(https://github.com/mjirik/pyseg_base/blob/master/INSTALL)





Install
-------



    git clone --recursive git@github.com:mjirik/lisa.git

    python ./mysetup.py

or

    git clone git@github.com:mjirik/lisa.git

    git submodule update --init --recursive


You can find more install notes in file 'notes.md'
(https://github.com/mjirik/liver-surgery/blob/master/notes.md)

Get sample data
---------------

    python mysetup.py -d



Run
---

Object (usualy liver) extraction is started by organ_segmentation script

    python python/organ_segmentation.py 

Segmentation use two types of seeds wich are setted by left and right mouse 
button. For better volume measurement control use additional parameters 
for example "-mroi" and "-vs 0.8". If "-mroi" parameter is used you can 
select region of interest.


Vessel localization uses saved data (organ.pkl) from organ_segmentation:

    python src/vessels_segmentation.py -bg -i organ.pkl

Now user interactivity is used to set threshold parametr.

Virtual liver resection is based on data stored in previous step 
(vessels.pkl).

    python src/vessel_cut.py -oe -i vessels.pkl

In this script is selected cut on vessel by user interactivity. Resected and
remaining volume is then calculated.


Tests
-----

    nosetests
    
    
