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
* ITK (optional) - Package for medical image analysis (http://www.itk.org/)



Install
-------

See our install notes for Linux, Mac OS and Windows (https://github.com/mjirik/lisa/blob/master/INSTALL.md)


Get stable branche

    git clone --recursive -b stable https://github.com/mjirik/lisa.git

or for current developement

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
    
    
