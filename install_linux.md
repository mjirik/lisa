Requirements
============

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
* VTK (optional) - The Visualization Toolkit (http://www.vtk.org/)


Install (L)Ubuntu 14.04
=======================

### Script way

    wget https://raw.githubusercontent.com/mjirik/lisa/master/installer.sh -O installer.sh
    source installer.sh

### Manual way


Tested with Ubuntu 14.04 and Linux Mint 16 Petra

    # 1. deb package requirements
    sudo apt-get install python git python-dev g++ python-numpy python-scipy python-matplotlib python-sklearn python-skimage python-dicom cython python-yaml sox make python-qt4 python-vtk python-setuptools curl

    # 2. install miniconda
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
    bash Miniconda-latest-Linux-x86_64.sh -b
    cd ~
    export PATH=`pwd`/miniconda/bin:$PATH

    # 2. easy_install requirements simpleITK  
    sudo easy_install -U SimpleITK mahotas

    # 3. pip install our packages pyseg_base and dicom2fem
    sudo pip install pysegbase dicom2fem sed3 sed3 io3d skelet3d ipdb
    
    # 4. install gco_python
    mkdir ~/projects
    cd ~/projects
    git clone https://github.com/mjirik/gco_python.git
    cd gco_python
    make
    sudo python setup.py install
    cd ..
    
    # 5. skelet3d - optional for Histology Analyser
    sudo apt-get install cmake python-numpy libinsighttoolkit3-dev libpng12-dev
    cd ~/projects
    git clone https://github.com/mjirik/skelet3d.git
    cd skelet3d
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    

1. First use package manager to satisfy requirements. 
2. SimpleITK is not in ubuntu packages. You can use easy_install
3. Our support packages can be installed with pip
4. For pygco use following (more info https://github.com/mjirik/pyseg_base/blob/master/INSTALL)
5. Last step is optional for Histology Analyser and some functions Lisa

Get stable branche

    git clone --recursive -b stable https://github.com/mjirik/lisa.git

or for current developement (if you want to participate)

    git clone --recursive git@github.com:mjirik/lisa.git


Make an icon

    cd lisa
    python mysetup.py -icn
    
Download sample data

    cd ~/projects/lisa
    python mysetup.py -d
    mkdir ~/lisa_data
    cp -r sample_data/ ~/lisa_data/

Test

    cd ~/projects/lisa
    nosetests
    

Install (L)Ubuntu 12.04 (13.10)
=========================

Use package manager

    sudo apt-get install python git python-numpy python-scipy python-matplotlib python-sklearn python-dicom cython python-yaml sox python-insighttoolkit3 make python-qt4 python-vtk python-setuptools
    
SimpleITK is not in ubuntu packages. You can use easy_install

    sudo easy_install -U SimpleITK
    
For pygco use following (more info https://github.com/mjirik/pyseg_base/blob/master/INSTALL)

    git clone https://github.com/amueller/gco_python.git
    cd gco_python
    make
    python setup.py install

Get stable branche

    mkdir ~/projects
    cd ~/projects
    git clone --recursive -b stable https://github.com/mjirik/lisa.git

    
Problems on Linux:

* Permission denied. You can try following command if there is a problem "Permission denied"


    sudo chmod a+r /usr/local/lib/python2.7/dist-packages/SimpleITK-0.7.0-py2.7-linux-x86_64.egg/EGG-INFO/top_level.txt





Testování funkčnosti windows pomocí wine (czech)
========================================


    sudo apt-get install wine

    mkdir ~/tmp/winpython

    wget -P ~/tmp/winpython/ http://msysgit.googlecode.com/files/Git-1.8.0-preview20121022.exe

# Install with linux tools
    wine ~/tmp/winpython/Git-1.8.0-preview20121022.exe


Next, Next, Next, Finish
    wget -P ~/tmp/winpython/ http://www.python.org/ftp/python/2.7.3/python-2.7.3.msi
    wine msiexec /i /home/mjirik/tmp/winpython/python-2.7.3.msi

    wget -P ~/tmp/winpython/  http://www.cmake.org/files/v2.8/cmake-2.8.10.2-win32-x86.exe
    wine ~/tmp/winpython/cmake-2.8.10.2-win32-x86.exe

Add CMake to system PATH for all users


Balíky (numpy a scipy) lze získad z následujících stránek, 

http://www.lfd.uci.edu/~gohlke/pythonlibs/

    
Teoreticky by mohlo jít stáhnout soubory přes wget, ale občas se změní 
název souboru (za něj se přidá znak '$'), nebo se soubor nestáhne celý

    wget -P ~/tmp/winpython/ http://www.lfd.uci.edu/~gohlke/pythonlibs/z86mtkth/numpy-MKL-1.6.2.win32-py2.7.exe
    wget -P ~/tmp/winpython/ http://www.lfd.uci.edu/~gohlke/pythonlibs/z86mtkth/scipy-0.11.0.win32-py2.7.exe


instalace pomocí pip

Nejprve potřebujeme distribute a pip

    wget -P ~/tmp/winpython/ http://python-distribute.org/distribute_setup.pyp
    wine c:\\python27\\python /home/mjirik/tmp/winpython/distribute_setup.py



    wget -P ~/tmp/winpython/ https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    wine c:\\python27\\python /home/mjirik/tmp/winpython/get-pip.py


cython nainstalujeme pomocí pip, možná by tak šlo instalovat i numpy a scipy
    wine cmd
    c:\\python27\\scripts\\pip install cython
    c:\\python27\\scripts\\pip install matplotlib
    exit

instalace gco

    wget -P ~/tmp/winpython/ http://vision.csd.uwo.ca/code/gco-v3.0.zip
    unzip /home/mjirik/tmp/winpython/gco-v3.0.zip -d/home/mjirik/tmp/winpython/gco/

    cd ~/tmp/winpython/
    wine cmd

git clone pycat



