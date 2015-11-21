Install (L)Ubuntu 14.04
=======================

### Script installer

    wget https://raw.githubusercontent.com/mjirik/lisa/master/installer.sh -O installer.sh
    source installer.sh
    

You can run `installer.py` with parameter `devel` or `noclone` to control source files cloning

More information read [linux install notes](https://github.com/mjirik/lisa/blob/master/install_linux.md)


Install Mac OS
==============

 * Xcode, gcc and make

    Install Xcode from appstore. You will need an AppleID.
    Then add terminal tools (http://stackoverflow.com/questions/10265742/how-to-install-make-and-gcc-on-a-mac)
    * Start XCode
    * Go to XCode/Preferences.
    * Click the "Downloads" tab.
    * Click "Components".
    * Click "Install" on the command line tools line.


 * [Install Anaconda](http://continuum.io/downloads)

 * All other dependencies install with script


        curl https://raw.githubusercontent.com/mjirik/lisa/master/installer.sh -o installer.sh
        source installer.sh

     You can run `installer.py` with parameter `devel` or `noclone` to control source files cloning
 
   

Install Windows with Anaconda
=========

* Download and install [miniconda](http://conda.pydata.org/miniconda.html)
* Download and install [Git](http://www.git-scm.com/download/win)
* Download and install [MS Visual C++ compiler](http://aka.ms/vcpython27)

     Select "Run Git from the Windows Command Prompt" or "Run Git and included Unix tools from the Windows Command Prompt"
* Download and install [CMake](http://www.cmake.org/download/#latest)
* Run command line and create conda-virtualenv


        conda create --no-default-packages -n lisa pip
        activate lisa

* In activated lisa virtualenv run following lines


        pip install wget
        python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_conda.txt
        conda install -y --file requirements_conda.txt
        python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_pip.txt
        pip install -r requirements_pip.txt
        easy_install SimpleITK mahotas
        mkdir projects
        cd projects
        
* gco_python
    
    Fallowing lines downloads gco and gco_python. Then it makes build with mingw compiler and install. 


        git clone https://github.com/amueller/gco_python.git
        cd gco_python
        mkdir gco_src && cd gco_src
        curl -O http://vision.csd.uwo.ca/code/gco-v3.0.zip
        unzip gco-v3.0.zip
        cd ..
        curl -O https://raw2.github.com/mjirik/pyseg_base/master/distr/gco_python.pyx.patch
        patch gco_python.pyx < gco_python.pyx.patch
        python setup.py build_ext -i --compiler=mingw32
        python setup.py build --compiler=mingw32
        python setup.py install --skip-build
        cd ..
        
        python -m wget http://downloads.sourceforge.net/project/itk/itk/4.7/InsightToolkit-4.7.1.zip?r=http%3A%2F%2Fwww.itk.org%2FITK%2Fresources%2Fsoftware.html&ts=1424632138&use_mirror=softlayer-ams
        unzip InsightToolkit-4.7.1.zip
        
        git clone https://github.com/mjirik/skelet3d.git
        mkdir skelet3d\build
        cd skelet3d\build
        cmake ..
        make
        sudo make install
        cd ..
        cd ..
        
        git clone --recursive -b stable https://github.com/mjirik/lisa.git




Use VirtualBox (old Lisa version)
==============

* Install VirtualBox (https://www.virtualbox.org/)
* Download Lisa Image (http://uloz.to/xU4oHfKw/lisa-ubuntu14-04-vdi)

or 
* Download Lisa Image (http://147.228.240.61/queetech/install/lisa_ubuntu14.04.vdi)

In VirtualBox

* Create new computer

    * Name: Lisa
    * Type: Linux
    * Version: Ubuntu (32bit)
    
* Set memory size to 1024MB and more
* Use existing hard disk and locate downloaded Lisa Image (lisa_ubuntu14.04.vdi)
* Password to Lisa account is: L1v3r.


