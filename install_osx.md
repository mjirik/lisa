Install Mac OS (Lisa until version 1.4)
==============

 * Xcode, gcc and make

    Install Xcode from appstore. You will need an AppleID.
    Then add terminal tools (http://stackoverflow.com/questions/10265742/how-to-install-make-and-gcc-on-a-mac)
    * Start XCode
    * Go to XCode/Preferences.
    * Click the "Downloads" tab.
    * Click "Components".
    * Click "Install" on the command line tools line.


 * MacPorts

    Use standard pkg package. (http://www.macports.org/install.php)
    
    Tested on [OS X 10.8 Mountain Lion installer](https://distfiles.macports.org/MacPorts/MacPorts-2.2.1-10.8-MountainLion.pkg)
    and [OSX 10.9 Mavericks installer](https://distfiles.macports.org/MacPorts/MacPorts-2.2.1-10.9-Mavericks.pkg)


 * GIT

    Use mac ports

        sudo port install git-core +svn +doc +bash_completion +gitweb

    Use mac installer  (http://git-scm.com/download/mac)
    There is a need to allow install applications from unknown developers
    (Settings - Security & Privacy - General - Allow applications downloaded from)

 * Numpy, Scipy, ...

c
        sudo port selfupdate
        sudo port upgrade outdated
        sudo port install py27-pyqt4 py27-numpy py27-scipy py27-matplotlib py27-ipython +notebook py27-pandas py27-sympy py27-nose  py-scikit-learn py-pydicom py27-yaml py27-cython vtk5 +qt4_mac +python27 py27-distutils-extra
 

 * Select default python


        sudo port select --set python python27

 * Cython (may work from port)


        sudo easy_install cython
        
    or

        sudo -E easy_install cython

 * gco_python

    Try this

        git clone https://github.com/amueller/gco_python.git
        cd gco_python
        make
        sudo -E python setup.py install

    or this

    See install notes to pyseg_base (https://github.com/mjirik/pyseg_base/blob/master/INSTALL).
    If there is a problem with clang compiler, you are should edit gco 
    source files and add "this->" string. Also you can use patch from pyseg_base

 * source files


        git clone git@github.com:mjirik/pyseg_base.git
        git submodule update --init --recursive


### Known problems on Mac OS 

 * Cython installation fail

        command 'cc' failed with exit status 1
        
    or
    
        import Cython.Distutils

    Install cython with easy_install 
    
        sudo -E easy_install cython
    
    


