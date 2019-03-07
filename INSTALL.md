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

    ```bash
    curl https://raw.githubusercontent.com/mjirik/lisa/master/installer.sh -o installer.sh
    source installer.sh
    ```

    You can run `installer.py` with parameter `devel` or `noclone` to control source files cloning
 
   

Install Lisa for Windows with Anaconda
=========

Use [windows installer](http://147.228.240.61/queetech/install/setup_lisa.exe)

or

* Download and install [miniconda](http://conda.pydata.org/miniconda.html)
* Download and install [C++ Compiler](https://wiki.python.org/moin/WindowsCompilers) 
    * Python 2.7: [MS Visual C++ compiler for Python 2.7](http://aka.ms/vcpython27)
    * Python 3.6: [Microsoft Build Tools for Visual Studio 2017](
    https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)
    
        Check "Python development" and "Desktop Application C++ development" during install. 
        You may remove all submodules. Keep just VC++ 2017 tools.
    
* Run command line and create conda-virtualenv
    ```bat
    conda create --no-default-packages -y -c mjirik -c SimpleITK -n lisa pip lisa
    activate lisa
    ```
    
* In activated lisa virtualenv run following lines to satisfy some requirements
    ```bash
    python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_pip.txt
    pip install -r requirements_pip.txt
    ```

* You can have Lisa from sources 
    ```bash
    conda install -y -c mjirik -c SimpleITK --file requirements_conda.txt
    git clone https://github.com/mjirik/lisa.git
    ```   
        
    or from conda package
    ```bash
    conda install -y -c mjirik -c SimpleITK lisa
    ```
        

* Run Lisa
    
    ```bash
    activate lisa
    python -m lisa
    ```



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


