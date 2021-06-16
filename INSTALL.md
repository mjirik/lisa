Install (L)Ubuntu 14.04
=======================

### Install with conda

    conda install -c mjirik -c conda-forge -c simpleitk -c menpo -c luispedro lisa
    pip install imcut pygco

### Install with conda for development

Working with `requirements_conda.txt` file.

    conda install -c mjirik -c conda-forge -c simpleitk -c menpo -c luispedro --file requirements_conda.txt
    pip install imcut pygco

There is imcut and pygco package for conda on windows and linux. 
It should be possible to install both of theme just with conda.


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

1) Download and install [miniconda](http://conda.pydata.org/miniconda.html)
2) Download and install [C++ Compiler](https://wiki.python.org/moin/WindowsCompilers) 
    * Python 2.7: [MS Visual C++ compiler for Python 2.7](http://aka.ms/vcpython27)
    * Python 3.6-3.8: [Visual C++ 14.X](https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.2_standalone:_Build_Tools_for_Visual_Studio_2019_.28x86.2C_x64.2C_ARM.2C_ARM64.29)
    
        Check "Python development" and "Desktop Application C++ development" during install. 
        You may remove all submodules. Keep just VC++ 2017 tools.
        
3) Install python packages. You have two options
    * Install environment with GitHub clone
        ```bash
        git clone https://github.com/mjirik/lisa.git
        cd lisa
        conda create -n lisa --yes -c conda-forge -c mjirik -c SimpleITK -c menpo -c luispedro --file requirements_conda.txt
        ```   
            
    * Install conda package
        ```bash
        conda create -n lisa --yes -c conda-forge -c mjirik -c SimpleITK -c menpo -c luispedro lisa
        ```
        and then download [`requirements_pip.txt`](https://raw.githubusercontent.com/mjirik/lisa/master/requirements_pip.txt)
        
        You can use commandlind utility:
        ```bash
        python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_pip.txt
        ```
      
4) Run command line and create conda-virtualenv
        ```bat
        activate lisa
        ```
        
5) In activated lisa virtualenv run following lines to satisfy some requirements
        ```bash
        pip install -r requirements_pip.txt
        ```

6) Run Lisa
    
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


