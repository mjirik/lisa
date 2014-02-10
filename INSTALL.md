Install (L)Ubuntu 13.10
=========================

Use package manager

    sudo apt-get install python git python-numpy python-scipy python-matplotlib python-sklearn python-dicom cython python-yaml sox python-insighttoolkit3 make python-qt4 python-vtk python-setuptools
    
SimpleITK is not in ubuntu packages. You can use easy_install

    sudo easy_install -U SimpleITK
    
For pygco use following (more info https://github.com/mjirik/pyseg_base/blob/master/INSTALL)

    git clone git@github.com:mjirik/liver-surgery.git
    git submodule update --init --recursive
    git clone https://github.com/amueller/gco_python.git
    cd gco_python
    make
    python setup.py install
    
Problems on Linux:

* Permission denied. You can try following command if there is a problem "Permission denied"


    sudo chmod a+r /usr/local/lib/python2.7/dist-packages/SimpleITK-0.7.0-py2.7-linux-x86_64.egg/EGG-INFO/top_level.txt




Install Mac OS
==============


 * GIT
 
    Use mac installer  (http://git-scm.com/download/mac)
    There is a need to allow install applications from unknown developers
    (Settings - Security & Privacy - General - Allow applications downloaded from)

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
    Tested on Mountain Lion installer 
    (https://distfiles.macports.org/MacPorts/MacPorts-2.2.1-10.8-MountainLion.pkg


 * Numpy, Scipy, ...


        sudo port selfupdate
        sudo port upgrade outdated
        sudo port install py27-pyqt4 py27-numpy py27-scipy py27-matplotlib py27-ipython +notebook py27-pandas py27-sympy py27-nose  py-scikit-learn py-pydicom py27-yaml py27-cython vtk5 +qt4_mac +python27
 

 * Select default python


        sudo port select --set python python27

 * Cython (may work from port)


        sudo easy_install cython

 * gco_python

    See install notes to pyseg_base (https://github.com/mjirik/pyseg_base/blob/master/INSTALL).
    If there is a problem with clang compiler, you are should edit gco 
    source files and add "this->" string. Also you can use patch from pyseg_base

 * source files


        git clone git@github.com:mjirik/pyseg_base.git
        git submodule update --init --recursive





Install Windows
======================


* Python XY (http://code.google.com/p/pythonxy/)
    
    Add all packages.
* Git (http://www.git-scm.com/download/win)

    Select "Run Git from the Windows Command Prompt" or "Run Git and included Unix tools from the Windows Command Prompt"
* gco_python
    
    Fallowing lines downloads gco and gco_python. Then it makes build with mingw compiler and install. 


        git clone https://github.com/amueller/gco_python.git
        cd gco_python
        mkdir gco_src && cd gco_src
        curl -O http://vision.csd.uwo.ca/code/gco-v3.0.zip
        unzip gco-v3.0.zip
        cd ..
        python setup.py build_ext -i --compiler=mingw32
        python.exe setup.py build --compiler=mingw32
        python.exe setup.py install --skip-build

Problems on Windows:

* Cython not found. Install cython from https://code.google.com/p/pythonxy/wiki/Downloads#Plugin_updates
* UnicodeDecodeError: 'ascii' codec can't decode byte 0x82 in position 0: ordinal
not in range(128)

    You can either find and remove the offending font (the best idea) or try to patch your installation using the following procedure:

    * Open the following in a text editor:

            \Users\dafonseca\AppData\Local\Enthought\Canopy\User\lib\site- packages\matplotlib\font_manager.py

    * Search for
    
            sfnt4 = sfnt4.decode('ascii').lower()
            
    * And replace with 
    
            sfnt4 = sfnt4.decode('ascii', 'ignore').lower()

    Note that this bug won't exist in the next release of matplotlib.

    For more information see: http://stackoverflow.com/questions/18689854/enthought-matplotlib-problems-with-plot-function

Windows Install - old version in czech
=================

http://www.lfd.uci.edu/~gohlke/pythonlibs/

1. Python (*)
2. Numpy (musi byt nainstalovan drive nez Scipy) (*)
3. Scipy (musi byt nainstalovan po Numpy) (*)

*) Poznamky:
	Je potreba pozorne volit 64/32 bit knihovny a nezamenovat je. Python, i knihovny museji byt jen 64 bit nebo jen 32 bit. Doporucujeme uzivat Numpy MKL. Jsou to vypocetni optimalizace, ktere pozdeji pozadujici navazujici baliky.

Errory:

1. V pripade zavaznejsich problemu (tzn. nefunkcnost) 	vymazat dane verze Pythonu, Numpy a Scipy a preinstalovat 	je.
2. Resit na strankach vyrobce softwaru.

Dodatecne instalace:

1. matplotlib
    Navod je zde: (http://matplotlib.org/users/installing.html)
    Melo by stacit ze stranky (http://sourceforge.net/projects/matplotlib/files/)
    ve slozce "matplotlib" stahnout prislusnou verzi matplotlib a nainstalovat po nainstalovani Numpy a Scipy.


gco_python v sedmi snadných krocích
-----------------------------------

ověřeno pro python 2.7 32   

1) Python 2.7 pro 32-bit architekturu

2) Zdrojové kódy

V zásadě je potřeba mít zdrojové kódy gco a gco_python.  
    
  * gco_python (https://github.com/amueller/gco_python/archive/master.zip)
  * gco (http://vision.csd.uwo.ca/code/gco-v3.0.zip)

Gco se normálně překládá pomocí matlabu. My to uděláme jinak.
Zdrojové kódy uspořádejme tak, že bude adresář gco_python se zdrojovými 
kódy gco_python a v něm bude adresář gco_src s kódy gco.

3) Překladač

Patrně lze použít jakýkoliv, ale mě to (po mnoha problémech) fungovalo s 
32-bitovým mingw

4) Moduly Pythonu

Musí se jednat o příslušné verze - 32bit
    
numpy scipy scikit-learn matplotlib cython pydicom pyyaml

lze nainstalovat pomocí pip:

    pip install numpy scipy scikit-learn matplotlib cython pydicom pyyaml

5) Oprava konfiguračního souboru v Pythonu
    V konfiguračním souboru je volání gcc se zastaralým parametrem -mno-cygwin.
    Je potřeba odstranit všechny výskyty tohoto slova z konfiguračního souboru:

    "C:\Python27\Lib\distutils\cygwinccompiler.py"

6) Kompilace

Všechny následující příkazy spouštíme v adresáři gco_python s pythonem 27
    
Překlad gco_src s překladačem mingw

    python.exe setup.py build_ext -i --compiler=mingw32

Vytvoření knihoven z gco_python

    python.exe setup.py build --compiler=mingw32
    python.exe setup.py install --skip-build

nakopírování knihoven na příslušná místa. Od teď půjde knihovna gco_python 
volat z jakéhokoliv python skriptu.

7) Ověření

    python example.py

Výsledek by se měl podobat obrázkům na adrese:
    http://peekaboo-vision.blogspot.cz/2012/05/graphcuts-for-python-pygco.html
    

nainstalovat mingw



Testování funkčnosti windows pomocí wine
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



