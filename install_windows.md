Install on Windows
====

* install [conda](https://conda.io/miniconda.html)
* install [Microsoft Visual C++ compiler for python 2.7.](https://download.microsoft.com/download/7/9/6/796EF2E4-801B-4FC4-AB28-B59FBF6D907B/VCForPython27.msi) or
 for python 3.6 [Build Tools for Visual Studio 2017](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)
* install Lisa using conda from cmd

      conda create -n lisa36 python=3.6
      activate lisa36
      conda install pip
      conda install -c mjirik -c SimpleITK -c menpo -c conda-forge pip lisa pygco
      
### Problems on Windows

* pygco install failed 
    1) install with pip can help. Try:
        
            pip install pygco
        
    2) download [compiled `.pyd` file](http://home.zcu.cz/~mjirik/lisa/install/pygco.cp36-win_amd64_site-packages.zip) 
and put it into conda's site-package directory: `Miniconda3\envs\lisa36\Lib\site-packages` 

* Unable to install SimpleITK

    * install version 1.0.1

For developers
=====

* Download and install [Git](http://www.git-scm.com/download/win)
     Select "Run Git from the Windows Command Prompt" or "Run Git and included Unix tools from the Windows Command Prompt"
* Download and install [CMake](http://www.cmake.org/download/#latest)



Install Windows with PythonXY
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
        curl -O https://raw2.github.com/mjirik/pyseg_base/master/distr/gco_python.pyx.patch
        patch gco_python.pyx < gco_python.pyx.patch
        python setup.py build_ext -i --compiler=mingw32
        python.exe setup.py build --compiler=mingw32
        python.exe setup.py install --skip-build
        
* Get requested modules

        pip install imcut dicom2fem sed3

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

