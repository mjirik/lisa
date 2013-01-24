set tmpdir=%home%\tmpwinpython
set projectdir=%home%\liver-surgery
#set pythondir=c:\python33
set pythondir=c:\python27
set installdir=%home%\liver



rem mkdir c:/tmp
rem mkdir c:/tmp/winpython
mkdir %tmpdir%
mkdir %installdir%
mkdir %projectdir%




curl -o %tmpdir%/python.msi http://www.python.org/ftp/python/2.7.3/python-2.7.3.msi
rem verze 3.3 win 64bit
::: curl -o %tmpdir%/python.msi http://www.python.org/ftp/python/3.3.0/python-3.3.0.amd64.msi
pause
rem msiexec /i /home/mjirik/tmp/winpython/python-2.7.3.msi
msiexec /i %tmpdir%/python.msi

curl -o %tmpdir%/cmakeinst.exe  http://www.cmake.org/files/v2.8/cmake-2.8.10.2-win32-x86.exe

rem c:/tmp/winpython/cmake-2.8.10.2-win32-x86.exe
%tmpdir%/cmakeinst.exe
    

rem distribute a pip
rem download and install
curl -o %tmpdir%/distribute_setup.py http://python-distribute.org/distribute_setup.py
%pythondir%/python %tmpdir%/distribute_setup.py

curl -o %tmpdir%/get-pip.py https://raw.github.com/pypa/pip/master/contrib/get-pip.py
%pythondir%/python %tmpdir%/get-pip.py

rem baliky pro python

%pythondir%/scripts/pip install numpy
%pythondir%/scripts/pip install scipy

echo If there are problems with installation numpy and scipy
echo you can download binaries from
echo http://www.lfd.uci.edu/~gohlke/pythonlibs/


pause

rem
rem curl -o %tmpdir%numpy.exe http://www.lfd.uci.edu/~gohlke/pythonlibs/z86mtkth/numpy-MKL-1.6.2.win32-py2.7.exe
rem curl -o %tmpdir%scipy.exe http://www.lfd.uci.edu/~gohlke/pythonlibs/z86mtkth/scipy-0.11.0.win32-py2.7.exe

rem %tmpdir%numpy.exe
rem %tmpdir%scipy.exe
curl -o %tmpdir%/mingw.exe http://downloads.sourceforge.net/project/mingw/Installer/mingw-get-inst/mingw-get-inst-20120426/mingw-get-inst-20120426.exe?r=&ts=1356869096&use_mirror=ignum

rem http://downloads.sourceforge.net/project/mingw/Installer/mingw-get-inst/mingw-get-inst-20120426/mingw-get-inst-20120426.exe?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fmingw%2Ffiles%2FInstaller%2Fmingw-get-inst%2Fmingw-get-inst-20120426%2F&ts=1356868906&use_mirror=ignum
%tmpdir%/mingw.exe

http://sourceforge.net/projects/mingw/files/Installer/mingw-get-inst/mingw-get-inst-20120426/mingw-get-inst-20120426.exe/download
http://sourceforge.net/projects/mingw/files/Installer/mingw-get-inst/mingw-get-inst-20120426/

%pythondir%/scripts/pip install cython
%pythondir%/scripts/pip install matplotlib

echo install python modules from zip
::: curl -o ...
unzip %tmpdir%/pymodules27.zip -d%tmpdir%
%tmpdir%/pymodules27/numpy-MKL-1.6.2.win32-py2.7.exe
%tmpdir%/pymodules27/scipy-0.11.0.win32-py2.7.exe
%tmpdir%/pymodules27/scikit-learn-0.13.win32-py2.7.exe
%tmpdir%/pymodules27/matplotlib-1.2.0.win32-py2.7.exe
%tmpdir%/pymodules27/Cython-0.17.4.win32-py2.7.exe


echo install mingw
pause

rem install gco
rem ===========

curl -o %tmpdir%/gco.zip http://vision.csd.uwo.ca/code/gco-v3.0.zip
mkdir %tmpdir%/gco/
unzip %tmpdir%/gco.zip -d%tmpdir%gco/

curl -o %tmpdir%/gco/CMakeLists.txt https://raw.github.com/mjirik/pycat/master/extern/CMakeLists.txt.in

mkdir %tmpdir%/gco/build
cd %tmpdir%/gco/build
pause
cmake ..


:git clone 
