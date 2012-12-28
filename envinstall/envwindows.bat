set tmpdir=c:/tmpwinpython/
set installdir=c:/liver-surgery/

mkdir c:/tmp
mkdir c:/tmp/winpython



wget -P c:/tmp/winpython/ http://www.python.org/ftp/python/2.7.3/python-2.7.3.msi
msiexec /i /home/mjirik/tmp/winpython/python-2.7.3.msi

wget -P c:/tmp/winpython/  http://www.cmake.org/files/v2.8/cmake-2.8.10.2-win32-x86.exe
c:/tmp/winpython/cmake-2.8.10.2-win32-x86.exe
    

rem distribute a pip
rem
wget -P c:/tmp/winpython/ http://python-distribute.org/distribute_setup.pyp
c:/python27/python c:/tmp/winpython/distribute_setup.py

wget -P c:/tmp/winpython/ https://raw.github.com/pypa/pip/master/contrib/get-pip.py
c:/python27/python c:/tmp/winpython/get-pip.py

rem baliky pro python

c:/python27/scripts/pip install numpy
c:/python27/scripts/pip install scipy
c:/python27/scripts/pip install cython
c:/python27/scripts/pip install matplotlib
