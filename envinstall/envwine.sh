#!/bin/bash 


sudo apt-get install wine
mkdir ~/tmp/winpython
wget -P ~/tmp/winpython/ http://msysgit.googlecode.com/files/Git-1.8.0-preview20121022.exe
echo 'Install for all users with unix command line features'
wine ~/tmp/winpython/Git-1.8.0-preview20121022.exe

wine "mkdir c:/liver-surgery/"
wine "git clone --recurse-submodules git://github.com/mjirik/liver-surgery.git c:/liver/"
wine "c:/liver-surgery/envinstall/envwindows.bat"
