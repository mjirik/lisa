#! /bin/sh
#
# run_this_in_vm.sh
# Copyright (C) 2014 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.
#

wget https://raw.githubusercontent.com/mjirik/lisa/master/ubuntu_installer.sh
chmod a+x ubuntu_installer.sh
echo trustz | sudo -S ./ubuntu_installer.sh
rm ubuntu_installer.sh
cd ~/projects/lisa
nosetests
exit $?
