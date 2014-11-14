#!
#relative path
SCRIPT_PATH="${BASH_SOURCE[0]}";

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
echo $SCRIPTPATH
#cd ../../../../

# echo "Automatic installation of Ubuntu 14.04"
#
# cd ~
# mkdir tmp
# cd tmp
# FILE="ubuntu-14.04.1-desktop-i386.iso"
# URL="http://releases.ubunu.com/trusty/ubuntu-14.04.1-desktop-i386.iso"
# VMNAME="Ubuntu1404_32-bit"
# USERNAME="mjirik"
#
# if [ -f $FILE ];
# then
#     echo "File $FILE exists"
# else
#     echo "Downloading $FILE"
#     wget URL
# fi
# VBoxManage createvm --name $VMNAME --register
# VBoxManage createhd --filename /home/mjirik/tmp/$VMNAME --size 16000
# VBoxManage modifyvm $VMNAME --ostype Ubuntu
# VBoxManage modifyvm $VMNAME --memory 2048
# VBoxManage modifyvm $VMNAME --pae on
# VBoxManage storagectl $VMNAME --name SATA --add sata --controller IntelAhci --bootable on
# VBoxManage storagectl $VMNAME --name IDE --add ide --controller PIIX4 --bootable on
# # echo "Mount HDD and DVD"
# VBoxManage storageattach $VMNAME --storagectl SATA --port 0 --device 0 --type hdd --medium /home/mjirik/tmp/$VMNAME.vdi
# VBoxManage storageattach $VMNAME --storagectl IDE --port 0 --device 0 --type dvddrive --medium "/home/mjirik/tmp/ubuntu-14.04.1-desktop-i386.iso"
# VBoxManage modifyvm $VMNAME --nic1 nat --nictype1 82540EM --cableconnected1 on
# VBoxManage startvm $VMNAME
# echo "end of automatic ubuntu installation"


echo "VirtualBox setup"
cd ~
mkdir tmp
cd tmp

VMNAME="Ubuntu1404_32-bit"
FILE="Ubuntu1404_32-bit.vdi"

FILEURL="http://147.228.240.61/queetech/install/Ubuntu1404_32-bit.vdi"
if [ -f $FILE ];
then
    echo "File $FILE exists"
else
    echo "Downloading $FILE"
    wget $FILEURL
fi

VBoxManage createvm --name $VMNAME --register
VBoxManage modifyvm $VMNAME --ostype Ubuntu
VBoxManage modifyvm $VMNAME --memory 2048
# must set next line for run wvbox on ubuntu 32 bit
VBoxManage modifyvm $VMNAME --pae on
# on wmbox 64 bit it must be setted this way
VBoxManage modifyvm $VMNAME --hwvirtex off
VBoxManage storagectl $VMNAME --name SATA --add sata --controller IntelAhci --bootable on
echo "Mount HDD and DVD"
VBoxManage storageattach $VMNAME --storagectl SATA --port 0 --device 0 --type hdd --medium "/home/mjirik/tmp/Ubuntu1404_32-bit.vdi"
VBoxManage modifyvm $VMNAME --nic1 nat --nictype1 82540EM --cableconnected1 on
VBoxManage modifyvm $VMNAME --natpf1 "guestssh,tcp,,2222,,22"
VBoxManage startvm $VMNAME
echo $?

echo "Waiting for VirtualBox start"
sleep 90

echo "SSH connection to VirtualBox"
echo $SCRIPTPATH
# ssh -p 2222 ubuntu@localhost
cd ${SCRIPTPATH} > /dev/null
# go to directory with install script
cd ..
# add host fingerpritn to avoid :
# Are you suder to continue connecting (yes/no)
# ssh -p 2222 -o StrictHostKeyChecking=no ubuntu@localhost
sshpass -p trustz ssh -p 2222 -o StrictHostKeyChecking=no ubuntu@localhost 'bash -s' < tests/run_this_in_vm.sh
# sshpass -p trustz ssh -p 2222 ubuntu@localhost 


