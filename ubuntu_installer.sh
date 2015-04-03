#!
#
# if script is called with:
#   no argument: stable version is installed 
#   devel: devel version with ssh 
#   any other argument: devel version with https is used
NARGS=$#
ARG1=$1

# echo "$ARG1"
# if [ "$ARG1" = "" ] ; then
#     echo "asdfa"
#     # stable version
# elif [ "$ARG1" = "devel" ] ; then
#     echo "Cloning unstable branch using ssh"
#         # if there is an any argument, install as developer
#         # apt-get install -y sshpass virtualbox
# else
#     echo "Cloning unstable branch using http"
# fi
# exit
cd ~
HOMEDIR="`pwd`"
USER="$(echo `pwd` | sed 's|.*home/\([^/]*\).*|\1|')"

echo "installing for user:"
echo "$USER"

# apt-get update
# apt-get upgrade -y

# 0. deb package requirements
# sudo -u $USER pip install wget --user
# sudo -u $USER python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_apt.txt
wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_apt.txt

sudo apt-get install -y -qq $(grep -vE "^\s*#" requirements_apt.txt | tr "\n" " ")
# apt-get install -y python git python-dev g++ python-numpy python-scipy python-matplotlib python-sklearn python-skimage python-dicom cython python-yaml sox make python-qt4 python-vtk python-setuptools curl python-pip cmake

# 1. conda python packages
if hash conda 2>/dec/null; then
    echo "Conda is installed"
else
    MACHINE_TYPE=`uname -m`
    if [ ${MACHINE_TYPE} == 'x86_64' ]; then
        echo "Installing 64-bit conda"
    # 64-bit stuff here
        wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
        bash Miniconda-latest-Linux-x86_64.sh -b
    else
    # 32-bit stuff here
        echo "Installing 32-bit conda"
        wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86.sh
        bash Miniconda-latest-Linux-x86.sh -b
    fi
    export PATH=$HOMEDIR/miniconda/bin:$PATH
    conda
fi

wget https://raw.githubusercontent.com/mjirik/lisa/master/install.sh
bash install.sh


sudo make install
# sudo -u $USER sh -c "cd ~/projects/skelet3d/build && cmake .. && make"
# sudo -u $USER mkdir build
# sudo -u $USER cd build

# Clone Lisa, make icons
cd
if [ "$ARG1" = "" ] ; then
    echo "Cloning stable version"
    # stable version
    git clone --recursive -b stable https://github.com/mjirik/lisa.git
elif [ $ARG1 -eq "devel" ] ; then
    echo "Cloning unstable branch using ssh"
    # if there is an any argument, install as developer
    # apt-get install -y sshpass virtualbox
    git clone --recursive git@github.com:mjirik/lisa.git
elif [ $ARG1 -eq "noclone" ] ; then
    echo "Just requirements, no git clone"

else
    echo "Cloning unstable branch using http"
    git clone --recursive https://github.com/mjirik/lisa.git

fi
if [ $ARG1 -eq "noclone" ] ; then
    echo "Just requirements"
else
    cd lisa
    python mysetup.py -d
    python mysetup.py -icn

    cd $HOMEDIR
fi
# python src/update_stable.py
# python lisa.py $@

