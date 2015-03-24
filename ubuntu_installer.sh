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
HOMEDIR="`pwd`"
USER="$(echo `pwd` | sed 's|.*home/\([^/]*\).*|\1|')"

echo "installing for user:"
echo "$USER"

# apt-get update
# apt-get upgrade -y

# 0. deb package requirements
sudo -u $USER pip install wget --user
sudo -u $USER python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_apt.txt

apt-get install -y -qq $(grep -vE "^\s*#" requirements_apt.txt | tr "\n" " ")
# apt-get install -y python git python-dev g++ python-numpy python-scipy python-matplotlib python-sklearn python-skimage python-dicom cython python-yaml sox make python-qt4 python-vtk python-setuptools curl python-pip cmake

# 1. conda python packages
if hash conda 2>/dec/null; then
    echo "Conda is installed"
else
    sudo -u $USER wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
    sudo -u $USER bash Miniconda-latest-Linux-x86_64.sh -b
    sudo -u $USER export PATH=$HOMEDIR/miniconda/bin:$PATH
fi

python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/install.sh
bash install.sh

sudo make install
# sudo -u $USER sh -c "cd ~/projects/skelet3d/build && cmake .. && make"
# sudo -u $USER mkdir build
# sudo -u $USER cd build

# Clone Lisa, make icons
cd ~/projects
if [ "$ARG1" = "" ] ; then
    echo "Cloning stable version"
    # stable version
    sudo -u $USER git clone --recursive -b stable https://github.com/mjirik/lisa.git
elif [ $ARG1 -eq "devel" ] ; then
    echo "Cloning unstable branch using ssh"
    # if there is an any argument, install as developer
    # apt-get install -y sshpass virtualbox
    sudo -u $USER git clone --recursive git@github.com:mjirik/lisa.git
else
    echo "Cloning unstable branch using http"
    sudo -u $USER git clone --recursive https://github.com/mjirik/lisa.git

fi
cd lisa
sudo -u $USER python mysetup.py -d
sudo -u $USER python mysetup.py -icn

cd $HOMEDIR
# python src/update_stable.py
# python lisa.py $@

