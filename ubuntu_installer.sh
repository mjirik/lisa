#!
SCRIPT_PATH="${BASH_SOURCE[0]}";
cd `dirname ${SCRIPT_PATH}` > /dev/null
#cd ../../../../

# 1. deb package requirements
sudo apt-get install python git python-dev g++ python-numpy python-scipy python-matplotlib python-sklearn python-skimage python-dicom cython python-yaml sox make python-qt4 python-vtk python-setuptools curl

# 2. easy_install requirements simpleITK  
sudo easy_install -U SimpleITK mahotas

# 3. pip install our packages pyseg_base and dicom2fem
sudo pip install pysegbase dicom2fem sed3 sed3 io3d

# 4. install gco_python
mkdir ~/projects
cd ~/projects
git clone https://github.com/mjirik/gco_python.git
cd gco_python
make
sudo python setup.py install
cd ..

# 5. skelet3d - optional for Histology Analyser
sudo apt-get install cmake python-numpy libinsighttoolkit3-dev libpng12-dev
cd ~/projects
git clone https://github.com/mjirik/skelet3d.git
cd skelet3d
mkdir build
cd build
cmake ..
make
sudo make install

cd ~/projects
git clone --recursive -b stable https://github.com/mjirik/lisa.git
cd lisa
python mysetup.py -d

# python src/update_stable.py
# python lisa.py $@

