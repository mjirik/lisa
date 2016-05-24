conda install --yes pip
pip install wget
python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_pip.txt -o requirements_pip.txt
python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_conda.txt -o requirements_conda.txt
python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_conda_root.txt -o requirements_conda_root.txt


conda install --yes --file requirements_conda_root.txt
conda install --yes -c SimpleITK -c menpo -c mjirik  --file requirements_conda.txt
# mahotas on luispedro is only for linux
# conda install --yes -c SimpleITK -c luispedro --file requirements_conda.txt

# 2. easy_install requirements simpleITK  
easy_install -U --user mahotas

# 3. pip install our packages pyseg_base and dicom2fem
pip install -U --no-deps -r requirements_pip.txt --user

# linux specific
pip install scikit-fmm

mkdir projects

# 4. install  - it is now installed with pip

cd projects
## mkdir gco_python
## cd gco_python
# git clone https://github.com/mjirik/gco_python.git 
#cd gco_python
## echo `pwd`
# make
# python setup.py install --user
# cd ..
## cd ..


# 5. skelet3d - optional for Histology Analyser
# sudo -u $USER cd ~/projects
# mkdir ~/projects/skelet3d
# mkdir /projects/skelet3d
git clone https://github.com/mjirik/skelet3d.git 
cd skelet3d
mkdir build
cd build
cmake ..
make
