liver-surgery
=============

Coputer-assisted liver surgery. 

Requirements
------------

    sudo apt-get install python git 

Fallowing packages are installed automatically with mysetup.py

    sudo apt-get install python-numpy python-scipy python-matplotlib python-sklearn python-dicom cython python-yaml


Install
-------

    git clone --recursive git@github.com:mjirik/liver-surgery.git

    python ./mysetup.py

or

    git clone git@github.com:mjirik/liver-surgery.git

    git submodule update --init --recursive

    python ./mysetup.py

You can find more install notes in file 'notes.txt'


Run
---

    cd src 

    python organ_segmentation.py 

for tumor volume measurement use:

    python organ_segmentation.py -mroi -vs 0.8
